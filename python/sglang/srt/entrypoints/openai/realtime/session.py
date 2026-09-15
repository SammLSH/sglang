"""WebSocket session for realtime ASR.

Pre-commit deltas reference the reserved current_item_id that the
subsequent input_audio_buffer.committed and conversation.item.created
events will announce — sglang-specific, deviates from OpenAI's
commit-only delta emission.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pybase64
from fastapi import WebSocket, WebSocketDisconnect
from openai.types.realtime import (
    ConversationItemCreatedEvent,
    InputAudioBufferAppendEvent,
    InputAudioBufferClearedEvent,
    InputAudioBufferClearEvent,
    InputAudioBufferCommitEvent,
    InputAudioBufferCommittedEvent,
    RealtimeErrorEvent,
    SessionCreatedEvent,
    SessionUpdatedEvent,
)
from openai.types.realtime.conversation_item_input_audio_transcription_completed_event import (
    ConversationItemInputAudioTranscriptionCompletedEvent,
    UsageTranscriptTextUsageDuration,
)
from openai.types.realtime.conversation_item_input_audio_transcription_delta_event import (
    ConversationItemInputAudioTranscriptionDeltaEvent,
)
from openai.types.realtime.conversation_item_input_audio_transcription_failed_event import (
    ConversationItemInputAudioTranscriptionFailedEvent,
)
from openai.types.realtime.conversation_item_input_audio_transcription_failed_event import (
    Error as TranscriptionFailedError,
)
from openai.types.realtime.realtime_conversation_item_user_message import (
    Content as InputAudioContent,
)
from openai.types.realtime.realtime_conversation_item_user_message import (
    RealtimeConversationItemUserMessage,
)
from openai.types.realtime.realtime_error import RealtimeError
from pydantic import BaseModel, ValidationError
from starlette.types import Message

from sglang.srt.entrypoints.openai.protocol import TranscriptionRequest
from sglang.srt.entrypoints.openai.realtime.audio_transcription.audio_buffer import (
    PCM_SAMPLE_WIDTH_BYTES as _SAMPLE_WIDTH,
)
from sglang.srt.entrypoints.openai.realtime.audio_transcription.transcription_processor import (
    RealtimeTranscriptionProcessor,
)
from sglang.srt.entrypoints.openai.realtime.audio_transcription.windowed_transcription import (
    ResolvedEncoderWindowPolicy,
)
from sglang.srt.entrypoints.openai.realtime.protocol import (
    DEFAULT_INPUT_SAMPLE_RATE,
    SUPPORTED_INPUT_SAMPLE_RATES,
    AudioPCM,
    SessionUpdateEvent,
    TranscriptionSessionAudioInput,
    TranscriptionSessionConfig,
)
from sglang.srt.entrypoints.openai.transcription_adapters.base import (
    TranscriptionAdapter,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import random_uuid

logger = logging.getLogger(__name__)

# 60 s * 48,000 samples/s * 2 bytes/sample (PCM16) * 2 for base64/JSON
# overhead = 11,520,000 bytes. Raising the item limit must not grow this backlog.
MAX_PENDING_INPUT_BYTES = 11_520_000

# Conservative per-frame bookkeeping budget, including empty frames.
QUEUED_FRAME_OVERHEAD_BYTES = 256


def _resample_to_target_rate(pcm: bytes, src_rate: int, target_rate: int) -> bytes:
    if src_rate == target_rate or not pcm:
        return pcm
    import torch
    import torchaudio

    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    audio = torch.from_numpy(samples).unsqueeze(0)
    audio = torchaudio.functional.resample(
        audio, orig_freq=src_rate, new_freq=target_rate
    )
    samples = audio.squeeze(0).numpy()
    # Clip to int16 range via 2^15 - 1 so a clipped 1.0 stays representable.
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


_CLIENT_EVENT_TYPES: Dict[str, type] = {
    "session.update": SessionUpdateEvent,
    "input_audio_buffer.append": InputAudioBufferAppendEvent,
    "input_audio_buffer.commit": InputAudioBufferCommitEvent,
    "input_audio_buffer.clear": InputAudioBufferClearEvent,
}


def _parse_client_event(raw: Dict[str, Any]) -> Optional[BaseModel]:
    """Parse, returning None if type is unknown. Raises ValidationError on
    a malformed payload of a known type."""
    cls = _CLIENT_EVENT_TYPES.get(raw.get("type"))
    if cls is None:
        return None
    return cls.model_validate(raw)


@dataclass
class _SessionConfig:
    """Session-level configuration negotiated via session.update: audio
    input format, requested language, sampling params. Persists until
    the session ends; ``configured`` gates audio-frame handling so the
    server doesn't run inference on PCM sent before session.update."""

    input_sample_rate: int = DEFAULT_INPUT_SAMPLE_RATE
    language: Optional[str] = None
    client_model: Optional[str] = None
    sampling_params: Optional[Dict[str, Any]] = None
    configured: bool = False


@dataclass
class _ItemState:
    """Per-item conversation ids. current_item_id is reserved at
    __init__ and only announced to the client by
    input_audio_buffer.committed."""

    current_item_id: str
    previous_item_id: Optional[str] = None


class RealtimeConnection:
    """One realtime transcription session. Drives the WS receive loop,
    dispatches typed client events to the matching _on_* handler, and
    triggers chunked ASR inference at audio buffer thresholds."""

    def __init__(
        self,
        websocket: WebSocket,
        tokenizer_manager: TokenizerManager,
        adapter: TranscriptionAdapter,
        server_args: ServerArgs,
        encoder_window: Optional[ResolvedEncoderWindowPolicy] = None,
    ) -> None:
        self.websocket = websocket
        self.tokenizer_manager = tokenizer_manager
        self.adapter = adapter
        self.server_args = server_args

        self.session_id = f"sess_{random_uuid()}"
        self._current_client_event_id: Optional[str] = None

        self.model_sample_rate = adapter.model_sample_rate
        self.bytes_per_second = self.model_sample_rate * _SAMPLE_WIDTH

        self.config = _SessionConfig()
        self.transcription_processor = RealtimeTranscriptionProcessor(
            tokenizer_manager=tokenizer_manager,
            adapter=adapter,
            encoder_window=encoder_window,
            session_id=self.session_id,
        )
        self.transcription_state = self.transcription_processor.create_state()

        self.item = _ItemState(current_item_id=f"item_{random_uuid()}")
        self.incoming_messages: asyncio.Queue[tuple[Message, int]] = asyncio.Queue()
        self.pending_input_bytes = 0

    async def run(self) -> None:
        await self._send(
            SessionCreatedEvent(
                event_id=f"event_{random_uuid()}",
                type="session.created",
                session=self._build_session_info(),
            )
        )

        try:
            # Keep receiving while inference waits, so disconnects and backlog
            # overflow can cancel the active backend request promptly.
            receiver_task = asyncio.create_task(self.receive_messages())
            consumer_task = asyncio.create_task(self._run_loop())
            overflow = False
            try:
                done, _ = await asyncio.wait(
                    (receiver_task, consumer_task), return_when=asyncio.FIRST_COMPLETED
                )
                if receiver_task in done:
                    overflow = receiver_task.result()
                else:
                    consumer_task.result()
            finally:
                for task in (receiver_task, consumer_task):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(
                    receiver_task, consumer_task, return_exceptions=True
                )
                self.incoming_messages = asyncio.Queue()
                self.pending_input_bytes = 0

            if overflow:
                # Overflow belongs to the connection, not the interrupted event.
                self._current_client_event_id = None
                await self._send_error_and_close(
                    "buffer_overflow",
                    "Pending realtime input exceeded the session buffer limit; "
                    "client is sending faster than inference can keep up",
                    close_code=1009,
                )
        except WebSocketDisconnect:
            logger.info("[realtime] client disconnected: %s", self.session_id)
        except Exception:
            logger.exception("[realtime] unexpected error: %s", self.session_id)
            try:
                await self._send_error(
                    "inference_failed",
                    "Internal server error",
                    error_type="server_error",
                )
            except (WebSocketDisconnect, RuntimeError) as e:
                logger.debug(
                    "[realtime] failed to notify client of unexpected error: %s",
                    e,
                )

    async def receive_messages(self) -> bool:
        """Queue raw WebSocket frames; return True if the backlog exceeds its cap."""
        while True:
            message = await self.websocket.receive()
            if message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect(code=message.get("code", 1000))
            else:
                size = (
                    len((message.get("text") or "").encode("utf-8"))
                    + len(message.get("bytes") or b"")
                    + QUEUED_FRAME_OVERHEAD_BYTES
                )
            if self.pending_input_bytes + size > MAX_PENDING_INPUT_BYTES:
                # Waiting for queue space would hide disconnects behind audio.
                return True
            else:
                self.pending_input_bytes += size
                self.incoming_messages.put_nowait((message, size))
                await asyncio.sleep(0)

    async def _run_loop(self) -> None:
        """Receive-and-dispatch loop. Validation errors emit an error event
        and continue; fatal append-path errors (buffer overflow, append-time
        inference failure) close the WebSocket and terminate the loop.
        """
        while True:
            self._current_client_event_id = None
            # A nonempty queue and short handlers may otherwise never yield.
            await asyncio.sleep(0)
            message, size = await self.incoming_messages.get()
            self.pending_input_bytes -= size

            text = message.get("text")
            if not text:
                if message.get("bytes") is not None:
                    # OpenAI Realtime is base64 PCM in JSON; binary frames aren't supported.
                    await self._send_error(
                        "invalid_payload",
                        "Binary frames are not supported on /v1/realtime; "
                        "use input_audio_buffer.append with base64 audio.",
                    )
                continue

            try:
                raw = json.loads(text)
            except json.JSONDecodeError:
                await self._send_error("invalid_payload", "Invalid JSON")
                continue
            if not isinstance(raw, dict):
                await self._send_error(
                    "invalid_payload", "Top-level event must be a JSON object"
                )
                continue

            client_event_id = raw.get("event_id")
            if client_event_id is not None and not isinstance(client_event_id, str):
                await self._send_error(
                    "invalid_value",
                    "event_id must be a string or null",
                    param="event_id",
                )
                continue
            self._current_client_event_id = client_event_id
            if not isinstance(raw.get("type"), str):
                await self._send_error(
                    "invalid_value", "type must be a string", param="type"
                )
                continue
            try:
                event = _parse_client_event(raw)
            except ValidationError as e:
                # Report first error only; matches OpenAI server behavior.
                err = e.errors()[0]
                loc = ".".join(str(x) for x in err["loc"])
                await self._send_error(
                    "invalid_value",
                    err.get("msg") or "Invalid payload",
                    param=loc or None,
                )
                continue
            if event is None:
                await self._send_error(
                    "unknown_event",
                    f"Unknown event type: {raw.get('type')!r}",
                )
                continue
            terminate = await self._dispatch(event)
            if terminate:
                return

    async def _dispatch(self, event: BaseModel) -> bool:
        """Returns True if the session should terminate."""
        if isinstance(event, InputAudioBufferAppendEvent):
            return await self._on_input_audio_buffer_append(event)
        if isinstance(event, SessionUpdateEvent):
            await self._on_session_update(event)
        elif isinstance(event, InputAudioBufferCommitEvent):
            await self._on_input_audio_buffer_commit(event)
        elif isinstance(event, InputAudioBufferClearEvent):
            await self._on_input_audio_buffer_clear(event)
        return False

    async def _on_session_update(self, event: SessionUpdateEvent) -> None:
        cfg = event.session

        # Normalize audio to an empty input cfg if absent so downstream
        # `audio.X is not None` reads as a business rule, not an existence check.
        # model_fields_set distinguishes omitted fields from explicit nulls.
        audio = (
            cfg.audio.input if cfg.audio else None
        ) or TranscriptionSessionAudioInput()
        audio_fields = audio.model_fields_set
        transcription = audio.transcription

        # Reject server-side VAD, noise reduction, and transcription prompts.
        if audio.turn_detection is not None:
            await self._send_error(
                "not_supported",
                "Server-side VAD is not implemented; "
                "set audio.input.turn_detection: null and commit explicitly.",
                param="session.audio.input.turn_detection",
            )
            return
        if audio.noise_reduction is not None:
            await self._send_error(
                "not_supported",
                "audio.input.noise_reduction is not supported; set to null.",
                param="session.audio.input.noise_reduction",
            )
            return
        if transcription is not None and transcription.prompt is not None:
            await self._send_error(
                "not_supported",
                "audio.input.transcription.prompt is not supported.",
                param="session.audio.input.transcription.prompt",
            )
            return

        # Omitted transcription fields keep their values; explicit null clears.
        new_client_model = self.config.client_model
        new_language = self.config.language
        if transcription is not None:
            if "model" in transcription.model_fields_set:
                new_client_model = transcription.model
            if "language" in transcription.model_fields_set:
                new_language = transcription.language
        elif "transcription" in audio_fields:
            new_client_model = None
            new_language = None

        # Reject a non-empty model name that differs from the server's model name.
        if (
            new_client_model
            and new_client_model != self.tokenizer_manager.served_model_name
        ):
            await self._send_error(
                "not_supported",
                f"Model {new_client_model!r} is not served by this endpoint "
                f"(serving {self.tokenizer_manager.served_model_name!r}); set "
                f"transcription.model to null or to the server's model name.",
                param="session.audio.input.transcription.model",
            )
            return

        # Reject language changes once this item contains audio or decoded text.
        # The client must commit or clear the item first.
        if new_language != self.config.language and (
            self.transcription_state.has_audio
            or self.transcription_state.has_transcript
        ):
            await self._send_error(
                "invalid_state",
                "Cannot change transcription language while an audio item is "
                "active; commit or clear the current item first.",
                param="session.audio.input.transcription.language",
            )
            return

        # Accept only PCM at 16, 24, or 48 kHz. Omitted format keeps the current
        # rate; null format or a PCM format without a rate resets it to 24 kHz.
        new_rate = self.config.input_sample_rate
        if "format" in audio_fields:
            fmt = audio.format
            if fmt is None:
                new_rate = DEFAULT_INPUT_SAMPLE_RATE
            elif not isinstance(fmt, AudioPCM):
                # G.711 (pcmu / pcma): not implemented.
                await self._send_error(
                    "not_supported",
                    f"audio.input.format.type must be 'audio/pcm'; "
                    f"{fmt.type!r} is not implemented",
                    param="session.audio.input.format.type",
                )
                return
            elif fmt.rate is not None and fmt.rate not in SUPPORTED_INPUT_SAMPLE_RATES:
                await self._send_error(
                    "invalid_value",
                    f"audio.input.format.rate must be one of "
                    f"{SUPPORTED_INPUT_SAMPLE_RATES}, got {fmt.rate}",
                    param="session.audio.input.format.rate",
                )
                return
            else:
                new_rate = fmt.rate or DEFAULT_INPUT_SAMPLE_RATE

            # Reject input rate changes after receiving audio for this item.
            # The client must commit or clear the item first.
            if (
                new_rate != self.config.input_sample_rate
                and self.transcription_state.has_audio
            ):
                await self._send_error(
                    "invalid_state",
                    "Cannot change audio.input.format.rate while audio is "
                    "buffered; commit or clear the current item first.",
                    param="session.audio.input.format.rate",
                )
                return

        # Save the input rate, model name, and language; build decoder sampling
        # parameters and mark the session ready to accept audio frames.
        self.config.input_sample_rate = new_rate
        self.config.client_model = new_client_model
        self.config.language = new_language
        self.config.sampling_params = self.adapter.build_sampling_params(
            TranscriptionRequest(language=self.config.language)
        )
        self.config.configured = True

        # Log ignored include[] options and required audio resampling, then send
        # the current input format, model, and language in session.updated.
        if cfg.include:
            logger.info(
                "[realtime] %s: include[] received but not implemented; ignoring: %s",
                self.session_id,
                cfg.include,
            )
        if self.config.input_sample_rate != self.model_sample_rate:
            logger.info(
                "[realtime] %s configured: resample %d→%d (ratio %.2f), language=%s",
                self.session_id,
                self.config.input_sample_rate,
                self.model_sample_rate,
                self.config.input_sample_rate / self.model_sample_rate,
                self.config.language,
            )
        await self._send(
            SessionUpdatedEvent(
                event_id=f"event_{random_uuid()}",
                type="session.updated",
                session=self._build_session_info(),
            )
        )

    async def _on_input_audio_buffer_append(
        self, event: InputAudioBufferAppendEvent
    ) -> bool:
        """Returns True if the session should terminate (buffer overflow or
        append-time inference failure)."""
        if not self.config.configured:
            await self._send_error(
                "invalid_state", "Send session.update before audio frames"
            )
            return False

        # Empty audio is a no-op (heartbeat frames); skip b64decode.
        if not event.audio:
            return False

        try:
            data = pybase64.b64decode(event.audio, validate=True)
        except (ValueError, TypeError):
            await self._send_error(
                "invalid_audio", "audio field is not valid base64", param="audio"
            )
            return False

        if len(data) % _SAMPLE_WIDTH != 0:
            await self._send_error(
                "invalid_audio_format",
                f"PCM16 frame length must be a multiple of {_SAMPLE_WIDTH} bytes",
            )
            return False

        # Estimate post-resample size before resampling so oversized frames fail early.
        src_samples = len(data) // _SAMPLE_WIDTH
        target_samples = math.ceil(
            src_samples * self.model_sample_rate / self.config.input_sample_rate
        )
        if (
            self.transcription_state.audio.received_bytes
            + target_samples * _SAMPLE_WIDTH
            > self.transcription_processor.max_buffer_bytes
        ):
            await self._send_error_and_close(
                "buffer_overflow",
                "Audio item exceeded "
                f"{self.transcription_processor.max_buffer_bytes / self.bytes_per_second:g}s",
                close_code=1009,
            )
            return True

        if self.config.input_sample_rate != self.model_sample_rate:
            data = await asyncio.to_thread(
                _resample_to_target_rate,
                data,
                self.config.input_sample_rate,
                self.model_sample_rate,
            )
        self.transcription_state.audio.append_pcm(data)
        # The processor selects the request range; windowing may leave more
        # inference chunks to drain from a single append.
        while self.transcription_processor.is_chunk_ready(self.transcription_state):
            ok = await self._run_inference(is_last=False)
            if not ok:
                # Stream already finished inside _run_inference.
                return True
        return False

    async def _on_input_audio_buffer_commit(
        self, event: InputAudioBufferCommitEvent
    ) -> None:
        if not self.config.configured:
            await self._send_error("invalid_state", "Send session.update before commit")
            return
        if (
            not self.transcription_state.has_audio
            and not self.transcription_state.has_transcript
        ):
            await self._send_error(
                "invalid_state", "Cannot commit an empty audio buffer"
            )
            return

        # A skipped or truncated intermediate decode leaves its audio unprocessed,
        # forcing one final decode so the item either recovers or fails closed.
        has_new_audio = self.transcription_state.has_new_audio
        item_id = self.item.current_item_id
        prev_item_id = self.item.previous_item_id

        partial_transcript = self.transcription_state.emitted_text

        await self._send(
            InputAudioBufferCommittedEvent(
                event_id=f"event_{random_uuid()}",
                type="input_audio_buffer.committed",
                item_id=item_id,
                previous_item_id=prev_item_id,
            )
        )
        await self._send(
            ConversationItemCreatedEvent(
                event_id=f"event_{random_uuid()}",
                type="conversation.item.created",
                previous_item_id=prev_item_id,
                item=RealtimeConversationItemUserMessage(
                    id=item_id,
                    type="message",
                    role="user",
                    status="completed",
                    content=[
                        InputAudioContent(
                            type="input_audio", transcript=partial_transcript
                        )
                    ],
                ),
            )
        )

        # Capture pcm duration before `_start_next_item()` runs: starting
        # the next item resets the audio state, so reading it after gives 0.
        pcm_duration_seconds = (
            self.transcription_state.audio.received_bytes / self.bytes_per_second
        )

        if has_new_audio or self.transcription_state.has_transcript:
            ok = await self._run_inference(is_last=True)
            if not ok:
                # _run_inference already emitted transcription.failed and
                # rolled the item; don't also emit completed.
                return

        await self._send(
            ConversationItemInputAudioTranscriptionCompletedEvent(
                event_id=f"event_{random_uuid()}",
                type="conversation.item.input_audio_transcription.completed",
                item_id=item_id,
                content_index=0,
                transcript=self.transcription_state.emitted_text,
                usage=UsageTranscriptTextUsageDuration(
                    type="duration", seconds=pcm_duration_seconds
                ),
            )
        )

        self._start_next_item()

    async def _on_input_audio_buffer_clear(
        self, event: InputAudioBufferClearEvent
    ) -> None:
        # Reserve a fresh current_item_id so post-clear pre-commit deltas
        # don't share an item_id with deltas the client already received
        # for the abandoned audio. previous_item_id is NOT touched — the
        # cleared item was never committed, so the prior-commit chain
        # shouldn't include it.
        self._reset_inference_state()
        self.item.current_item_id = f"item_{random_uuid()}"
        await self._send(
            InputAudioBufferClearedEvent(
                event_id=f"event_{random_uuid()}", type="input_audio_buffer.cleared"
            )
        )

    async def _run_inference(self, is_last: bool) -> bool:
        """Run ASR on the current buffer. Returns False on failure:
        commit-time emits transcription.failed and rolls the item; append-time
        emits a generic error envelope and closes the WebSocket."""
        try:
            if is_last and not self.transcription_state.has_new_audio:
                await self.transcription_processor.flush_pending_transcript(
                    self.transcription_state, self._emit_transcription_delta
                )
            else:
                await self.transcription_processor.process(
                    self.transcription_state,
                    is_last=is_last,
                    sampling_params=self.config.sampling_params,
                    on_transcript_delta=self._emit_transcription_delta,
                )
        except Exception:
            logger.exception(
                "[realtime] inference failed: session=%s item=%s buffer_bytes=%d",
                self.session_id,
                self.item.current_item_id,
                len(self.transcription_state.audio.data),
            )
            if is_last:
                # Commit-time failure: committed + created already emitted,
                # so the item exists client-side and transcription.failed
                # can reference it. Wire message is hardcoded "Transcription
                # failed" — don't leak backend traces to the client; full
                # error is in the logger.exception above.
                await self._send(
                    ConversationItemInputAudioTranscriptionFailedEvent(
                        event_id=f"event_{random_uuid()}",
                        type="conversation.item.input_audio_transcription.failed",
                        item_id=self.item.current_item_id,
                        content_index=0,
                        error=TranscriptionFailedError(
                            type="server_error",
                            code="inference_failed",
                            message="Transcription failed",
                        ),
                    )
                )
                self._start_next_item()
            else:
                # Append-time failure: the item isn't visible client-side
                # yet (committed/created fire at commit), so
                # transcription.failed would reference a ghost id.
                await self._send_error_and_close(
                    "inference_failed",
                    "Transcription failed",
                    close_code=1011,
                )
            return False

        return True

    async def _emit_transcription_delta(self, delta: str) -> None:
        """Wrap the processor's exact append string without reformatting."""
        await self._send(
            ConversationItemInputAudioTranscriptionDeltaEvent(
                event_id=f"event_{random_uuid()}",
                type="conversation.item.input_audio_transcription.delta",
                item_id=self.item.current_item_id,
                content_index=0,
                delta=delta,
            )
        )

    def _start_next_item(self) -> None:
        self.item.previous_item_id = self.item.current_item_id
        self.item.current_item_id = f"item_{random_uuid()}"
        self._reset_inference_state()

    def _reset_inference_state(self) -> None:
        """Missing any of these resets leaks state across items."""
        self.transcription_state = self.transcription_processor.create_state()

    def _build_session_info(self) -> TranscriptionSessionConfig:
        # id / object aren't SDK fields; round-trip via extra='allow' so
        # dumps emit them like the real server.
        return TranscriptionSessionConfig.model_validate(
            {
                "type": "transcription",
                "id": self.session_id,
                "object": "realtime.transcription_session",
                "audio": {
                    "input": {
                        "format": {
                            "type": "audio/pcm",
                            "rate": self.config.input_sample_rate,
                        },
                        "transcription": {
                            "model": self.config.client_model,
                            "language": self.config.language,
                        },
                        "noise_reduction": None,
                        "turn_detection": None,
                    }
                },
            }
        )

    async def _send(self, event: BaseModel) -> None:
        await self.websocket.send_text(event.model_dump_json())

    async def _send_error(
        self,
        code: str,
        message: str,
        *,
        error_type: str = "invalid_request_error",
        param: Optional[str] = None,
    ) -> None:
        envelope = RealtimeErrorEvent(
            event_id=f"event_{random_uuid()}",
            type="error",
            error=RealtimeError(
                type=error_type,
                code=code,
                message=message,
                param=param,
                event_id=self._current_client_event_id,
            ),
        )
        await self.websocket.send_text(envelope.model_dump_json())

    async def _send_error_and_close(
        self,
        code: str,
        message: str,
        *,
        close_code: int,
        error_type: str = "server_error",
    ) -> None:
        # Independent try-blocks: a failed send must not skip the close.
        # We still need to release local starlette socket state even when
        # the wire send doesn't reach the peer.
        try:
            await self._send_error(code, message, error_type=error_type)
        except (WebSocketDisconnect, RuntimeError) as e:
            logger.debug("[realtime] send error %s before close failed: %s", code, e)
        try:
            await self.websocket.close(code=close_code)
        except (WebSocketDisconnect, RuntimeError) as e:
            logger.debug("[realtime] close %d after %s failed: %s", close_code, code, e)
