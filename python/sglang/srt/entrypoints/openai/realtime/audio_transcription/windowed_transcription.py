from __future__ import annotations

import logging
from copy import copy
from typing import Dict, Optional

import msgspec
from msgspec.structs import replace
from pydantic import JsonValue

from sglang.srt.entrypoints.openai.realtime.audio_transcription.audio_buffer import (
    PCM_SAMPLE_WIDTH_BYTES,
    extract_audio_samples,
)
from sglang.srt.entrypoints.openai.realtime.audio_transcription.transcription_mode import (
    TranscriptCandidateCallback,
    TranscriptionMode,
    TranscriptionOutcome,
    TranscriptionRequest,
)
from sglang.srt.entrypoints.openai.realtime.audio_transcription.transcription_state import (
    RealtimeTranscriptionState,
    WindowedTranscriptionState,
)
from sglang.srt.entrypoints.openai.realtime.audio_transcription.transcription_suffix import (
    TranscriptionSuffixState,
)
from sglang.srt.entrypoints.openai.streaming_asr import (
    TranscriptionBackendAborted,
    generate_transcript,
    iter_text_unit_spans,
)
from sglang.srt.entrypoints.openai.transcription_adapters.base import (
    RealtimeEncoderWindowPolicy,
    TranscriptionAdapter,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.multimodal.encoder_window import (
    EncoderWindowConfig,
    EncoderWindowMixin,
)
from sglang.srt.runtime_context import get_serving

logger = logging.getLogger(__name__)

# Stop retrying after two failures so an audio item cannot retain PCM indefinitely.
MAX_CONSECUTIVE_WINDOW_FAILURES = 2


class ResolvedEncoderWindowPolicy(msgspec.Struct, frozen=True):
    """Combine the model's audio window sizes with realtime transcription settings."""

    config: EncoderWindowConfig
    policy: RealtimeEncoderWindowPolicy

    @property
    def window_bytes(self) -> int:
        return self.config.window_samples * PCM_SAMPLE_WIDTH_BYTES

    @property
    def leading_context_bytes(self) -> int:
        return self.config.leading_context_samples * PCM_SAMPLE_WIDTH_BYTES


class WindowedTranscriptionMode(TranscriptionMode):
    """Reuse recent audio and confirmed text instead of decoding the entire recording."""

    def __init__(
        self,
        tokenizer_manager: TokenizerManager,
        adapter: TranscriptionAdapter,
        *,
        encoder_window: ResolvedEncoderWindowPolicy,
    ) -> None:
        super().__init__(tokenizer_manager, adapter)
        self.encoder_window = encoder_window
        self.unfixed_text_units = int(
            self.chunked_streaming_config["unfixed_token_num"]
        )
        # Retain the context, mutable tail, and one extra window for retries.
        self.max_retained_audio_bytes = (
            encoder_window.policy.max_audio_context_windows + 2
        ) * encoder_window.window_bytes

    def create_state(self) -> WindowedTranscriptionState:
        return WindowedTranscriptionState(suffix=TranscriptionSuffixState())

    def build_transcription_request(
        self,
        state: RealtimeTranscriptionState,
        *,
        is_last: bool,
    ) -> TranscriptionRequest:
        """Select the audio range and decoder text for windowed transcription."""
        mode_state = state.mode_state
        assert isinstance(mode_state, WindowedTranscriptionState)
        suffix_state = mode_state.suffix
        # Bound each window request even when one append contains many chunks.
        audio = state.audio
        end_offset_bytes = (
            audio.received_bytes
            if is_last
            else min(
                audio.received_bytes,
                audio.last_attempted_offset_bytes + self.chunk_size_bytes,
            )
        )
        start_offset_bytes = self.get_audio_start_offset_bytes(state, end_offset_bytes)
        decoder_prefix = suffix_state.build_decoder_prefix(
            self.tokenizer_manager.tokenizer,
            self.encoder_window.policy.decoder_prefix_max_tokens,
            emitted_text=state.emitted_text,
        )
        if (
            state.emitted_text or suffix_state.confirmed_pending_text_chars
        ) and not decoder_prefix:
            raise RuntimeError(
                "realtime ASR decoder prefix budget cannot retain a complete text unit"
            )
        else:
            return TranscriptionRequest(
                is_last=is_last,
                start_offset_bytes=start_offset_bytes,
                end_offset_bytes=end_offset_bytes,
                last_attempted_offset_bytes=audio.last_attempted_offset_bytes,
                last_processed_offset_bytes=audio.last_processed_offset_bytes,
                emitted_text=state.emitted_text,
                mode_state=mode_state,
                leading_context_bytes=min(
                    self.encoder_window.leading_context_bytes,
                    start_offset_bytes - audio.base_offset_bytes,
                ),
                decoder_prefix=decoder_prefix,
            )

    def get_audio_start_offset_bytes(
        self, state: RealtimeTranscriptionState, end_offset_bytes: int
    ) -> int:
        """Start at a complete encoder window and retain audio with unconfirmed text."""
        mode_state = state.mode_state
        assert isinstance(mode_state, WindowedTranscriptionState)
        audio = state.audio
        window_bytes = self.encoder_window.window_bytes
        complete_windows_end_offset_bytes = (
            end_offset_bytes // window_bytes * window_bytes
        )
        context_windows = self.encoder_window.policy.max_audio_context_windows
        start_offset_bytes = min(
            complete_windows_end_offset_bytes - context_windows * window_bytes,
            audio.last_processed_offset_bytes // window_bytes * window_bytes,
        )
        # The buffer still includes samples needed to extract the first audio frame.
        if audio.base_offset_bytes == 0:
            earliest_start_offset_bytes = 0
        else:
            context_end_offset_bytes = (
                audio.base_offset_bytes + self.encoder_window.leading_context_bytes
            )
            earliest_start_offset_bytes = (
                -(-context_end_offset_bytes // window_bytes) * window_bytes
            )
        # A short result after a retry may cover little audio, so remove at most
        # one encoder window at a time.
        start_offset_bytes = max(
            earliest_start_offset_bytes,
            min(start_offset_bytes, earliest_start_offset_bytes + window_bytes),
        )
        start_offset_bytes = (
            min(start_offset_bytes, mode_state.unconfirmed_audio_start_offset_bytes)
            if mode_state.unconfirmed_audio_start_offset_bytes is not None
            else start_offset_bytes
        )
        if end_offset_bytes - start_offset_bytes > self.max_retained_audio_bytes:
            raise RuntimeError(
                "realtime ASR could not confirm text within the retained audio limit"
            )
        else:
            return start_offset_bytes

    async def transcribe_audio(
        self,
        state: RealtimeTranscriptionState,
        request: TranscriptionRequest,
        *,
        sampling_params: Dict[str, JsonValue],
        on_transcript_candidate: Optional[TranscriptCandidateCallback],
    ) -> TranscriptionOutcome:
        """Prepare transcript updates while keeping state unchanged until text is sent."""
        previous_mode_state = request.mode_state
        assert isinstance(previous_mode_state, WindowedTranscriptionState)
        previous_suffix_state = previous_mode_state.suffix
        # Complete encoder windows can reuse cached encoder output across requests.
        assert request.start_offset_bytes % self.encoder_window.window_bytes == 0
        samples = await extract_audio_samples(
            state.audio,
            request.start_offset_bytes - request.leading_context_bytes,
            request.end_offset_bytes,
        )

        async def _send_partial_transcript(text: str) -> None:
            assert on_transcript_candidate is not None
            text_unit_spans = list(iter_text_unit_spans(text))
            if len(text_unit_spans) < 2:
                return
            else:
                # The decoder may still extend the last word; keep it unsent.
                complete_text = text[: text_unit_spans[-2][1]]
            transcript_candidate = previous_suffix_state.prepare_transcript_update(
                complete_text,
                is_last=request.is_last,
                holdback_units=self.encoder_window.policy.decoder_prefix_holdback_units,
                unfixed_units=self.unfixed_text_units,
            ).delta
            await on_transcript_candidate(transcript_candidate)

        try:
            generated_transcript = await generate_transcript(
                tokenizer_manager=self.tokenizer_manager,
                adapter=self.adapter,
                audio_data=samples,
                sampling_params=sampling_params,
                decoder_prefix=request.decoder_prefix,
                mm_processor_kwargs={
                    "encoder_window": {
                        "leading_context_samples": request.leading_context_bytes
                        // PCM_SAMPLE_WIDTH_BYTES,
                    }
                },
                on_transcript_update=_send_partial_transcript
                if on_transcript_candidate is not None
                else None,
            )
        except TranscriptionBackendAborted as error:
            return self.handle_transcription_failure(state, request, error)
        if generated_transcript is None:
            return self.handle_transcription_failure(
                state,
                request,
                RuntimeError("realtime ASR request returned no response"),
            )
        elif generated_transcript.finish_reason == "length":
            return self.handle_transcription_failure(
                state,
                request,
                RuntimeError("realtime ASR decode reached max_new_tokens"),
            )
        else:
            return self.prepare_transcription_outcome(
                request, generated_transcript.text
            )

    def handle_transcription_failure(
        self,
        state: RealtimeTranscriptionState,
        request: TranscriptionRequest,
        error: RuntimeError,
    ) -> TranscriptionOutcome:
        """Keep audio for retries; fail if retrying could revise already sent text."""
        if (
            request.is_last
            or state.emitted_text != request.emitted_text
            or (isinstance(error, TranscriptionBackendAborted) and not error.retryable)
        ):
            raise error
        else:
            logger.warning(
                "[realtime] encoder-window ASR request failed (%s); retaining audio",
                error,
            )
        previous_mode_state = request.mode_state
        assert isinstance(previous_mode_state, WindowedTranscriptionState)
        failure_count = previous_mode_state.consecutive_failures + 1
        if failure_count >= MAX_CONSECUTIVE_WINDOW_FAILURES:
            raise RuntimeError("encoder-window ASR failed repeatedly") from error
        else:
            next_mode_state = replace(
                previous_mode_state, consecutive_failures=failure_count
            )
        # Keep failed audio in the buffer so the next request can retry it.
        return TranscriptionOutcome(
            next_mode_state=next_mode_state, audio_processed=False
        )

    def prepare_transcription_outcome(
        self,
        request: TranscriptionRequest,
        text: str,
    ) -> TranscriptionOutcome:
        """Decide which text can be sent and which audio needs another transcription."""
        previous_mode_state = request.mode_state
        assert isinstance(previous_mode_state, WindowedTranscriptionState)
        suffix_state = previous_mode_state.suffix
        update = suffix_state.prepare_transcript_update(
            text,
            is_last=request.is_last,
            holdback_units=self.encoder_window.policy.decoder_prefix_holdback_units,
            unfixed_units=self.unfixed_text_units,
        )
        if (
            request.is_last
            and request.end_offset_bytes > request.last_processed_offset_bytes
            and len(suffix_state.pending_text)
            > suffix_state.confirmed_pending_text_chars
            and not text
        ):
            raise RuntimeError("final realtime ASR recovery returned empty text")
        elif (
            update.empty_generated_text
            and not request.is_last
            and request.end_offset_bytes > request.last_processed_offset_bytes
            and not previous_mode_state.retrying_empty_text
        ):
            # Retry an empty result once before treating it as silence.
            next_mode_state = replace(
                previous_mode_state,
                retrying_empty_text=True,
                unconfirmed_audio_start_offset_bytes=request.start_offset_bytes,
            )
            return TranscriptionOutcome(
                next_mode_state=next_mode_state, audio_processed=False
            )
        else:
            # Confirmed but unsent text remains available in the next decoder prompt.
            audio_processed = (
                request.is_last
                or bool(update.delta)
                or len(update.pending_text) == update.confirmed_pending_text_chars
            )
        return TranscriptionOutcome(
            next_mode_state=WindowedTranscriptionState(
                suffix=TranscriptionSuffixState(
                    pending_text=update.pending_text,
                    confirmed_pending_text_chars=update.confirmed_pending_text_chars,
                ),
                unconfirmed_audio_start_offset_bytes=None
                if audio_processed
                else request.start_offset_bytes,
            ),
            audio_processed=audio_processed,
            delta=update.delta,
            trim_buffer_offset_bytes=(
                max(
                    0,
                    request.start_offset_bytes
                    - self.encoder_window.leading_context_bytes,
                )
                if audio_processed and not request.is_last
                else None
            ),
        )

    def flush_pending_transcript(
        self, state: RealtimeTranscriptionState
    ) -> TranscriptionOutcome:
        mode_state = state.mode_state
        assert isinstance(mode_state, WindowedTranscriptionState)
        suffix = copy(mode_state.suffix)
        delta = suffix.flush_pending_transcript()
        return TranscriptionOutcome(
            next_mode_state=replace(mode_state, suffix=suffix),
            audio_processed=False,
            delta=delta,
        )


def resolve_realtime_encoder_window_policy(
    *,
    adapter: TranscriptionAdapter,
    tokenizer_manager: TokenizerManager,
) -> Optional[ResolvedEncoderWindowPolicy]:
    """Check model support and apply server settings before accepting audio sessions."""
    serving_config = get_serving()
    mm_processor = tokenizer_manager.mm_processor
    tokenizer = tokenizer_manager.tokenizer

    policy = adapter.realtime_encoder_window_policy
    if policy is None or not isinstance(mm_processor, EncoderWindowMixin):
        logger.warning(
            "[realtime] --enable-asr-encoder-window is set but the model or its "
            "transcription adapter does not declare encoder windowing; realtime "
            "transcription stays cumulative"
        )
        return None
    else:
        overrides = {
            "max_audio_context_windows": serving_config.asr_encoder_window_max_context_windows,
            "decoder_prefix_max_tokens": serving_config.asr_decoder_prefix_max_tokens,
            "decoder_prefix_holdback_units": serving_config.asr_decoder_prefix_holdback_units,
        }
    overrides = {name: value for name, value in overrides.items() if value is not None}
    policy = replace(policy, **overrides) if overrides else policy
    if tokenizer is None:
        raise RuntimeError(
            "encoder-window ASR requires a tokenizer for the decoder prefix"
        )
    else:
        config = mm_processor.encoder_window_config()
    if config.sample_rate != adapter.model_sample_rate:
        raise ValueError(
            f"feature extractor sample rate {config.sample_rate} differs from the "
            f"model sample rate {adapter.model_sample_rate}"
        )
    else:
        resolved = ResolvedEncoderWindowPolicy(config=config, policy=policy)
    logger.info(
        "[realtime] encoder windows enabled from the first request: "
        "%.1f s windows of %d tokens, max_context_windows %d, "
        "prefix_tokens %d, holdback_units %d",
        config.window_seconds,
        config.window_tokens,
        policy.max_audio_context_windows,
        policy.decoder_prefix_max_tokens,
        policy.decoder_prefix_holdback_units,
    )
    return resolved
