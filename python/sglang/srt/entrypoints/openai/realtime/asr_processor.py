"""Application service for realtime ASR inference.

The processor is connection-scoped but keeps no stream progress itself. Each
call builds and executes one backend transcription step, then commits its
audio outcome to the explicit ``RealtimeASRState`` supplied by the endpoint.

Per-chunk flow, driven by the endpoint::

    append audio -> is_chunk_ready()? -> process()
        _build_transcription_step  resolve mode, audio range, and prompt
        _execute_*_step            compute candidate text without changing state
        preview.remaining         validate the final candidate against sent text
        _commit_outcome            adopt text, advance cursors, compact old PCM
    -> transcript delta
    commit event -> process(is_last=True), or flush_pending_transcript()
                    if no new audio

Mode lifecycle: requests stay cumulative until the adapter's activation
threshold is crossed. Eligible items then switch, once and one-way, to
encoder-aligned audio windows reused through the embedding cache, plus a
bounded decoder prefix so only the new suffix is decoded. Only languages the
adapter declares may enter this mode.
"""

from __future__ import annotations

import asyncio
import logging
import math
from copy import copy
from typing import Any, Awaitable, Callable, Dict, Optional

import msgspec
import numpy as np

from sglang.srt.entrypoints.openai.realtime.audio_buffer import (
    PCM_SAMPLE_WIDTH_BYTES,
    AudioBuffer,
    pcm_to_float_samples,
)
from sglang.srt.entrypoints.openai.realtime.decoder_suffix import (
    DecoderSuffixState,
    SuffixUpdate,
)
from sglang.srt.entrypoints.openai.realtime.encoder_window_policy import (
    ResolvedEncoderWindowPolicy,
)
from sglang.srt.entrypoints.openai.streaming_asr import (
    ASRBackendAborted,
    StreamingASRState,
    apply_cumulative_transcript,
    common_unit_prefix,
    generate_asr_transcript,
    join_text,
    join_units,
    needs_space,
    split_units,
)
from sglang.srt.entrypoints.openai.transcription_adapters.base import (
    TranscriptionAdapter,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.multimodal.encoder_window import build_audio_window_processor_kwargs
from sglang.srt.runtime_context import get_serving

logger = logging.getLogger(__name__)

# Consecutive windowed requests that may fail to produce a usable decode
# (abort, no response, truncation) before the item stops trying windows.
_MAX_CONSECUTIVE_WINDOW_FAILURES = 2

_TranscriptDeltaCallback = Callable[[str], Awaitable[None]]


class _StreamPreview(msgspec.Struct):
    """Text published to the client while one backend decode is running.

    ``streamed`` is the candidate delta already on the wire for this step;
    every later publication must extend it at a valid wire boundary.
    """

    streamed: str = ""

    def remaining(self, candidate: str) -> Optional[str]:
        """Return the appendable suffix, or None if publication would revise text."""
        if not candidate.startswith(self.streamed):
            return None
        suffix = candidate[len(self.streamed) :]
        # The session inserts spaces between separately emitted words. A
        # mid-word extension ("two" -> "twofold") would become "two fold".
        if (
            self.streamed
            and suffix
            and not suffix.startswith(" ")
            and needs_space(self.streamed, suffix)
        ):
            return None
        return suffix.lstrip(" ")


class RealtimeASRState(msgspec.Struct):
    """Mutable per-item progress shared by the endpoint and the processor.

    ``transcript`` owns cumulative reconciliation until a successful one-way
    handoff. ``decoder_suffix`` is created when that handoff commits, so its
    presence is the source of truth that encoder windowing is active.
    """

    audio: AudioBuffer
    transcript: StreamingASRState
    decoder_suffix: Optional[DecoderSuffixState] = None
    # One-way per-item fallback to cumulative decoding, taken only before the
    # handoff commits (afterwards old PCM is gone and there is no way back).
    encoder_window_disabled: bool = False
    consecutive_window_failures: int = 0
    # An empty windowed continuation is retried once before it is accepted.
    deferred_empty_continuation: bool = False
    # Freeze the request start while decoded text is still unconfirmed.
    unconfirmed_window_start: Optional[int] = None

    @property
    def has_audio(self) -> bool:
        return self.audio.received_bytes > 0

    @property
    def has_transcript(self) -> bool:
        if self.decoder_suffix is not None:
            return bool(self.decoder_suffix.emitted_text or self.decoder_suffix.pending)
        return bool(self.transcript.full_transcript)

    @property
    def emitted_text(self) -> str:
        """Everything published to the client for this item."""
        if self.decoder_suffix is not None:
            return self.decoder_suffix.emitted_text
        return self.transcript.emitted_text

    @property
    def has_new_audio(self) -> bool:
        return self.audio.received_bytes > self.audio.last_processed_offset_bytes

    @property
    def encoder_window_active(self) -> bool:
        return self.decoder_suffix is not None


class _TranscriptionStep(msgspec.Struct, frozen=True):
    """Immutable backend-request snapshot built before generation awaits.

    Execution and commit consume the same instance so the selected mode, PCM
    range, and decoder context cannot be recomputed from state changed while
    the request was in flight.
    """

    is_last: bool
    uses_encoder_windows: bool
    start_offset_bytes: int
    end_offset_bytes: int
    # Audio before start_offset_bytes sent along as extraction context for
    # the first window; never part of the window itself.
    leading_context_bytes: int = 0
    decoder_prefix: str = ""
    # Cumulative text decoded but not yet published when the item hands off
    # to windowing; it seeds the pending suffix so Local Agreement can
    # confirm it against the first windowed decode.
    handoff_pending: str = ""


class _TranscriptionOutcome(msgspec.Struct, frozen=True):
    """Result contract consumed when a transcription step commits.

    ``audio_processed=False`` retains the covered audio for the next step.
    Text updates stay unapplied until publication is validated and committed.
    """

    audio_processed: bool
    delta: str = ""
    transcript: Optional[StreamingASRState] = None
    suffix_update: Optional[SuffixUpdate] = None
    # The step produced no usable decode; counts toward the fallback limit.
    window_failed: bool = False
    # The step deferred an empty continuation for one more request.
    deferred_empty: bool = False


class RealtimeASRProcessor:
    """Build and execute realtime ASR steps against explicit stream state."""

    def __init__(
        self,
        tokenizer_manager: TokenizerManager,
        adapter: TranscriptionAdapter,
        *,
        encoder_window: Optional[ResolvedEncoderWindowPolicy] = None,
        session_id: str = "",
    ) -> None:
        self.tokenizer_manager = tokenizer_manager
        self.adapter = adapter
        self.pcm_bytes_per_second = adapter.model_sample_rate * PCM_SAMPLE_WIDTH_BYTES

        state = StreamingASRState(**self.adapter.chunked_streaming_config)
        if not math.isfinite(state.chunk_size_sec) or state.chunk_size_sec <= 0:
            raise ValueError("realtime ASR chunk_size_sec must be finite and positive")
        self.chunk_size_bytes = int(state.chunk_size_sec * self.pcm_bytes_per_second)
        if self.chunk_size_bytes <= 0:
            raise ValueError(
                "realtime ASR chunk_size_sec is shorter than one PCM sample"
            )
        if self.chunk_size_bytes % PCM_SAMPLE_WIDTH_BYTES:
            raise ValueError(
                "realtime ASR chunk_size_sec must resolve to whole PCM samples"
            )
        self.max_buffer_bytes = (
            get_serving().asr_max_buffer_seconds * self.pcm_bytes_per_second
        )
        self.decoder_streaming = bool(get_serving().enable_asr_decoder_streaming)

        self.encoder_window = encoder_window
        self.activation_threshold_bytes: Optional[int] = None
        self.routed_dp_rank: Optional[int] = None
        if encoder_window is not None:
            if encoder_window.window_bytes <= 0:
                raise ValueError("invalid audio encoder-window config")
            self.activation_threshold_bytes = encoder_window.activation_threshold_bytes(
                self.chunk_size_bytes, state.chunk_size_sec
            )
            if session_id:
                self.routed_dp_rank = encoder_window.pinned_dp_rank(session_id)

    def create_state(self) -> RealtimeASRState:
        return RealtimeASRState(
            audio=AudioBuffer(),
            transcript=StreamingASRState(**self.adapter.chunked_streaming_config),
        )

    def is_chunk_ready(self, state: RealtimeASRState) -> bool:
        """True once a full chunk of audio has arrived past the last attempt."""
        return (
            state.audio.received_bytes - state.audio.last_attempted_offset_bytes
            >= self.chunk_size_bytes
        )

    async def process(
        self,
        state: RealtimeASRState,
        *,
        is_last: bool,
        language: Optional[str],
        sampling_params: Dict[str, Any],
        on_transcript_delta: Optional[_TranscriptDeltaCallback] = None,
    ) -> str:
        """Run one transcription step and return its publishable delta.

        With decoder streaming enabled and a callback supplied, append-safe
        parts of the delta are published through the callback while the
        decode runs; the returned string is only the remainder.
        """
        if (
            state.encoder_window_active
            and self.encoder_window is not None
            and not self.encoder_window.supports_language(language)
        ):
            raise RuntimeError(
                "realtime ASR language cannot change after encoder windowing starts"
            )
        step = self._build_transcription_step(state, is_last, language)
        preview: Optional[_StreamPreview] = None
        if on_transcript_delta is not None and self.decoder_streaming and not is_last:
            preview = _StreamPreview()
        execute_step = (
            self._execute_encoder_window_step
            if step.uses_encoder_windows
            else self._execute_cumulative_step
        )
        outcome = await execute_step(
            state, step, sampling_params, preview, on_transcript_delta
        )
        remaining = outcome.delta
        if preview is not None:
            remaining = preview.remaining(outcome.delta)
            if remaining is None:
                raise RuntimeError("completed ASR decode revised already streamed text")
        self._commit_outcome(state, step, outcome)
        return remaining

    def flush_pending_transcript(self, state: RealtimeASRState) -> str:
        """Emit text still held back when the item commits without new audio."""
        if state.decoder_suffix is not None:
            return state.decoder_suffix.flush()
        return state.transcript.finalize()

    # ---- step construction -------------------------------------------------

    def _build_transcription_step(
        self,
        state: RealtimeASRState,
        is_last: bool,
        language: Optional[str],
    ) -> _TranscriptionStep:
        """Resolve the mode, audio range, and text context for the next call.

        Intermediate steps grow by one chunk past the last attempt so a large
        append is drained at the normal cadence; the final step covers every
        received byte.
        """
        audio = state.audio
        end_offset_bytes = (
            audio.received_bytes
            if is_last
            else min(
                audio.received_bytes,
                audio.last_attempted_offset_bytes + self.chunk_size_bytes,
            )
        )
        if not self._should_use_encoder_windows(
            state, is_last=is_last, language=language, end_offset_bytes=end_offset_bytes
        ):
            return _TranscriptionStep(
                is_last=is_last,
                uses_encoder_windows=False,
                start_offset_bytes=0,
                end_offset_bytes=end_offset_bytes,
            )

        assert self.encoder_window is not None
        max_prefix_tokens = self.encoder_window.policy.decoder_prefix_max_tokens
        if state.decoder_suffix is None:
            # Handoff: send the full audio once. The prefix is everything
            # published so far; the cumulative text decoded but not yet
            # published becomes the first pending suffix.
            suffix_state = DecoderSuffixState(
                emitted_text=state.transcript.emitted_text
            )
            handoff_pending = _cumulative_unpublished_text(state.transcript)
            start_offset_bytes = 0
        else:
            suffix_state = state.decoder_suffix
            handoff_pending = ""
            start_offset_bytes = self._encoder_window_start_offset(
                state, end_offset_bytes
            )
        decoder_prefix = suffix_state.bounded_prefix(
            self.tokenizer_manager.tokenizer, max_prefix_tokens
        )
        if suffix_state.emitted_text and not decoder_prefix:
            raise RuntimeError(
                "realtime ASR decoder prefix budget cannot retain a complete text unit"
            )
        return _TranscriptionStep(
            is_last=is_last,
            uses_encoder_windows=True,
            start_offset_bytes=start_offset_bytes,
            end_offset_bytes=end_offset_bytes,
            leading_context_bytes=min(
                self.encoder_window.context_bytes,
                start_offset_bytes - state.audio.base_offset_bytes,
            ),
            decoder_prefix=decoder_prefix,
            handoff_pending=handoff_pending,
        )

    def _should_use_encoder_windows(
        self,
        state: RealtimeASRState,
        *,
        is_last: bool,
        language: Optional[str],
        end_offset_bytes: int,
    ) -> bool:
        if self.encoder_window is None or state.encoder_window_disabled:
            return False
        if state.encoder_window_active:
            return True
        return (
            not is_last
            and self.encoder_window.supports_language(language)
            and end_offset_bytes > self.activation_threshold_bytes
        )

    def _encoder_window_start_offset(
        self, state: RealtimeASRState, end_offset_bytes: int
    ) -> int:
        """Pick a window-aligned start so resent windows keep their cache
        identity, never advancing past audio that is still unprocessed."""
        assert self.encoder_window is not None
        audio = state.audio
        window_bytes = self.encoder_window.window_bytes
        complete_end = end_offset_bytes // window_bytes * window_bytes
        context_windows = self.encoder_window.policy.max_audio_context_windows
        start = min(
            complete_end - context_windows * window_bytes,
            audio.last_processed_offset_bytes // window_bytes * window_bytes,
        )
        # Compaction keeps the previous start's extraction context resident, so
        # the earliest usable start is the previous one (base plus context).
        if audio.base_offset_bytes == 0:
            floor = 0
        else:
            resident = audio.base_offset_bytes + self.encoder_window.context_bytes
            floor = -(-resident // window_bytes) * window_bytes
        # A small amount of text after a stall cannot release the entire
        # retained interval at once. Advance by at most one native window.
        start = max(floor, min(start, floor + window_bytes))
        if state.unconfirmed_window_start is not None:
            start = min(start, state.unconfirmed_window_start)
        # Allow one extra window to resolve a stall, including the initial
        # full-audio handoff when its threshold exceeds rolling context.
        max_retained = (
            max(
                (context_windows + 1) * window_bytes,
                self.activation_threshold_bytes + self.chunk_size_bytes,
            )
            + window_bytes
        )
        if end_offset_bytes - start > max_retained:
            raise RuntimeError(
                "realtime ASR transcript did not advance within the retained audio limit"
            )
        return start

    # ---- execution -----------------------------------------------------------

    async def _execute_cumulative_step(
        self,
        state: RealtimeASRState,
        step: _TranscriptionStep,
        sampling_params: Dict[str, Any],
        preview: Optional[_StreamPreview],
        on_transcript_delta: Optional[_TranscriptDeltaCallback],
    ) -> _TranscriptionOutcome:
        """Re-transcribe accumulated audio and reconcile its cumulative text."""
        samples = await self._snapshot_samples(
            state.audio, step.start_offset_bytes, step.end_offset_bytes
        )
        # Each suffix starts after this request's prefix. Reconcile complete
        # hypotheses so identical suffixes can represent new repeated speech.
        decoder_prefix = state.transcript.get_prefix_text()

        async def publish_snapshot(text: str) -> None:
            # Match StreamingASRState's word holdback. The final word may
            # still be incomplete in a decoder snapshot.
            snapshot = " ".join((decoder_prefix + text).split()[:-1])
            # An empty or partial first word must not trim the supplied prefix.
            if not snapshot or not snapshot.startswith(decoder_prefix):
                return
            # StreamingASRState holds only scalars and strings, so a shallow
            # copy isolates the preview reconciliation from the real state.
            candidate = apply_cumulative_transcript(
                copy(state.transcript), snapshot, is_last=False
            )
            await self._extend_streamed(preview, candidate, on_transcript_delta)

        generation = await generate_asr_transcript(
            tokenizer_manager=self.tokenizer_manager,
            adapter=self.adapter,
            decoder_prefix=decoder_prefix,
            audio_data=samples,
            sampling_params=sampling_params,
            routed_dp_rank=self.routed_dp_rank,
            on_update=publish_snapshot if preview is not None else None,
        )
        if generation is None:
            if step.is_last:
                # Completing without the final decode would silently drop the
                # last audio; fail the item instead of publishing it.
                raise RuntimeError("final realtime ASR request returned no response")
            # An empty backend iterator must not commit uncovered audio. Only
            # the attempted cursor advances, so the next step or the forced
            # final decode covers the audio again from offset zero.
            logger.warning("[realtime] cumulative ASR step returned no response")
            return _TranscriptionOutcome(audio_processed=False)
        if generation.finish_reason == "length":
            # A truncated candidate must not enter published text or the next
            # decoder prefix, including on intermediate cumulative steps.
            raise RuntimeError("realtime ASR decode reached max_new_tokens")
        recovering_unprocessed_audio = (
            state.audio.last_attempted_offset_bytes
            > state.audio.last_processed_offset_bytes
        )
        if (
            recovering_unprocessed_audio
            and state.has_transcript
            and not generation.text
        ):
            if step.is_last:
                raise RuntimeError("final realtime ASR recovery returned empty text")
            return _TranscriptionOutcome(audio_processed=False)
        transcript = copy(state.transcript)
        delta = apply_cumulative_transcript(
            transcript, decoder_prefix + generation.text, is_last=step.is_last
        )
        return _TranscriptionOutcome(
            audio_processed=True,
            delta=delta,
            transcript=transcript,
        )

    async def _execute_encoder_window_step(
        self,
        state: RealtimeASRState,
        step: _TranscriptionStep,
        sampling_params: Dict[str, Any],
        preview: Optional[_StreamPreview],
        on_transcript_delta: Optional[_TranscriptDeltaCallback],
    ) -> _TranscriptionOutcome:
        """Decode the continuation of encoder-aligned rolling context."""
        policy = self.encoder_window
        assert policy is not None
        if step.start_offset_bytes % policy.window_bytes:
            raise RuntimeError("encoder-window request is not window aligned")
        samples = await self._snapshot_samples(
            state.audio,
            step.start_offset_bytes - step.leading_context_bytes,
            step.end_offset_bytes,
        )

        async def publish_snapshot(text: str) -> None:
            units = split_units(text)[:-1]
            if not units:
                return
            snapshot = join_units(units)
            candidate = self._reconcile_encoder_window_text(state, step, snapshot).delta
            await self._extend_streamed(preview, candidate, on_transcript_delta)

        try:
            generation = await generate_asr_transcript(
                tokenizer_manager=self.tokenizer_manager,
                adapter=self.adapter,
                audio_data=samples,
                sampling_params=sampling_params,
                decoder_prefix=step.decoder_prefix,
                mm_processor_kwargs=build_audio_window_processor_kwargs(
                    leading_context_samples=step.leading_context_bytes
                    // PCM_SAMPLE_WIDTH_BYTES,
                ),
                routed_dp_rank=self.routed_dp_rank,
                on_update=publish_snapshot if preview is not None else None,
            )
        except ASRBackendAborted as error:
            # Text already on the wire can be neither retracted nor safely
            # re-decoded, so the item fails as it does on the final step.
            if (
                not error.retryable
                or step.is_last
                or (preview is not None and preview.streamed)
            ):
                raise
            logger.warning(
                "[realtime] encoder-window ASR step aborted (%s); retaining audio",
                error,
            )
            return _TranscriptionOutcome(audio_processed=False, window_failed=True)
        if generation is None:
            if step.is_last:
                raise RuntimeError("final realtime ASR request returned no response")
            logger.warning("[realtime] encoder-window ASR step returned no response")
            return _TranscriptionOutcome(audio_processed=False, window_failed=True)
        if generation.finish_reason == "length":
            if step.is_last or (preview is not None and preview.streamed):
                raise RuntimeError("realtime ASR decode reached max_new_tokens")
            logger.warning(
                "[realtime] encoder-window ASR step reached max_new_tokens; "
                "retaining audio for retry"
            )
            return _TranscriptionOutcome(audio_processed=False, window_failed=True)
        return self._reconcile_encoder_window_text(state, step, generation.text)

    def _reconcile_encoder_window_text(
        self, state: RealtimeASRState, step: _TranscriptionStep, text: str
    ) -> _TranscriptionOutcome:
        """Reconcile one windowed decode without mutating state."""
        assert self.encoder_window is not None
        suffix_state = state.decoder_suffix or DecoderSuffixState(
            emitted_text=state.transcript.emitted_text, pending=step.handoff_pending
        )
        update = suffix_state.reconcile(
            text,
            is_last=step.is_last,
            holdback_units=self.encoder_window.policy.decoder_prefix_holdback_units,
        )
        if (
            step.is_last
            and state.audio.last_attempted_offset_bytes
            > state.audio.last_processed_offset_bytes
            and suffix_state.pending
            and not text
        ):
            raise RuntimeError("final realtime ASR recovery returned empty text")
        if (
            update.empty_continuation
            and not step.is_last
            and step.end_offset_bytes > state.audio.last_processed_offset_bytes
            and not state.deferred_empty_continuation
        ):
            # Give the model one more chunk of audio before accepting silence.
            return _TranscriptionOutcome(audio_processed=False, deferred_empty=True)
        return _TranscriptionOutcome(
            audio_processed=(
                step.is_last
                or bool(update.delta)
                or (update.empty_continuation and not suffix_state.pending)
            ),
            delta=update.delta,
            suffix_update=update,
        )

    async def _extend_streamed(
        self,
        preview: _StreamPreview,
        candidate: str,
        on_transcript_delta: Optional[_TranscriptDeltaCallback],
    ) -> None:
        """Publish only compatible extensions; the final candidate decides acceptance."""
        assert on_transcript_delta is not None
        extra = preview.remaining(candidate)
        if not extra:
            return
        await on_transcript_delta(extra)
        preview.streamed = candidate

    async def _snapshot_samples(
        self, audio: AudioBuffer, start_offset_bytes: int, end_offset_bytes: int
    ) -> np.ndarray:
        pcm = audio.snapshot(start_offset_bytes, end_offset_bytes)
        return await asyncio.to_thread(pcm_to_float_samples, pcm)

    # ---- commit ----------------------------------------------------------------

    def _commit_outcome(
        self,
        state: RealtimeASRState,
        step: _TranscriptionStep,
        outcome: _TranscriptionOutcome,
    ) -> None:
        """Adopt validated text and advance audio only as its outcome permits."""
        audio = state.audio
        audio.last_attempted_offset_bytes = step.end_offset_bytes
        if outcome.window_failed:
            state.consecutive_window_failures += 1
            if state.consecutive_window_failures >= _MAX_CONSECUTIVE_WINDOW_FAILURES:
                if state.encoder_window_active:
                    # Earlier PCM was compacted after activation, so there is
                    # no cumulative request that could still cover it.
                    raise RuntimeError(
                        "encoder-window ASR failed repeatedly after activation"
                    )
                logger.warning(
                    "[realtime] encoder-window handoff failed %d times; this item "
                    "falls back to cumulative transcription",
                    state.consecutive_window_failures,
                )
                state.encoder_window_disabled = True
            return
        state.deferred_empty_continuation = outcome.deferred_empty
        if outcome.transcript is not None:
            state.transcript = outcome.transcript
        if outcome.suffix_update is not None:
            state.consecutive_window_failures = 0
            if state.decoder_suffix is None:
                state.decoder_suffix = DecoderSuffixState(
                    emitted_text=state.transcript.emitted_text,
                    pending=step.handoff_pending,
                )
            state.decoder_suffix.apply(outcome.suffix_update)
        if step.uses_encoder_windows:
            state.unconfirmed_window_start = (
                None if outcome.audio_processed else step.start_offset_bytes
            )
        if not outcome.audio_processed:
            return
        state.consecutive_window_failures = 0
        audio.last_processed_offset_bytes = step.end_offset_bytes
        if step.uses_encoder_windows and not step.is_last:
            # Compact only after accepted transcript progress (or confirmed
            # silence), keeping the next request's extraction context.
            audio.discard_before(
                max(0, step.start_offset_bytes - self.encoder_window.context_bytes)
            )


def _cumulative_unpublished_text(transcript: StreamingASRState) -> str:
    """Cumulative text after the published boundary, ready for handoff."""
    full = join_text("", transcript.full_transcript)
    confirmed = transcript.confirmed_text
    if full.startswith(transcript.emitted_text):
        confirmed = transcript.emitted_text
        remainder = full[len(transcript.emitted_text) :]
        if not remainder or not needs_space(transcript.emitted_text, remainder):
            return remainder.lstrip(" ")
    confirmed_units = split_units(confirmed)
    full_units = split_units(transcript.full_transcript)
    common = common_unit_prefix(confirmed_units, full_units)
    return join_units(full_units[common:])
