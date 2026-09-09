"""Application service for realtime ASR inference.

The processor is connection-scoped but keeps no stream progress itself. Each
call builds and executes one backend transcription step, then commits its
audio outcome to the explicit ``RealtimeASRState`` supplied by the endpoint.

Per-chunk flow, driven by the endpoint::

    append audio -> is_chunk_ready()? -> process()
        _build_transcription_step  resolve mode, audio range, and prompt
        _execute_step              run the backend request, reconcile text
                                   (cumulative reconciliation mutates its
                                   state; windowed reconciliation returns an
                                   update applied only on commit)
        _commit_outcome            advance cursors, compact old PCM
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
    generate_cumulative_transcript,
    join_text,
    join_units,
    split_units,
)
from sglang.srt.entrypoints.openai.transcription_adapters.base import (
    TranscriptionAdapter,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.multimodal.encoder_window import encoder_window_kwargs
from sglang.srt.runtime_context import get_serving
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

# Consecutive windowed requests that may fail to produce a usable decode
# (abort, no response, truncation) before the item stops trying windows.
_MAX_CONSECUTIVE_WINDOW_FAILURES = 2

_TranscriptDeltaCallback = Callable[[str], Awaitable[None]]


class _StreamPreview(msgspec.Struct):
    """Text published to the client while one backend decode is running.

    ``streamed`` is the candidate delta already on the wire for this step;
    every later publication must extend it unit by unit.
    """

    streamed: str = ""


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

    @property
    def has_audio(self) -> bool:
        return self.audio.received_bytes > 0

    @property
    def has_transcript(self) -> bool:
        if self.decoder_suffix is not None:
            return bool(self.decoder_suffix.latest_text)
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
    ``suffix_update`` stays unapplied until commit so a failed step leaves
    the windowed state untouched.
    """

    audio_processed: bool
    delta: str = ""
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
        server_args: ServerArgs,
        *,
        encoder_window: Optional[ResolvedEncoderWindowPolicy] = None,
        session_id: str = "",
    ) -> None:
        self.tokenizer_manager = tokenizer_manager
        self.adapter = adapter
        self.model_sample_rate = adapter.model_sample_rate
        self.pcm_bytes_per_second = self.model_sample_rate * PCM_SAMPLE_WIDTH_BYTES

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
        self.max_buffer_seconds = get_serving().asr_max_buffer_seconds
        self.max_buffer_bytes = self.max_buffer_seconds * self.pcm_bytes_per_second
        self.decoder_streaming = bool(get_serving().enable_asr_decoder_streaming)

        self.encoder_window = encoder_window
        self.activation_threshold_bytes: Optional[int] = None
        self.routed_dp_rank: Optional[int] = None
        if encoder_window is not None:
            if encoder_window.window_bytes <= 0:
                raise ValueError("invalid audio encoder-window geometry")
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

    def max_item_bytes(
        self, language: Optional[str], *, encoder_window_disabled: bool = False
    ) -> int:
        """Total PCM bytes one item may receive.

        Items that cannot use windows (no policy, unsupported language, or a
        fallback) keep the cumulative path and therefore the smaller of the
        buffer cap and the activation threshold.
        """
        if self.encoder_window is None:
            return self.max_buffer_bytes
        if encoder_window_disabled or not self.encoder_window.supports_language(
            language
        ):
            return min(self.max_buffer_bytes, self.activation_threshold_bytes)
        return self.max_buffer_bytes

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
        emitted_before = state.emitted_text
        outcome = await self._execute_step(
            state, step, sampling_params, preview, on_transcript_delta
        )
        self._commit_outcome(state, step, outcome)
        if preview is None or not preview.streamed:
            return outcome.delta
        return self._remaining_after_streamed(
            state, emitted_before, preview.streamed, outcome.delta
        )

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
        return _TranscriptionStep(
            is_last=is_last,
            uses_encoder_windows=True,
            start_offset_bytes=start_offset_bytes,
            end_offset_bytes=end_offset_bytes,
            leading_context_bytes=min(
                self.encoder_window.context_bytes,
                start_offset_bytes - state.audio.base_offset_bytes,
            ),
            decoder_prefix=suffix_state.bounded_prefix(
                self.tokenizer_manager.tokenizer, max_prefix_tokens
            ),
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
        return max(floor, start)

    # ---- execution -----------------------------------------------------------

    async def _execute_step(
        self,
        state: RealtimeASRState,
        step: _TranscriptionStep,
        sampling_params: Dict[str, Any],
        preview: Optional[_StreamPreview],
        on_transcript_delta: Optional[_TranscriptDeltaCallback],
    ) -> _TranscriptionOutcome:
        if step.uses_encoder_windows:
            return await self._execute_encoder_window_step(
                state, step, sampling_params, preview, on_transcript_delta
            )
        return await self._execute_cumulative_step(
            state, step, sampling_params, preview, on_transcript_delta
        )

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

        async def publish_snapshot(text: str) -> None:
            snapshot = _complete_units(text)
            if not snapshot:
                return
            # StreamingASRState holds only scalars and strings, so a shallow
            # copy isolates the preview reconciliation from the real state.
            candidate = apply_cumulative_transcript(
                copy(state.transcript), snapshot, is_last=False
            )
            await self._extend_streamed(preview, candidate, on_transcript_delta)

        generation = await generate_cumulative_transcript(
            tokenizer_manager=self.tokenizer_manager,
            adapter=self.adapter,
            state=state.transcript,
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
        if step.is_last and generation.finish_reason == "length":
            # Never publish a known-truncated transcript as completed.
            raise RuntimeError("realtime ASR decode reached max_new_tokens")
        delta = apply_cumulative_transcript(
            state.transcript, generation.text, is_last=step.is_last
        )
        # Keep truncated audio unprocessed so the cursor state forces a clean
        # final decode at commit. The cumulative transcript update above still
        # preserves the existing intermediate streaming behavior.
        return _TranscriptionOutcome(
            audio_processed=generation.finish_reason != "length",
            delta=delta,
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
        prefix_units = split_units(step.decoder_prefix)

        async def publish_snapshot(text: str) -> None:
            units = split_units(text)[:-1]
            if not units:
                return
            # While the leading units still match the decoder prefix the
            # model may be replaying it; wait for the replay to complete or
            # diverge before treating anything as new transcript text.
            if len(units) < len(prefix_units) and common_unit_prefix(
                prefix_units, units, normalized=True
            ) == len(units):
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
                prompt=self.adapter.prompt_template,
                decoder_prefix=step.decoder_prefix,
                mm_processor_kwargs=encoder_window_kwargs(
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
            decoder_prefix=step.decoder_prefix,
        )
        if (
            update.empty_continuation
            and not step.is_last
            and step.end_offset_bytes > state.audio.last_processed_offset_bytes
            and not state.deferred_empty_continuation
        ):
            # Give the model one more chunk of audio before accepting silence.
            return _TranscriptionOutcome(audio_processed=False, deferred_empty=True)
        return _TranscriptionOutcome(
            audio_processed=True, delta=update.delta, suffix_update=update
        )

    async def _extend_streamed(
        self,
        preview: _StreamPreview,
        candidate: str,
        on_transcript_delta: Optional[_TranscriptDeltaCallback],
    ) -> None:
        """Publish the units by which a candidate delta extends the streamed one.

        A candidate that does not extend what is already on the wire is
        skipped: the completed decode reconciles it, never a retraction.
        """
        assert on_transcript_delta is not None
        streamed_units = split_units(preview.streamed)
        candidate_units = split_units(candidate)
        if candidate_units[: len(streamed_units)] != streamed_units:
            return
        extra = join_units(candidate_units[len(streamed_units) :])
        if not extra:
            return
        await on_transcript_delta(extra)
        preview.streamed = candidate

    def _remaining_after_streamed(
        self,
        state: RealtimeASRState,
        emitted_before: str,
        streamed: str,
        final_delta: str,
    ) -> str:
        """Reconcile the completed decode with text streamed during it.

        Streamed text is never retracted. If the final delta rewrote it, the
        wire keeps both the streamed units and the final units past the
        common prefix, and the transcript state adopts that wire text so
        the next decoder prefix is what the client has seen.
        """
        streamed_units = split_units(streamed)
        final_units = split_units(final_delta)
        common = common_unit_prefix(streamed_units, final_units)
        remaining = join_units(final_units[common:])
        if common == len(streamed_units):
            return remaining
        logger.warning(
            "[realtime] completed decode revised streamed text: streamed=%r final=%r",
            streamed[-80:],
            final_delta[:80],
        )
        wire_text = join_text(emitted_before, join_text(streamed, remaining))
        if state.decoder_suffix is not None:
            # A revision can move already-streamed units back into pending.
            # Consume only the matching prefix of this step's continuation,
            # not later occurrences that may be genuine repetitions.
            pending_units = split_units(state.decoder_suffix.pending)
            matched = common_unit_prefix(streamed_units, final_units + pending_units)
            consumed_pending = max(0, matched - len(final_units))
            if consumed_pending:
                state.decoder_suffix.pending = join_units(
                    pending_units[consumed_pending:]
                )
            state.decoder_suffix.emitted_text = wire_text
        else:
            state.transcript.emitted_text = wire_text
        return remaining

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
        """Advance attempted audio unconditionally and processed audio only
        after its decode enters transcript state."""
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
        if not outcome.audio_processed:
            return
        state.consecutive_window_failures = 0
        if step.uses_encoder_windows:
            assert outcome.suffix_update is not None
            if state.decoder_suffix is None:
                state.decoder_suffix = DecoderSuffixState(
                    emitted_text=state.transcript.emitted_text,
                    pending=step.handoff_pending,
                )
            state.decoder_suffix.apply(outcome.suffix_update)
        audio.last_processed_offset_bytes = step.end_offset_bytes
        if step.uses_encoder_windows and not step.is_last:
            # Everything before this step's start is represented by the
            # decoder prefix from now on; keep the extraction context the next
            # request needs for its first window.
            audio.discard_before(
                max(0, step.start_offset_bytes - self.encoder_window.context_bytes)
            )


def _complete_units(text: str) -> str:
    """A decoder snapshot without its last unit, which may be cut mid-token."""
    return join_units(split_units(text)[:-1])


def _cumulative_unpublished_text(transcript: StreamingASRState) -> str:
    """Cumulative text decoded but not yet confirmed, unit-aligned."""
    confirmed_units = split_units(transcript.confirmed_text)
    full_units = split_units(transcript.full_transcript)
    common = common_unit_prefix(confirmed_units, full_units)
    return join_units(full_units[common:])
