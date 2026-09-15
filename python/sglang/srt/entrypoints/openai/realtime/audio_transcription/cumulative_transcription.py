from __future__ import annotations

import logging
from copy import copy
from typing import Dict, Optional

from msgspec.structs import replace
from pydantic import JsonValue

from sglang.srt.entrypoints.openai.realtime.audio_transcription.audio_buffer import (
    snapshot_samples,
)
from sglang.srt.entrypoints.openai.realtime.audio_transcription.transcription_mode import (
    TranscriptCandidateCallback,
    TranscriptionMode,
    TranscriptionOutcome,
    TranscriptionStep,
)
from sglang.srt.entrypoints.openai.realtime.audio_transcription.transcription_state import (
    CumulativeState,
    RealtimeTranscriptionState,
)
from sglang.srt.entrypoints.openai.streaming_asr import (
    apply_cumulative_transcript,
    generate_transcript,
    iter_unit_spans,
)

logger = logging.getLogger(__name__)


class CumulativeMode(TranscriptionMode):
    """Re-transcribe audio from the item start using its cumulative prefix."""

    def build_step(
        self,
        state: RealtimeTranscriptionState,
        *,
        end_offset_bytes: int,
        is_last: bool,
    ) -> TranscriptionStep:
        current = state.mode_state
        assert isinstance(current, CumulativeState)
        return TranscriptionStep(
            is_last=is_last,
            start_offset_bytes=0,
            end_offset_bytes=end_offset_bytes,
            last_attempted_offset_bytes=state.audio.last_attempted_offset_bytes,
            last_processed_offset_bytes=state.audio.last_processed_offset_bytes,
            emitted_text=state.emitted_text,
            mode_state=current,
            decoder_prefix=current.transcript.get_prefix_text(
                emitted_text=state.emitted_text
            ),
        )

    async def execute_step(
        self,
        state: RealtimeTranscriptionState,
        step: TranscriptionStep,
        *,
        sampling_params: Dict[str, JsonValue],
        on_candidate: Optional[TranscriptCandidateCallback],
    ) -> TranscriptionOutcome:
        """Reconcile complete hypotheses against the request's published prefix."""
        before = step.mode_state
        assert isinstance(before, CumulativeState)
        transcript_before = before.transcript
        decoder_prefix = step.decoder_prefix
        samples = await snapshot_samples(
            state.audio, step.start_offset_bytes, step.end_offset_bytes
        )

        async def _publish_snapshot(text: str) -> None:
            assert on_candidate is not None
            # The final unit may still be incomplete. Slice the original text
            # so mixed-language boundaries match the final candidate exactly.
            full = decoder_prefix + text
            spans = list(iter_unit_spans(full))
            snapshot = full[: spans[-2][1]] if len(spans) > 1 else ""
            if not snapshot or not snapshot.startswith(decoder_prefix):
                return
            else:
                candidate = apply_cumulative_transcript(
                    copy(transcript_before),
                    snapshot,
                    is_last=False,
                    emitted_text=step.emitted_text,
                )
                await on_candidate(candidate)

        generation = await generate_transcript(
            tokenizer_manager=self.tokenizer_manager,
            adapter=self.adapter,
            decoder_prefix=decoder_prefix,
            audio_data=samples,
            sampling_params=sampling_params,
            routed_dp_rank=self.routed_dp_rank,
            on_update=_publish_snapshot if on_candidate is not None else None,
        )
        if generation is None:
            if step.is_last:
                raise RuntimeError("final realtime ASR request returned no response")
            else:
                logger.warning("[realtime] cumulative ASR step returned no response")
            # Only the attempted cursor advances; the next decode covers the
            # retained audio again from offset zero.
            return TranscriptionOutcome(next_mode_state=before, audio_covered=False)
        elif generation.finish_reason == "length":
            raise RuntimeError("realtime ASR decode reached max_new_tokens")
        else:
            recovering_unprocessed_audio = (
                step.last_attempted_offset_bytes > step.last_processed_offset_bytes
            )
        if (
            recovering_unprocessed_audio
            and transcript_before.full_transcript
            and not generation.text
        ):
            if step.is_last:
                raise RuntimeError("final realtime ASR recovery returned empty text")
            else:
                return TranscriptionOutcome(next_mode_state=before, audio_covered=False)
        else:
            transcript = copy(transcript_before)
        delta = apply_cumulative_transcript(
            transcript,
            decoder_prefix + generation.text,
            is_last=step.is_last,
            emitted_text=step.emitted_text,
        )
        return TranscriptionOutcome(
            next_mode_state=replace(before, transcript=transcript),
            audio_covered=True,
            delta=delta,
        )

    def flush_pending(self, state: RealtimeTranscriptionState) -> TranscriptionOutcome:
        current = state.mode_state
        assert isinstance(current, CumulativeState)
        transcript = copy(current.transcript)
        delta = transcript.finalize(emitted_text=state.emitted_text)
        return TranscriptionOutcome(
            next_mode_state=replace(current, transcript=transcript),
            audio_covered=False,
            delta=delta,
        )
