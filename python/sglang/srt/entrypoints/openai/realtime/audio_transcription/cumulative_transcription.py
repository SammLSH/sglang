from __future__ import annotations

import logging
from copy import copy
from typing import Dict, Optional

from pydantic import JsonValue

from sglang.srt.entrypoints.openai.realtime.audio_transcription.audio_buffer import (
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
)
from sglang.srt.entrypoints.openai.streaming_asr import (
    IncompleteTranscription,
    StreamingASRState,
    apply_cumulative_transcript,
    generate_transcript,
    iter_text_unit_spans,
)

logger = logging.getLogger(__name__)


class CumulativeTranscriptionMode(TranscriptionMode):
    """Transcribe all received audio using the sent transcript as decoder context."""

    def create_state(self) -> StreamingASRState:
        return StreamingASRState(**self.chunked_streaming_config)

    def build_transcription_request(
        self,
        state: RealtimeTranscriptionState,
        *,
        is_last: bool,
    ) -> TranscriptionRequest:
        transcript = state.mode_state
        assert isinstance(transcript, StreamingASRState)
        return TranscriptionRequest(
            is_last=is_last,
            start_offset_bytes=0,
            end_offset_bytes=state.audio.received_bytes,
            last_attempted_offset_bytes=state.audio.last_attempted_offset_bytes,
            last_processed_offset_bytes=state.audio.last_processed_offset_bytes,
            emitted_text=state.emitted_text,
            mode_state=transcript,
            decoder_prefix=transcript.get_prefix_text(emitted_text=state.emitted_text),
        )

    async def transcribe_audio(
        self,
        state: RealtimeTranscriptionState,
        request: TranscriptionRequest,
        *,
        sampling_params: Dict[str, JsonValue],
        on_transcript_candidate: Optional[TranscriptCandidateCallback],
    ) -> TranscriptionOutcome:
        """Compare generated text with the sent transcript to find new text."""
        previous_transcript_state = request.mode_state
        assert isinstance(previous_transcript_state, StreamingASRState)
        decoder_prefix = request.decoder_prefix
        samples = await extract_audio_samples(
            state.audio, request.start_offset_bytes, request.end_offset_bytes
        )

        async def _send_partial_transcript(text: str) -> None:
            assert on_transcript_candidate is not None
            # The decoder may still extend the last word. Keep original spacing
            # so partial results match the final transcript exactly.
            full_transcript = decoder_prefix + text
            text_unit_spans = list(iter_text_unit_spans(full_transcript))
            complete_text = (
                full_transcript[: text_unit_spans[-2][1]]
                if len(text_unit_spans) > 1
                else ""
            )
            if not complete_text or not complete_text.startswith(decoder_prefix):
                return
            else:
                transcript_candidate = apply_cumulative_transcript(
                    copy(previous_transcript_state),
                    complete_text,
                    is_last=False,
                    emitted_text=request.emitted_text,
                )
                await on_transcript_candidate(transcript_candidate)

        try:
            text = await generate_transcript(
                tokenizer_manager=self.tokenizer_manager,
                adapter=self.adapter,
                decoder_prefix=decoder_prefix,
                audio_data=samples,
                sampling_params=sampling_params,
                on_transcript_update=_send_partial_transcript
                if on_transcript_candidate is not None
                else None,
            )
        except IncompleteTranscription as error:
            if (
                error.reason == "length"
                or request.is_last
                or state.emitted_text != request.emitted_text
            ):
                raise
            else:
                logger.warning("Cumulative ASR request incomplete: %s", error)
            # Keep the audio unprocessed so the next request retries it.
            return TranscriptionOutcome(
                next_mode_state=previous_transcript_state, audio_processed=False
            )
        recovering_unprocessed_audio = (
            request.last_attempted_offset_bytes > request.last_processed_offset_bytes
        )
        if (
            recovering_unprocessed_audio
            and previous_transcript_state.full_transcript
            and not text
        ):
            if request.is_last:
                raise RuntimeError("final realtime ASR recovery returned empty text")
            else:
                return TranscriptionOutcome(
                    next_mode_state=previous_transcript_state, audio_processed=False
                )
        else:
            transcript = copy(previous_transcript_state)
        delta = apply_cumulative_transcript(
            transcript,
            decoder_prefix + text,
            is_last=request.is_last,
            emitted_text=request.emitted_text,
        )
        return TranscriptionOutcome(
            next_mode_state=transcript,
            audio_processed=True,
            delta=delta,
        )

    def flush_pending_transcript(
        self, state: RealtimeTranscriptionState
    ) -> TranscriptionOutcome:
        transcript = state.mode_state
        assert isinstance(transcript, StreamingASRState)
        transcript = copy(transcript)
        delta = transcript.finalize(emitted_text=state.emitted_text)
        return TranscriptionOutcome(
            next_mode_state=transcript,
            audio_processed=False,
            delta=delta,
        )
