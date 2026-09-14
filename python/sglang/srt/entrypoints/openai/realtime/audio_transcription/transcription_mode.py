"""Transcription mode interface and per-step input and outcome types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Dict, Optional

import msgspec

from sglang.srt.entrypoints.openai.realtime.audio_transcription.transcription_state import (
    ModeState,
    RealtimeTranscriptionState,
)
from sglang.srt.entrypoints.openai.transcription_adapters.base import (
    TranscriptionAdapter,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager

# Receives an exact append to TranscriptionStep.emitted_text, including boundary
# spaces. Later candidates may include text already sent by earlier previews.
TranscriptCandidateCallback = Callable[[str], Awaitable[None]]


class TranscriptionStep(msgspec.Struct, frozen=True):
    """Request and candidate baseline fixed before generation awaits.

    mode_state contains a private copy of the transcript candidate. Published
    text and audio cursors are snapshots, so intermediate sends cannot change
    the reconciliation baseline of this request.
    """

    is_last: bool
    start_offset_bytes: int
    end_offset_bytes: int
    last_attempted_offset_bytes: int
    last_processed_offset_bytes: int
    emitted_text: str
    mode_state: ModeState
    # Extraction context before the first window, outside the window itself.
    leading_context_bytes: int = 0
    decoder_prefix: str = ""
    # Unpublished cumulative text seeds the first continuation candidate.
    handoff_pending: str = ""


class TranscriptionOutcome(msgspec.Struct, frozen=True):
    """Changes accepted after publication, including recoverable attempts.

    Modes decide their next working state and safe PCM release boundary.
    Audio coverage is independent of candidate acceptance and mode handoff.
    """

    next_mode_state: ModeState
    audio_covered: bool
    # Exact append to the step's published baseline (or the flush baseline).
    # The publisher sends only the portion not already emitted by previews.
    delta: str = ""
    discard_before_bytes: int | None = None


class TranscriptionMode(ABC):
    """Compute request outcomes without owning or mutating per-item progress."""

    def __init__(
        self,
        tokenizer_manager: TokenizerManager,
        adapter: TranscriptionAdapter,
        *,
        routed_dp_rank: Optional[int] = None,
    ) -> None:
        self.tokenizer_manager = tokenizer_manager
        self.adapter = adapter
        self.routed_dp_rank = routed_dp_rank

    @abstractmethod
    def build_step(
        self,
        state: RealtimeTranscriptionState,
        *,
        end_offset_bytes: int,
        is_last: bool,
    ) -> TranscriptionStep:
        """Snapshot audio bounds, published text, and private candidate state."""
        ...

    @abstractmethod
    async def execute_step(
        self,
        state: RealtimeTranscriptionState,
        step: TranscriptionStep,
        *,
        sampling_params: Dict[str, Any],
        on_candidate: Optional[TranscriptCandidateCallback],
    ) -> TranscriptionOutcome:
        """Compute candidates and return proposed changes for the caller to commit."""
        ...

    @abstractmethod
    def flush_pending(self, state: RealtimeTranscriptionState) -> TranscriptionOutcome:
        """Prepare held text for publication without another backend request."""
        ...
