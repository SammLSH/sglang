import io
import logging
import re
import unicodedata
from contextlib import aclosing
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Union

import msgspec
import numpy as np
import soundfile as sf
from fastapi import HTTPException, Request

from sglang.srt.entrypoints.openai.transcription_adapters.base import (
    TranscriptionAdapter,
)
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.runtime_context import get_serving

logger = logging.getLogger(__name__)


# Collapse whitespace before punctuation so batched-inference token
# boundary jitter (" ," vs ",") doesn't leak into deltas. Covers both
# ASCII punctuation and the CJK / fullwidth equivalents.
_PUNCT_WS_RE = re.compile(r"\s+([,.;:!?，。！？；：、])")


@dataclass
class StreamingASRState:
    """State for chunk-based streaming ASR with prefix rollback.

    Published text belongs to the caller and is supplied to each operation.
    Updating a candidate never records publication.

    Parameters are model-specific and should be provided via the
    adapter's ``chunked_streaming_config``.

    Known limitation: rollback uses str.split() which is ineffective
    for CJK languages (no whitespace between words).
    TODO: implement token-level rollback to handle all languages
    correctly.
    """

    chunk_size_sec: float
    unfixed_chunk_num: int
    unfixed_token_num: int
    confirmed_text: str = ""
    full_transcript: str = ""
    chunk_index: int = 0

    def get_prefix_text(self, *, emitted_text: str) -> str:
        if self.chunk_index < self.unfixed_chunk_num or not emitted_text:
            return ""
        return emitted_text

    def update(self, new_transcript: str, *, emitted_text: str) -> str:
        """Update the candidate hypothesis; the caller publishes the delta."""
        # A shorter continuation can move holdback into the supplied prefix.
        # Keep that published boundary when it belongs to this hypothesis.
        old_confirmed = (
            emitted_text
            if new_transcript.startswith(emitted_text)
            else self.confirmed_text
        )
        words = new_transcript.split()
        if len(words) > self.unfixed_token_num:
            self.confirmed_text = " ".join(words[: -self.unfixed_token_num])
        else:
            self.confirmed_text = ""
        self.full_transcript = new_transcript
        self.chunk_index += 1
        if self.confirmed_text.startswith(old_confirmed):
            return join_text("", self.confirmed_text[len(old_confirmed) :].strip())
        # Model revised earlier text, use word level common prefix to avoid
        # re-emitting already-sent content and cutting mid-word.
        old_words = old_confirmed.split()
        new_words = self.confirmed_text.split()
        common_count = 0
        for ow, nw in zip(old_words, new_words):
            if ow != nw:
                break
            common_count += 1
        return join_text("", " ".join(new_words[common_count:]))

    def unpublished_text(self, *, emitted_text: str, split_cjk: bool = False) -> str:
        """Read the candidate tail without publishing or changing state.

        Cumulative finalization retains its whitespace-word rollback. Window
        handoff splits CJK runs into text units, as suffix agreement does.
        Both use the same exact published boundary and word-extension guard.
        """
        text = join_text("", self.full_transcript)
        confirmed = self.confirmed_text
        if text.startswith(emitted_text):
            confirmed = emitted_text
            remainder = text[len(confirmed) :]
            # Preserve punctuation added after the published word, without
            # allowing an extension that the wire would split mid-word.
            if not remainder or not needs_space(confirmed, remainder):
                return remainder.lstrip(" ")
        if split_cjk:
            confirmed_units = split_units(confirmed)
            full_units = split_units(self.full_transcript)
        else:
            confirmed_units = confirmed.split()
            full_units = text.split()
        common = common_unit_prefix(confirmed_units, full_units)
        if not split_cjk and common == 0 and confirmed_units and full_units:
            return text
        tail = full_units[common:]
        return join_units(tail) if split_cjk else " ".join(tail)

    def finalize(self, *, emitted_text: str) -> str:
        """Finalize the hypothesis; publishing and recording remain separate."""
        delta = join_text("", self.unpublished_text(emitted_text=emitted_text))
        self.confirmed_text = join_text("", self.full_transcript)
        return delta


def split_audio_chunks(audio_data: bytes, chunk_size_sec: float) -> List[bytes]:
    if not audio_data:
        raise ValueError("audio_data is empty")
    if chunk_size_sec <= 0:
        raise ValueError(f"chunk_size_sec must be positive, got {chunk_size_sec}")
    audio_file = io.BytesIO(audio_data)
    try:
        data, sample_rate = sf.read(audio_file, dtype="float32")
    except sf.LibsndfileError as e:
        raise ValueError(f"failed to decode audio: {e}") from e
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    chunk_size_samples = int(chunk_size_sec * sample_rate)
    total_samples = len(data)
    chunks = []
    for end in range(
        chunk_size_samples, total_samples + chunk_size_samples, chunk_size_samples
    ):
        end = min(end, total_samples)
        buf = io.BytesIO()
        sf.write(buf, data[:end], sample_rate, format="WAV")
        chunks.append(buf.getvalue())
    return chunks


def normalize_whitespace(text: str) -> str:
    """Normalize blank text and punctuation, preserving continuation spaces."""
    if not text.strip():
        return ""
    return _PUNCT_WS_RE.sub(r"\1", text)


_NO_SPACE_BEFORE = frozenset(".,!?;:%)]}，。！？；：、）】》」』")
_NO_SPACE_AFTER = frozenset("([{（【《「『")


def _is_cjk(c: str) -> bool:
    """Whether char is a CJK-context glyph that doesn't take inter-word
    spaces — ideographs, Japanese kana, CJK punctuation, fullwidth forms.
    Excludes Hangul / Devanagari / Arabic etc., which are non-ASCII but
    space-separated and need the normal boundary space."""
    cp = ord(c)
    return (
        0x3000 <= cp <= 0x303F  # CJK Symbols and Punctuation (，。、《》「」…)
        or 0x3040 <= cp <= 0x309F  # Hiragana
        or 0x30A0 <= cp <= 0x30FF  # Katakana
        or 0x3400 <= cp <= 0x4DBF  # CJK Unified Ideographs Ext A
        or 0x4E00 <= cp <= 0x9FFF  # CJK Unified Ideographs
        or 0xFF00 <= cp <= 0xFFEF  # Halfwidth & Fullwidth Forms (fullwidth ASCII)
    )


def needs_space(prev: str, cur: str) -> bool:
    """Return whether a boundary space is needed between emitted deltas.

    Avoid spaces around punctuation and between adjacent CJK-context glyphs.
    Shared by the realtime WS and HTTP SSE chunked streaming paths.
    """
    if not prev or not cur:
        return False
    if prev[-1].isspace() or cur[0].isspace():
        return False
    if cur[0] in _NO_SPACE_BEFORE or prev[-1] in _NO_SPACE_AFTER:
        return False
    if _is_cjk(prev[-1]) and _is_cjk(cur[0]):
        return False
    return True


async def process_asr_chunk(
    tokenizer_manager: TokenizerManager,
    adapter: TranscriptionAdapter,
    state: StreamingASRState,
    audio_data: bytes,
    sampling_params: Dict[str, Any],
    is_last: bool,
    raw_request: Optional[Request] = None,
    routing_key: Optional[str] = None,
    *,
    emitted_text: str,
) -> str:
    """Update a caller-owned candidate for one HTTP streaming chunk.

    The caller accepts this candidate after publishing its returned delta.
    """
    decoder_prefix = state.get_prefix_text(emitted_text=emitted_text)
    result = await generate_transcript(
        tokenizer_manager=tokenizer_manager,
        adapter=adapter,
        decoder_prefix=decoder_prefix,
        audio_data=audio_data,
        sampling_params=sampling_params,
        raw_request=raw_request,
        routing_key=routing_key,
    )
    if result is None:
        return ""
    return apply_cumulative_transcript(
        state, decoder_prefix + result.text, is_last=is_last, emitted_text=emitted_text
    )


def split_units(text: str) -> List[str]:
    """Split text into reconciliation units.

    Whitespace-separated tokens are kept whole unless they contain CJK
    characters, in which case each such character becomes its own unit and
    the Latin runs around them stay whole:
    ``"hello 你好，world"`` -> ``["hello", "你", "好", "，", "world"]``.
    """
    units: List[str] = []
    for token in text.split():
        run: List[str] = []
        for char in token:
            if _is_cjk(char):
                if run:
                    units.append("".join(run))
                    run = []
                units.append(char)
            else:
                run.append(char)
        if run:
            units.append("".join(run))
    return units


def _space_between(prev: str, cur: str) -> bool:
    prev_char, cur_char = prev[-1], cur[0]
    prev_script = _is_cjk(prev_char)
    cur_script = _is_cjk(cur_char)
    if not prev_script and not cur_script:
        # Plain whitespace-delimited text: always one space, like " ".join.
        return True
    return needs_space(prev, cur)


def join_text(left: str, right: str) -> str:
    """Append published text using the WS/SSE ASCII-space formatting rules."""
    parts = [left]
    for word in right.split(" "):
        if word:
            parts.append(f" {word}" if needs_space(parts[-1], word) else word)
    return "".join(parts)


def join_units(units: Sequence[str]) -> str:
    """Inverse of ``split_units`` up to the boundary spaces it cannot recover."""
    text = ""
    for unit in units:
        if unit:
            if text and _space_between(text, unit):
                text += " "
            text += unit
    return text


def normalize_unit(unit: str) -> str:
    """Canonical form for tolerant unit comparison: NFKC, outer punctuation
    stripped, lowercase. A unit that is only punctuation keeps its NFKC form."""
    normalized = unicodedata.normalize("NFKC", unit)
    start, end = 0, len(normalized)
    while start < end and unicodedata.category(normalized[start])[0] == "P":
        start += 1
    while end > start and unicodedata.category(normalized[end - 1])[0] == "P":
        end -= 1
    return (normalized[start:end] or normalized).lower()


def common_unit_prefix(
    left: Sequence[str], right: Sequence[str], *, normalized: bool = False
) -> int:
    """Length of the longest common prefix, comparing units exactly or after
    ``normalize_unit``."""
    count = 0
    for left_unit, right_unit in zip(left, right):
        if normalized:
            if normalize_unit(left_unit) != normalize_unit(right_unit):
                break
        elif left_unit != right_unit:
            break
        count += 1
    return count


class TranscriptionBackendAborted(RuntimeError):
    """The backend aborted the request instead of finishing it."""

    def __init__(self, message: str, *, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code

    @property
    def retryable(self) -> bool:
        return self.status_code in (
            HTTPStatus.INTERNAL_SERVER_ERROR,
            HTTPStatus.SERVICE_UNAVAILABLE,
        )


class GeneratedTranscript(msgspec.Struct, frozen=True):
    """Normalized ASR text plus the backend stop reason."""

    text: str
    finish_reason: Optional[str]


async def generate_transcript(
    tokenizer_manager: TokenizerManager,
    adapter: TranscriptionAdapter,
    audio_data: Union[bytes, np.ndarray],
    sampling_params: Dict[str, Any],
    decoder_prefix: str = "",
    raw_request: Optional[Request] = None,
    routing_key: Optional[str] = None,
    mm_processor_kwargs: Optional[Dict[str, Any]] = None,
    routed_dp_rank: Optional[int] = None,
    on_update: Optional[Callable[[str], Awaitable[None]]] = None,
) -> Optional[GeneratedTranscript]:
    """Run one backend request and return its normalized transcript.

    With ``on_update`` the backend request streams, and every partial decoder
    snapshot the adapter deems visible is passed to the callback as
    normalized cumulative text before the final result is returned.
    ``decoder_prefix`` is transcript text appended to the adapter's prompt
    for the model to continue; its snapshots are not held back
    waiting for a leading marker the continuation never carries.
    Results and snapshots contain only generated text, excluding that prefix.
    """
    stream = on_update is not None
    chunk_request = GenerateReqInput(
        text=adapter.prompt_template + decoder_prefix,
        audio_data=audio_data,
        sampling_params=sampling_params,
        stream=stream,
        modalities=["audio"],
        routing_key=routing_key,
        mm_processor_kwargs=mm_processor_kwargs,
        routed_dp_rank=routed_dp_rank,
    )

    cumulative_text = ""
    incremental = stream and bool(get_serving().incremental_streaming_output)
    ret = None
    async with aclosing(
        tokenizer_manager.generate_request(chunk_request, raw_request)
    ) as responses:
        while True:
            try:
                ret = await anext(responses)
            except StopAsyncIteration:
                break
            except HTTPException as error:
                # Non-stream scheduler errors are raised by the manager;
                # stream errors arrive as terminal frames below. Normalize
                # only backend reads, never errors from the update callback.
                if error.status_code not in (
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    HTTPStatus.SERVICE_UNAVAILABLE,
                ):
                    raise
                raise TranscriptionBackendAborted(
                    str(error.detail), status_code=error.status_code
                ) from error
            _raise_for_aborted_response(ret)
            if not stream:
                break
            chunk_text = ret.get("text") or ""
            cumulative_text = (
                cumulative_text + chunk_text if incremental else chunk_text
            )
            if _finish_reason_type(ret) == "length":
                # Let the caller handle truncation before this terminal frame
                # publishes anything that would make a retry unsafe.
                continue
            visible_text = adapter.postprocess_streaming_text(
                cumulative_text, continuation=bool(decoder_prefix)
            )
            if visible_text is not None:
                await on_update(normalize_whitespace(visible_text))

    if ret is None:
        logger.warning("[streaming_asr] ASR request returned no response")
        return None

    raw_text = cumulative_text if stream else (ret.get("text") or "")
    return GeneratedTranscript(
        text=normalize_whitespace(adapter.postprocess_text(raw_text)),
        finish_reason=_finish_reason_type(ret),
    )


def _finish_reason_type(response: Dict[str, Any]) -> Optional[str]:
    finish_reason = response.get("meta_info", {}).get("finish_reason")
    if isinstance(finish_reason, dict):
        finish_reason = finish_reason.get("type")
    return finish_reason


def _raise_for_aborted_response(response: Dict[str, Any]) -> None:
    if _finish_reason_type(response) != "abort":
        return
    finish_reason = response.get("meta_info", {}).get("finish_reason")
    message = finish_reason.get("message") if isinstance(finish_reason, dict) else None
    status_code = (
        finish_reason.get("status_code") if isinstance(finish_reason, dict) else None
    )
    raise TranscriptionBackendAborted(
        message or "ASR backend request aborted", status_code=status_code
    )


def apply_cumulative_transcript(
    state: StreamingASRState, text: str, *, is_last: bool, emitted_text: str
) -> str:
    """Apply a cumulative hypothesis to the shared rollback state."""
    if is_last:
        state.full_transcript = text
        return state.finalize(emitted_text=emitted_text)
    return state.update(text, emitted_text=emitted_text)
