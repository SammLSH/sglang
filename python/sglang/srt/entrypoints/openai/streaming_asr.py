import asyncio
import io
import logging
import math
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

    Parameters are model-specific and should be provided via the
    adapter's ``chunked_streaming_config``.

    Rollback compares script-aware units (see ``split_units``): whitespace
    words for Latin-like scripts and single characters for scripts written
    without spaces, so ``unfixed_token_num`` holds back that many units.
    """

    chunk_size_sec: float
    unfixed_chunk_num: int
    unfixed_token_num: int
    confirmed_text: str = ""
    # Monotonic accumulator; used as prompt prefix so the model sees a
    # natural continuation point, not the rolled-back ``confirmed_text``.
    emitted_text: str = ""
    full_transcript: str = ""
    chunk_index: int = 0

    def get_prefix_text(self) -> str:
        if self.chunk_index < self.unfixed_chunk_num or not self.emitted_text:
            return ""
        return self.emitted_text

    def _record_emit(self, delta: str) -> str:
        if delta:
            self.emitted_text = join_text(self.emitted_text, delta).strip()
        return delta

    def update(self, new_transcript: str) -> str:
        old_confirmed = self.confirmed_text
        words = split_units(new_transcript)
        if len(words) > self.unfixed_token_num:
            self.confirmed_text = join_units(words[: -self.unfixed_token_num])
        else:
            self.confirmed_text = ""
        self.full_transcript = new_transcript
        self.chunk_index += 1
        if self.confirmed_text.startswith(old_confirmed):
            return self._record_emit(self.confirmed_text[len(old_confirmed) :].strip())
        # Model revised earlier text, use word level common prefix to avoid
        # re-emitting already-sent content and cutting mid-word.
        old_words = split_units(old_confirmed)
        new_words = split_units(self.confirmed_text)
        common_count = 0
        for ow, nw in zip(old_words, new_words):
            if ow != nw:
                break
            common_count += 1
        return self._record_emit(join_units(new_words[common_count:]))

    def finalize(self) -> str:
        confirmed_words = split_units(self.confirmed_text)
        all_words = split_units(self.full_transcript)
        # Use word level common prefix to handle punctuation differences
        # between intermediate chunks and the final full transcription.
        common_count = 0
        for cw, aw in zip(confirmed_words, all_words):
            if cw != aw:
                break
            common_count += 1
        self.confirmed_text = self.full_transcript
        if common_count == 0 and confirmed_words and all_words:
            return self._record_emit(self.full_transcript)
        return self._record_emit(join_units(all_words[common_count:]))


# Bound the number of cumulative backend decodes per upload. With the default
# two-second chunks this permits over 34 minutes of input.
MAX_ASR_STREAMING_CHUNKS = 1024


@dataclass
class AudioChunks:
    """Decoded audio with cumulative WAV prefixes encoded only on demand."""

    data: np.ndarray
    sample_rate: int
    chunk_size_samples: int

    def __len__(self) -> int:
        return (len(self.data) + self.chunk_size_samples - 1) // self.chunk_size_samples

    def __getitem__(self, index: int) -> bytes:
        index = range(len(self))[index]
        end = min((index + 1) * self.chunk_size_samples, len(self.data))
        buf = io.BytesIO()
        sf.write(buf, self.data[:end], self.sample_rate, format="WAV")
        return buf.getvalue()


def split_audio_chunks(audio_data: bytes, chunk_size_sec: float) -> AudioChunks:
    """Validate the split and decode once, without retaining all prefix WAVs."""
    if not audio_data:
        raise ValueError("audio_data is empty")
    if not math.isfinite(chunk_size_sec) or chunk_size_sec <= 0:
        raise ValueError("chunk_size_sec must be finite and positive")
    try:
        with sf.SoundFile(io.BytesIO(audio_data)) as audio_file:
            sample_rate = audio_file.samplerate
            chunk_samples = chunk_size_sec * sample_rate
            if not math.isfinite(chunk_samples):
                raise ValueError("chunk_size_sec must resolve to a finite sample count")
            chunk_size_samples = int(chunk_samples)
            if chunk_size_samples < 1:
                raise ValueError("chunk_size_sec must cover at least one audio sample")
            total_samples = len(audio_file)
            if not total_samples:
                raise ValueError("audio_data contains no audio samples")
            num_chunks = (total_samples + chunk_size_samples - 1) // chunk_size_samples
            if num_chunks > MAX_ASR_STREAMING_CHUNKS:
                raise ValueError(
                    f"Audio requires {num_chunks} streaming chunks; the limit is "
                    f"{MAX_ASR_STREAMING_CHUNKS}. Increase chunk_size_sec."
                )
            data = audio_file.read(dtype="float32")
    except sf.LibsndfileError as e:
        raise ValueError(f"failed to decode audio: {e}") from e
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    return AudioChunks(data, sample_rate, chunk_size_samples)


def normalize_whitespace(text: str) -> str:
    return _PUNCT_WS_RE.sub(r"\1", text)


_NO_SPACE_BEFORE = frozenset(".,!?;:%)]}，。！？；：、）】》」』")
_NO_SPACE_AFTER = frozenset("([{（【《「『")


def is_cjk_char(c: str) -> bool:
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
    if is_cjk_char(prev[-1]) and is_cjk_char(cur[0]):
        return False
    return True


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
            if is_cjk_char(char):
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
    prev_script = is_cjk_char(prev_char)
    cur_script = is_cjk_char(cur_char)
    if not prev_script and not cur_script:
        # Plain whitespace-delimited text: always one space, like " ".join.
        return True
    if prev_script and cur_script:
        return False
    return needs_space(prev, cur)


def join_text(left: str, right: str) -> str:
    """Join two text fragments with a boundary space only where scripts need it."""
    if not left or not right:
        return left or right
    return f"{left} {right}" if _space_between(left, right) else f"{left}{right}"


def join_units(units: Sequence[str]) -> str:
    """Inverse of ``split_units`` up to the boundary spaces it cannot recover."""
    text = ""
    for unit in units:
        if unit:
            text = join_text(text, unit)
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


class ASRBackendAborted(RuntimeError):
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


async def generate_asr_transcript(
    tokenizer_manager: TokenizerManager,
    adapter: TranscriptionAdapter,
    audio_data: Union[bytes, np.ndarray],
    sampling_params: Dict[str, Any],
    prompt: str,
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
    ``decoder_prefix`` is transcript text appended to ``prompt`` for the model
    to continue; the adapter is told so its snapshots are not held back
    waiting for a leading marker the continuation never carries.
    """
    stream = on_update is not None
    chunk_request = GenerateReqInput(
        text=prompt + decoder_prefix,
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
    try:
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
                    raise ASRBackendAborted(
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
    except asyncio.CancelledError:
        raise
    except ValueError:
        logger.warning("[streaming_asr] ASR request failed", exc_info=True)
        raise

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
    raise ASRBackendAborted(
        message or "ASR backend request aborted", status_code=status_code
    )


async def generate_cumulative_transcript(
    tokenizer_manager: TokenizerManager,
    adapter: TranscriptionAdapter,
    state: StreamingASRState,
    audio_data: Union[bytes, np.ndarray],
    sampling_params: Dict[str, Any],
    raw_request: Optional[Request] = None,
    routing_key: Optional[str] = None,
    routed_dp_rank: Optional[int] = None,
    on_update: Optional[Callable[[str], Awaitable[None]]] = None,
) -> Optional[GeneratedTranscript]:
    """Generate one cumulative hypothesis with the rollback state's prefix."""
    return await generate_asr_transcript(
        tokenizer_manager=tokenizer_manager,
        adapter=adapter,
        audio_data=audio_data,
        sampling_params=sampling_params,
        prompt=adapter.prompt_template,
        decoder_prefix=state.get_prefix_text(),
        raw_request=raw_request,
        routing_key=routing_key,
        routed_dp_rank=routed_dp_rank,
        on_update=on_update,
    )


def apply_cumulative_transcript(
    state: StreamingASRState, text: str, *, is_last: bool
) -> str:
    """Apply a cumulative hypothesis to the shared rollback state."""
    if is_last:
        state.full_transcript = text
        return state.finalize()
    return state.update(text)


async def process_asr_chunk(
    tokenizer_manager: TokenizerManager,
    adapter: TranscriptionAdapter,
    state: StreamingASRState,
    audio_data: Union[bytes, np.ndarray],
    sampling_params: Dict[str, Any],
    is_last: bool,
    raw_request: Optional[Request] = None,
    routing_key: Optional[str] = None,
) -> str:
    """Run and reconcile one cumulative chunk for HTTP streaming ASR."""
    result = await generate_cumulative_transcript(
        tokenizer_manager=tokenizer_manager,
        adapter=adapter,
        state=state,
        audio_data=audio_data,
        sampling_params=sampling_params,
        raw_request=raw_request,
        routing_key=routing_key,
    )
    if result is None:
        return ""
    return apply_cumulative_transcript(state, result.text, is_last=is_last)
