import asyncio
import io
import logging
import re
import unicodedata
from contextlib import aclosing
from dataclasses import dataclass
from http import HTTPStatus
from typing import (
    Awaitable,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Union,
)

import msgspec
import numpy as np
import soundfile as sf
from fastapi import HTTPException, Request
from pydantic import JsonValue

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
    """Keep recent words or CJK characters unsent so later audio can correct them."""

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
        """Save the latest transcription and return text the caller can send."""
        text_unit_spans = list(iter_text_unit_spans(new_transcript))
        confirmed_unit_count = max(0, len(text_unit_spans) - self.unfixed_token_num)
        confirmed_text = (
            new_transcript[: text_unit_spans[confirmed_unit_count - 1][1]]
            if confirmed_unit_count
            else ""
        )
        delta = self.get_unsent_text(confirmed_text, emitted_text=emitted_text)
        self.confirmed_text = confirmed_text
        self.full_transcript = new_transcript
        self.chunk_index += 1
        return delta

    def get_unsent_text(self, text: str, *, emitted_text: str) -> str:
        """Return new text without repeating the part already sent."""
        if text.startswith(emitted_text):
            return text[len(emitted_text) :]
        if emitted_text.startswith(text):
            # A shorter result can leave all confirmed text already sent.
            return ""
        # Without sent text in the prompt, the model may revise earlier words.
        text_unit_spans = list(iter_text_unit_spans(text))
        common_unit_count = count_common_text_units(
            split_text_units(self.confirmed_text),
            [text[start:end] for start, end in text_unit_spans],
        )
        tail_start = (
            text_unit_spans[common_unit_count - 1][1] if common_unit_count else 0
        )
        return join_text(emitted_text, text[tail_start:])[len(emitted_text) :]

    def get_pending_transcript(self, *, emitted_text: str) -> str:
        """Keep unsent text available when finishing the item or switching modes."""
        return self.get_unsent_text(self.full_transcript, emitted_text=emitted_text)

    def finalize(self, *, emitted_text: str) -> str:
        """Return the remaining text now that no more audio can revise it."""
        delta = self.get_pending_transcript(emitted_text=emitted_text)
        self.confirmed_text = self.full_transcript
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
    """Remove spacing artifacts without losing spaces needed between text deltas."""
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
    sampling_params: Dict[str, JsonValue],
    is_last: bool,
    raw_request: Optional[Request] = None,
    routing_key: Optional[str] = None,
    *,
    emitted_text: str,
) -> str:
    """Transcribe one HTTP audio chunk and return text that has not been sent."""
    decoder_prefix = state.get_prefix_text(emitted_text=emitted_text)
    try:
        result = await generate_transcript(
            tokenizer_manager=tokenizer_manager,
            adapter=adapter,
            decoder_prefix=decoder_prefix,
            audio_data=audio_data,
            sampling_params=sampling_params,
            raw_request=raw_request,
            routing_key=routing_key,
        )
    except asyncio.CancelledError:
        raise
    except ValueError:
        logger.warning(
            "[streaming_asr] chunk %d failed", state.chunk_index, exc_info=True
        )
        raise
    if result is None:
        return ""
    return apply_cumulative_transcript(
        state,
        decoder_prefix + result.text,
        is_last=is_last,
        emitted_text=emitted_text,
    )


def iter_text_unit_spans(text: str) -> Iterator[tuple[int, int]]:
    """Find words and individual CJK characters without changing their spacing."""
    start = None
    for index, char in enumerate(text):
        if char.isspace() or _is_cjk(char):
            if start is not None:
                yield start, index
                start = None
            if not char.isspace():
                yield index, index + 1
        elif start is None:
            start = index
    if start is not None:
        yield start, len(text)


def split_text_units(text: str) -> List[str]:
    """Split whitespace-separated words and individual CJK characters."""
    return [text[start:end] for start, end in iter_text_unit_spans(text)]


def join_text(left: str, right: str) -> str:
    """Join transcript parts with spaces where word boundaries require them."""
    parts = [left]
    for word in right.split(" "):
        if word:
            parts.append(f" {word}" if needs_space(parts[-1], word) else word)
    return "".join(parts)


def normalize_text_unit(unit: str) -> str:
    """Ignore case, surrounding punctuation, and Unicode forms when comparing text."""
    normalized = unicodedata.normalize("NFKC", unit)
    start, end = 0, len(normalized)
    while start < end and unicodedata.category(normalized[start])[0] == "P":
        start += 1
    while end > start and unicodedata.category(normalized[end - 1])[0] == "P":
        end -= 1
    return (normalized[start:end] or normalized).lower()


def count_common_text_units(
    left: Sequence[str], right: Sequence[str], *, normalized: bool = False
) -> int:
    """Count matching words or CJK characters at the start of both transcripts."""
    count = 0
    for left_unit, right_unit in zip(left, right):
        matches = (
            normalize_text_unit(left_unit) == normalize_text_unit(right_unit)
            if normalized
            else left_unit == right_unit
        )
        if not matches:
            break
        count += 1
    return count


class TranscriptionBackendAborted(RuntimeError):
    """The backend aborted the request instead of finishing it."""

    def __init__(self, message: str, *, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code

    @property
    def retryable(self) -> bool:
        return self.status_code in (
            HTTPStatus.INTERNAL_SERVER_ERROR,
            HTTPStatus.SERVICE_UNAVAILABLE,
        )


class GeneratedTranscript(msgspec.Struct, frozen=True):
    """Keep the stop reason so callers can retry or reject incomplete transcriptions."""

    text: str
    finish_reason: Optional[str]


async def generate_transcript(
    tokenizer_manager: TokenizerManager,
    adapter: TranscriptionAdapter,
    audio_data: Union[bytes, np.ndarray],
    sampling_params: Dict[str, JsonValue],
    decoder_prefix: str = "",
    raw_request: Optional[Request] = None,
    routing_key: Optional[str] = None,
    mm_processor_kwargs: Optional[Dict[str, JsonValue]] = None,
    on_transcript_update: Optional[Callable[[str], Awaitable[None]]] = None,
) -> Optional[GeneratedTranscript]:
    """Generate transcript text without the decoder prefix supplied in the prompt.

    Each callback receives all text generated so far, allowing revised partial text.
    """
    stream = on_transcript_update is not None
    generation_request = GenerateReqInput(
        text=adapter.prompt_template + decoder_prefix,
        audio_data=audio_data,
        sampling_params=sampling_params,
        stream=stream,
        modalities=["audio"],
        routing_key=routing_key,
        mm_processor_kwargs=mm_processor_kwargs,
    )

    cumulative_text = ""
    incremental = stream and get_serving().incremental_streaming_output
    response = None
    async with aclosing(
        tokenizer_manager.generate_request(generation_request, raw_request)
    ) as responses:
        while True:
            try:
                response = await anext(responses)
            except StopAsyncIteration:
                break
            except HTTPException as error:
                # Match errors returned in streamed responses, while keeping
                # callback failures separate from backend failures.
                if error.status_code not in (
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    HTTPStatus.SERVICE_UNAVAILABLE,
                ):
                    raise
                raise TranscriptionBackendAborted(
                    str(error.detail), status_code=error.status_code
                ) from error
            if get_finish_reason(response) == "abort":
                finish_reason = response.get("meta_info", {}).get("finish_reason")
                details = finish_reason if isinstance(finish_reason, dict) else {}
                raise TranscriptionBackendAborted(
                    details.get("message") or "ASR backend request aborted",
                    status_code=details.get("status_code"),
                )
            if not stream:
                break
            chunk_text = response.get("text") or ""
            cumulative_text = (
                cumulative_text + chunk_text if incremental else chunk_text
            )
            if get_finish_reason(response) == "length":
                # Do not send this incomplete result; the caller may need to retry it.
                continue
            visible_text = adapter.postprocess_streaming_text(
                cumulative_text, continuation=bool(decoder_prefix)
            )
            if visible_text is not None:
                await on_transcript_update(normalize_whitespace(visible_text))

    if response is None:
        logger.warning("[streaming_asr] ASR request returned no response")
        return None

    raw_text = cumulative_text if stream else (response.get("text") or "")
    return GeneratedTranscript(
        text=normalize_whitespace(adapter.postprocess_text(raw_text)),
        finish_reason=get_finish_reason(response),
    )


def get_finish_reason(response: Dict[str, JsonValue]) -> Optional[str]:
    finish_reason = response.get("meta_info", {}).get("finish_reason")
    if isinstance(finish_reason, dict):
        return finish_reason.get("type")
    return finish_reason


def apply_cumulative_transcript(
    state: StreamingASRState, text: str, *, is_last: bool, emitted_text: str
) -> str:
    """Keep recent text available for correction until the final audio request."""
    if is_last:
        state.full_transcript = text
        return state.finalize(emitted_text=emitted_text)
    else:
        return state.update(text, emitted_text=emitted_text)
