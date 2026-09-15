from __future__ import annotations

import msgspec
from transformers import PreTrainedTokenizerBase

from sglang.srt.entrypoints.openai.streaming_asr import (
    _is_cjk,
    common_unit_prefix,
    iter_unit_spans,
    split_units,
)

# Limit tokenizer input work; the actual token budget is checked after truncation.
MAX_CHARS_PER_TOKEN = 64


class SuffixUpdate(msgspec.Struct, frozen=True):
    """One reconcile result, applied only after its request is accepted."""

    # Exact append to the step's published text, including boundary whitespace.
    delta: str
    pending: str
    # Character prefix of pending confirmed by agreement but held from publication.
    confirmed_pending_chars: int = 0
    # True when the decode produced no new text; the caller decides whether
    # to defer the audio or accept the silence.
    empty_continuation: bool = False


def align_to_unit_boundary(source: str, tail: str) -> str:
    """Drop a leading partial unit introduced by token-level slicing."""
    tail = tail.lstrip()
    if not tail:
        return ""
    else:
        start = len(source) - len(tail)
    if source.endswith(tail) and (
        start == 0
        or source[start - 1].isspace()
        or _is_cjk(source[start - 1])
        or _is_cjk(tail[0])
    ):
        return tail
    else:
        units = split_units(tail)
        return tail[len(units[0]) :].lstrip() if len(units) > 1 else ""


class TranscriptionSuffixState(msgspec.Struct):
    """Unpublished continuation and its confirmed prefix; published text is external."""

    # Exact unpublished suffix, including any separator after published text.
    # Its confirmed prefix is also supplied to the decoder.
    pending: str = ""
    confirmed_pending_chars: int = 0

    @property
    def confirmed_pending(self) -> str:
        return self.pending[: self.confirmed_pending_chars]

    def bounded_prefix(
        self, tokenizer: PreTrainedTokenizerBase, max_tokens: int, *, emitted_text: str
    ) -> str:
        """Bound recent confirmed context by tokens, retaining whole text units."""
        if max_tokens <= 0:
            return ""
        else:
            max_chars = max_tokens * MAX_CHARS_PER_TOKEN
        # Retain one predecessor character for partial-word boundary checks.
        pending_start = max(0, self.confirmed_pending_chars - max_chars - 1)
        confirmed_tail = self.pending[pending_start : self.confirmed_pending_chars]
        history_chars = max_chars + 1 - len(confirmed_tail)
        source = (
            emitted_text[-history_chars:] + confirmed_tail
            if history_chars
            else confirmed_tail
        )
        if not source:
            return ""
        else:
            tail = source[-max_chars:]
        token_ids = tokenizer.encode(tail, add_special_tokens=False)
        tail = (
            tokenizer.decode(
                token_ids[-max_tokens:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if len(token_ids) > max_tokens
            else tail
        )
        tail = align_to_unit_boundary(source, tail)
        # Removing a leading space or partial unit can change tokenization.
        # Check the actual prefix, dropping whole units until it fits.
        while (
            tail and len(tokenizer.encode(tail, add_special_tokens=False)) > max_tokens
        ):
            first_unit = split_units(tail)[0]
            tail = tail[len(first_unit) :].lstrip()
        return tail

    def reconcile(
        self,
        continuation: str,
        *,
        is_last: bool,
        holdback_units: int,
    ) -> SuffixUpdate:
        """Compare text units, then slice exact delta and pending text without mutating."""
        if not continuation.strip():
            if is_last:
                return SuffixUpdate(delta=self.pending, pending="")
            else:
                return SuffixUpdate(
                    delta="",
                    pending=self.pending,
                    confirmed_pending_chars=self.confirmed_pending_chars,
                    empty_continuation=True,
                )
        else:
            # The decoder already received the holdback; "car" + "pet" is "carpet".
            candidate = self.confirmed_pending + continuation
        if is_last:
            return SuffixUpdate(delta=candidate, pending="")
        else:
            spans = list(iter_unit_spans(candidate))
        continuation_units = [candidate[start:end] for start, end in spans]
        pending_units = split_units(self.pending)
        if not pending_units:
            return SuffixUpdate(delta="", pending=candidate)
        else:
            agreed = common_unit_prefix(
                pending_units, continuation_units, normalized=True
            )
        emit = max(0, agreed - holdback_units)
        # Leave the separator after the last emitted unit with the pending tail.
        pending_start = spans[emit - 1][1] if emit else 0
        # Keep an accepted character prefix even when new audio extends its
        # final word; that word still needs agreement before publication.
        confirmed_end = (
            len(candidate)
            if pending_units == continuation_units
            else self.confirmed_pending_chars
        )
        return SuffixUpdate(
            delta=candidate[:pending_start],
            pending=candidate[pending_start:],
            confirmed_pending_chars=max(0, confirmed_end - pending_start),
        )

    def flush(self) -> str:
        """Release the pending tail for the caller to publish."""
        delta = self.pending
        self.pending = ""
        self.confirmed_pending_chars = 0
        return delta
