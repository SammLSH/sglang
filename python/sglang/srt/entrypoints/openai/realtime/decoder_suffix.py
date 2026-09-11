"""Bounded decoder prefix and suffix reconciliation for windowed realtime ASR.

Once encoder windowing is active, each request continues a bounded text prefix.
Common units from consecutive candidates are published minus a small holdback.
When the entire pending candidate agrees, its held tail also enters the next
decoder prefix so it survives audio compaction without forcing publication.
Generated text is a suffix, including genuine repeated speech.
"""

from __future__ import annotations

import msgspec

from sglang.srt.entrypoints.openai.streaming_asr import (
    common_unit_prefix,
    is_cjk_char,
    join_text,
    join_units,
    split_units,
)

# Character budget for the tokenizer round trip in bounded_prefix; generous
# enough that any real tokenizer needs fewer characters than this per token.
_MAX_CHARS_PER_TOKEN = 64


class SuffixUpdate(msgspec.Struct, frozen=True):
    """One reconcile result, applied only after its request is accepted."""

    delta: str
    pending: str
    # Character prefix of pending confirmed by agreement but held from publication.
    confirmed_pending_chars: int = 0
    # True when the decode produced no new text; the caller decides whether
    # to defer the audio or accept the silence.
    empty_continuation: bool = False


def _align_to_unit_boundary(source: str, tail: str) -> str:
    """Drop a leading partial unit introduced by token-level slicing."""
    tail = tail.lstrip()
    if not tail:
        return ""
    if source.endswith(tail):
        start = len(source) - len(tail)
        if (
            start == 0
            or source[start - 1].isspace()
            or is_cjk_char(source[start - 1])
            or is_cjk_char(tail[0])
        ):
            return tail
    units = split_units(tail)
    if len(units) <= 1:
        return ""
    return tail[len(units[0]) :].lstrip()


class DecoderSuffixState(msgspec.Struct):
    """Authoritative transcript state while encoder windowing is active."""

    # Everything published to the client so far.
    emitted_text: str
    # Unpublished text; its confirmed prefix is also supplied to the decoder.
    pending: str = ""
    confirmed_pending_chars: int = 0

    @property
    def confirmed_pending(self) -> str:
        return self.pending[: self.confirmed_pending_chars]

    def bounded_prefix(self, tokenizer, max_tokens: int) -> str:
        """The most recent confirmed text, at most ``max_tokens`` tokens long,
        starting on a unit boundary."""
        source = join_text(self.emitted_text, self.confirmed_pending)
        if not source or max_tokens <= 0:
            return ""
        tail = source[-max_tokens * _MAX_CHARS_PER_TOKEN :]
        token_ids = tokenizer.encode(tail, add_special_tokens=False)
        if len(token_ids) > max_tokens:
            tail = tokenizer.decode(
                token_ids[-max_tokens:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        tail = _align_to_unit_boundary(source, tail)
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
        """Compute the publishable delta for one decode without mutating."""
        if not split_units(continuation):
            if is_last:
                return SuffixUpdate(delta=join_text("", self.pending), pending="")
            return SuffixUpdate(
                delta="",
                pending=self.pending,
                confirmed_pending_chars=self.confirmed_pending_chars,
                empty_continuation=True,
            )
        # The decoder already received the confirmed holdback. Preserve exact
        # character continuation: "car" + "pet" differs from "car" + " pet".
        candidate = self.confirmed_pending + continuation
        continuation_units = split_units(candidate)
        if is_last:
            return SuffixUpdate(
                delta=join_text("", candidate),
                pending="",
            )
        pending_units = split_units(self.pending)
        if not pending_units:
            return SuffixUpdate(delta="", pending=join_units(continuation_units))
        agreed = common_unit_prefix(pending_units, continuation_units, normalized=True)
        emit = max(0, agreed - holdback_units)
        pending = join_units(continuation_units[emit:])
        # Keep an accepted character prefix even when new audio extends its
        # final word; that word still needs agreement before publication.
        confirmed_end = self.confirmed_pending_chars
        if pending_units == continuation_units:
            confirmed_end = len(join_units(continuation_units))
        pending_start = len(join_units(continuation_units)) - len(pending)
        return SuffixUpdate(
            delta=join_text("", join_units(continuation_units[:emit])),
            pending=pending,
            confirmed_pending_chars=max(0, confirmed_end - pending_start),
        )

    def apply(self, update: SuffixUpdate) -> None:
        if update.delta:
            self.emitted_text = join_text(self.emitted_text, update.delta)
        self.pending = update.pending
        self.confirmed_pending_chars = update.confirmed_pending_chars

    def flush(self) -> str:
        """Publish the pending tail when the item commits without new audio."""
        delta = join_text("", self.pending)
        if delta:
            self.emitted_text = join_text(self.emitted_text, delta)
        self.pending = ""
        self.confirmed_pending_chars = 0
        return delta
