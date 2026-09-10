"""Bounded decoder prefix and suffix reconciliation for windowed realtime ASR.

Once encoder windowing is active, each backend request decodes only the text
that continues a bounded prefix of what was already published. Successive
continuations of overlapping audio are reconciled with Local Agreement: units
that two consecutive decodes agree on are published, minus a small holdback,
and the rest stays pending. Generated text is a suffix, even when the speaker
repeats words already present in the prompt.
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
    # The unpublished tail of the previous decode, awaiting agreement.
    pending: str = ""

    def bounded_prefix(self, tokenizer, max_tokens: int) -> str:
        """The most recent emitted text, at most ``max_tokens`` tokens long,
        starting on a unit boundary."""
        source = self.emitted_text
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
        continuation_units = split_units(continuation)
        if is_last:
            # The final decode supersedes the provisional tail; if it decoded
            # nothing, publish the held-back tail rather than losing it.
            return SuffixUpdate(
                delta=join_text(
                    "", continuation if continuation_units else self.pending
                ),
                pending="",
            )
        if not continuation_units:
            return SuffixUpdate(delta="", pending=self.pending, empty_continuation=True)
        pending_units = split_units(self.pending)
        if not pending_units:
            return SuffixUpdate(delta="", pending=continuation)
        agreed = common_unit_prefix(pending_units, continuation_units, normalized=True)
        emit = max(0, agreed - holdback_units)
        return SuffixUpdate(
            delta=join_text("", join_units(continuation_units[:emit])),
            pending=join_units(continuation_units[emit:]),
        )

    def apply(self, update: SuffixUpdate) -> None:
        if update.delta:
            self.emitted_text = join_text(self.emitted_text, update.delta)
        self.pending = update.pending

    def flush(self) -> str:
        """Publish the pending tail when the item commits without new audio."""
        delta = join_text("", self.pending)
        if delta:
            self.emitted_text = join_text(self.emitted_text, delta)
        self.pending = ""
        return delta
