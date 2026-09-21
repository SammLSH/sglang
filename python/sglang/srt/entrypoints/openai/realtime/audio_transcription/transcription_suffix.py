from __future__ import annotations

import msgspec
from transformers import PreTrainedTokenizerBase

from sglang.srt.entrypoints.openai.streaming_asr import (
    _is_cjk,
    count_common_text_units,
    iter_text_unit_spans,
    split_text_units,
)

# Limit text passed to the tokenizer so long transcripts do not slow each request.
MAX_CHARS_PER_TOKEN = 64


class TranscriptionSuffixUpdate(msgspec.Struct, frozen=True):
    """Separate text ready to send from text that still needs confirmation."""

    # Preserve whitespace so this text can be appended directly to the transcript.
    delta: str
    pending_text: str
    # Keep confirmed but unsent text available for the next decoder prompt.
    confirmed_pending_text_chars: int = 0
    # The caller decides whether an empty result needs a retry or is silence.
    empty_generated_text: bool = False


def trim_partial_word(confirmed_text: str, decoder_prefix: str) -> str:
    """Avoid starting the decoder prompt in the middle of a word after truncation."""
    decoder_prefix = decoder_prefix.lstrip()
    if not decoder_prefix:
        return ""
    else:
        start = len(confirmed_text) - len(decoder_prefix)
    if confirmed_text.endswith(decoder_prefix) and (
        start == 0
        or confirmed_text[start - 1].isspace()
        or _is_cjk(confirmed_text[start - 1])
        or _is_cjk(decoder_prefix[0])
    ):
        return decoder_prefix
    else:
        units = split_text_units(decoder_prefix)
        return decoder_prefix[len(units[0]) :].lstrip() if len(units) > 1 else ""


class TranscriptionSuffixState(msgspec.Struct):
    """Hold unsent text so later transcriptions can confirm or revise it."""

    # Keep leading whitespace so sending this text preserves word boundaries.
    pending_text: str = ""
    confirmed_pending_text_chars: int = 0

    @property
    def confirmed_pending_text(self) -> str:
        return self.pending_text[: self.confirmed_pending_text_chars]

    def build_decoder_prefix(
        self, tokenizer: PreTrainedTokenizerBase, max_tokens: int, *, emitted_text: str
    ) -> str:
        """Fit recent confirmed text into the decoder token limit without cutting words."""
        max_chars = max_tokens * MAX_CHARS_PER_TOKEN
        # Keep one extra character to check whether truncation cuts a word.
        pending_text_start = max(0, self.confirmed_pending_text_chars - max_chars - 1)
        confirmed_tail = self.pending_text[
            pending_text_start : self.confirmed_pending_text_chars
        ]
        history_chars = max_chars + 1 - len(confirmed_tail)
        confirmed_text = (
            emitted_text[-history_chars:] + confirmed_tail
            if history_chars
            else confirmed_tail
        )
        if not confirmed_text:
            return ""
        else:
            decoder_prefix = confirmed_text[-max_chars:]
        token_ids = tokenizer.encode(decoder_prefix, add_special_tokens=False)
        decoder_prefix = (
            tokenizer.decode(
                token_ids[-max_tokens:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if len(token_ids) > max_tokens
            else decoder_prefix
        )
        decoder_prefix = trim_partial_word(confirmed_text, decoder_prefix)
        # Removing a space or partial word can change the token count.
        while (
            decoder_prefix
            and len(tokenizer.encode(decoder_prefix, add_special_tokens=False))
            > max_tokens
        ):
            first_unit = split_text_units(decoder_prefix)[0]
            decoder_prefix = decoder_prefix[len(first_unit) :].lstrip()
        return decoder_prefix

    def prepare_transcript_update(
        self,
        generated_text: str,
        *,
        is_last: bool,
        holdback_units: int,
        unfixed_units: int | None = None,
    ) -> TranscriptionSuffixUpdate:
        """Compare consecutive transcriptions to choose text ready to send."""
        if not generated_text.strip():
            if is_last:
                return TranscriptionSuffixUpdate(
                    delta=self.pending_text, pending_text=""
                )
            else:
                return TranscriptionSuffixUpdate(
                    delta="",
                    pending_text=self.pending_text,
                    confirmed_pending_text_chars=self.confirmed_pending_text_chars,
                    empty_generated_text=True,
                )
        else:
            # The prompt already contains confirmed text: "car" + "pet" is "carpet".
            transcript_candidate = self.confirmed_pending_text + generated_text
        if is_last:
            return TranscriptionSuffixUpdate(
                delta=transcript_candidate, pending_text=""
            )
        else:
            text_unit_spans = list(iter_text_unit_spans(transcript_candidate))
        candidate_text_units = [
            transcript_candidate[start:end] for start, end in text_unit_spans
        ]
        pending_text_units = split_text_units(self.pending_text)
        agreed_unit_count = count_common_text_units(
            pending_text_units, candidate_text_units, normalized=True
        )
        emitted_unit_count = max(0, agreed_unit_count - holdback_units)
        if unfixed_units is not None:
            # Use the adapter's rollback budget when a revised prefix prevents
            # agreement, so every request need not regenerate the whole candidate.
            emitted_unit_count = max(
                emitted_unit_count, len(candidate_text_units) - unfixed_units
            )
        emitted_unit_count = min(
            emitted_unit_count, max(0, len(candidate_text_units) - holdback_units)
        )
        # Keep the separator with unsent text so the next delta preserves spacing.
        pending_text_start = (
            text_unit_spans[emitted_unit_count - 1][1] if emitted_unit_count else 0
        )
        # Keep confirmed characters when the decoder extends their last word;
        # the extended word still needs confirmation before it can be sent.
        confirmed_text_end = (
            len(transcript_candidate)
            if pending_text_units == candidate_text_units
            else self.confirmed_pending_text_chars
        )
        return TranscriptionSuffixUpdate(
            delta=transcript_candidate[:pending_text_start],
            pending_text=transcript_candidate[pending_text_start:],
            confirmed_pending_text_chars=max(
                0, confirmed_text_end - pending_text_start
            ),
        )

    def flush_pending_transcript(self) -> str:
        """Return and clear unsent text when the audio item is complete."""
        delta = self.pending_text
        self.pending_text = ""
        self.confirmed_pending_text_chars = 0
        return delta
