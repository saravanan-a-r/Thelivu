"""
Splitting a document into chunks for late chunking.

A chunk is a run of whole sentences of at most `chunk_tokens` tokens.
Paragraph breaks are respected first: a chunk never runs across a blank line
unless a single paragraph is all that would be in it and the next one fits
too. A sentence longer than the budget is cut between tokens.

The splitter works on character offsets and token spans only, so it needs no
language model and never changes the text: the chunks, joined, are the
document.
"""

from __future__ import annotations

import re

import numpy as np

# A sentence ends at . ! ? (plus closing quotes/brackets) followed by white space,
# or at a line break.
_SENTENCE_END = re.compile(r"""(?<=[.!?])["')\]]*\s+|\n+""")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Character spans that cover `text` completely, one per sentence."""
    spans = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        if match.end() > start:
            spans.append((start, match.end()))
            start = match.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def chunk_spans(text: str, token_ends: np.ndarray, chunk_tokens: int) -> list[tuple[int, int]]:
    """
    Character spans of the chunks of `text`.

    `token_ends[i]` is the offset just past the last character of content
    token i, ascending. A token belongs to the span that holds its last
    character.
    """
    if not text:
        return []

    def tokens_in(start: int, end: int) -> int:
        return int(np.searchsorted(token_ends, end, side="right") - np.searchsorted(token_ends, start, side="right"))

    # Sentences, with over-long ones cut between tokens.
    pieces: list[tuple[int, int]] = []
    for start, end in sentence_spans(text):
        while tokens_in(start, end) > chunk_tokens:
            first = int(np.searchsorted(token_ends, start, side="right"))
            cut = int(token_ends[first + chunk_tokens - 1])
            pieces.append((start, cut))
            start = cut
        if end > start:
            pieces.append((start, end))

    breaks = {match.end() for match in _PARAGRAPH_BREAK.finditer(text)}
    chunks: list[tuple[int, int]] = []
    start, end = pieces[0]
    for piece_start, piece_end in pieces[1:]:
        fits = tokens_in(start, piece_end) <= chunk_tokens
        # At a paragraph break, close the chunk once it is at least half full.
        at_break = piece_start in breaks and tokens_in(start, end) * 2 >= chunk_tokens
        if fits and not at_break:
            end = piece_end
        else:
            chunks.append((start, end))
            start, end = piece_start, piece_end
    chunks.append((start, end))
    return chunks
