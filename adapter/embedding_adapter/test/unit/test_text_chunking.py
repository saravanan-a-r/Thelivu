"""Tokenising with spans, sentence chunks and long-document windows."""

import numpy as np
import pytest

from conftest import build_tokenizer
from src.chunking import chunk_spans, sentence_spans
from src.embedder import window_starts
from src.text import TextTokenizer, format_passage, text_hash


@pytest.fixture(scope="module")
def tokenizer():
    return TextTokenizer(build_tokenizer(), eos_id=1, pad_id=0, digest="test")


def test_encode_batch_adds_the_prefix_and_eos_and_cuts(tokenizer):
    plain = tokenizer.encode_batch(["the river of paris"], "query: ", 16)[0]
    assert plain[-1] == 1 and len(plain) == 2 + 4 + 1  # "query", ":", four words, EOS
    cut = tokenizer.encode_batch(["the river of paris"], "query: ", 4)[0]
    assert len(cut) == 4 and cut[-1] == 1 and np.array_equal(cut[:3], plain[:3])


def test_spans_point_back_into_the_text(tokenizer):
    text = "the river of paris. the lake of rome!"
    encoded = tokenizer.encode_with_spans(text, "passage: paris river. ")
    assert encoded.ids[-1] == 1 and not encoded.truncated
    assert encoded.content.start == 5  # passage : paris river .
    words = [text[start:end] for start, end in encoded.spans[encoded.content]]
    assert words == ["the", "river", "of", "paris", ".", "the", "lake", "of", "rome", "!"]
    same = tokenizer.encode_batch([text], "passage: paris river. ", 64)[0]
    assert np.array_equal(encoded.ids, same)


def test_format_and_hash():
    assert format_passage("Title", "text") == "Title. text"
    assert format_passage("", "text") == "text"
    assert text_hash("The  River\n") == text_hash("the river")
    assert text_hash("the river") != text_hash("the rivers")


def test_sentence_spans_cover_the_text():
    text = 'One two. Three "four!" Five?\nSix\n\nSeven.'
    spans = sentence_spans(text)
    assert "".join(text[a:b] for a, b in spans) == text
    assert [text[a:b].strip() for a, b in spans] == ["One two.", 'Three "four!"', "Five?", "Six", "Seven."]


def ends_of(text: str) -> np.ndarray:
    """One token per word or punctuation mark."""
    import re

    return np.asarray([match.end() for match in re.finditer(r"\w+|[^\w\s]", text)], dtype=np.int64)


def test_chunks_are_whole_sentences_within_the_budget():
    text = "one two three. four five six. seven eight nine. ten one two."
    ends = ends_of(text)  # 4 tokens per sentence
    chunks = chunk_spans(text, ends, 8)
    assert [text[a:b].strip() for a, b in chunks] == [
        "one two three. four five six.",
        "seven eight nine. ten one two.",
    ]
    assert "".join(text[a:b] for a, b in chunks) == text
    assert chunk_spans(text, ends, 100) == [(0, len(text))]
    assert chunk_spans("", np.zeros(0, dtype=np.int64), 8) == []


def test_a_long_sentence_is_cut_between_tokens():
    text = "one two three four five six seven eight nine ten."
    ends = ends_of(text)
    chunks = chunk_spans(text, ends, 4)
    assert "".join(text[a:b] for a, b in chunks) == text
    counts = [int(((ends > a) & (ends <= b)).sum()) for a, b in chunks]
    assert counts == [4, 4, 3]


def test_a_paragraph_break_closes_a_half_full_chunk():
    text = "one two three. four five six.\n\nseven eight nine. ten one two."
    chunks = chunk_spans(text, ends_of(text), 12)
    assert [text[a:b].strip() for a, b in chunks] == [
        "one two three. four five six.",
        "seven eight nine. ten one two.",
    ]


def test_window_starts_cover_every_token_with_the_overlap():
    assert window_starts(10, 20, 4) == [0]
    assert window_starts(20, 20, 4) == [0]
    starts = window_starts(50, 20, 4)
    assert starts == [0, 16, 30]
    assert starts[-1] + 20 == 50
    for tokens, capacity, overlap in [(21, 20, 4), (100, 20, 4), (4097, 2040, 256), (37, 20, 8)]:
        starts = window_starts(tokens, capacity, overlap)
        assert starts[0] == 0 and starts[-1] + capacity == tokens
        assert all(b > a for a, b in zip(starts, starts[1:]))
        assert all(b - a <= capacity - overlap for a, b in zip(starts, starts[1:]))
