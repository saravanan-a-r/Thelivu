"""
Turning text into token ids, the one way the adapter is trained and used.

    ids = tokenizer(prefix + text)  cut to (max_tokens - 1)  + [EOS]

The prefix says what the text is (`query: ` or `passage: `). A passage with a
title is `title + ". " + text`. Nothing else is added and nothing is
lower-cased or stripped: the model sees the text as it is.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from tokenizers import Tokenizer

from .errors import DataError

TOKENIZER_FILE = "tokenizer.json"
TOKEN_DTYPE = np.uint16


def format_passage(title: str, text: str) -> str:
    return f"{title}. {text}" if title else text


def text_hash(text: str) -> int:
    """64-bit identity of a passage text: case and whitespace do not count."""
    normal = " ".join(text.lower().split())
    return int.from_bytes(hashlib.sha1(normal.encode("utf-8")).digest()[:8], "little")


def split_key(salt: str, record_id: str) -> int:
    """Where a record falls in the dev split; the lowest keys of a source are dev."""
    digest = hashlib.sha1((salt + record_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


@dataclass
class Encoded:
    """One tokenised text with the character span of every token."""

    ids: np.ndarray  # int64, prefix tokens first, EOS last
    spans: np.ndarray  # int64 [tokens, 2], offsets into the text; (0, 0) for prefix and EOS
    content: slice  # the tokens that belong to the text itself
    truncated: bool


class TextTokenizer:
    """The base model's tokenizer with the embedding conventions on top."""

    def __init__(self, tokenizer: Tokenizer, eos_id: int, pad_id: int, digest: str) -> None:
        tokenizer.no_truncation()
        tokenizer.no_padding()
        self._tokenizer = tokenizer
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.digest = digest
        if tokenizer.get_vocab_size(with_added_tokens=True) > np.iinfo(TOKEN_DTYPE).max + 1:
            raise DataError("the vocabulary does not fit the 16-bit token cache")

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path, eos_id: int = 1, pad_id: int = 0) -> "TextTokenizer":
        path = Path(checkpoint) / TOKENIZER_FILE
        if not path.is_file():
            raise DataError(f"no {TOKENIZER_FILE} in {checkpoint}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return cls(Tokenizer.from_file(str(path)), eos_id, pad_id, digest)

    def encode_batch(
        self, texts: Sequence[str], prefix: str, max_tokens: int
    ) -> list[np.ndarray]:
        """Token ids (uint16) of `prefix + text`, cut to `max_tokens` with EOS last."""
        encodings = self._tokenizer.encode_batch(
            [prefix + text for text in texts], add_special_tokens=False
        )
        keep = max_tokens - 1
        output = []
        for encoding in encodings:
            ids = encoding.ids[:keep]
            ids.append(self.eos_id)
            output.append(np.asarray(ids, dtype=TOKEN_DTYPE))
        return output

    def encode_with_spans(self, text: str, prefix: str, max_tokens: int | None = None) -> Encoded:
        """
        Tokenise one text and keep each token's character span in `text`.

        `max_tokens=None` keeps every token (late chunking windows long
        documents itself).
        """
        encoding = self._tokenizer.encode(prefix + text, add_special_tokens=False)
        ids, offsets = encoding.ids, encoding.offsets
        truncated = max_tokens is not None and len(ids) > max_tokens - 1
        if truncated:
            ids, offsets = ids[: max_tokens - 1], offsets[: max_tokens - 1]
        shift = len(prefix)
        spans = np.zeros((len(ids) + 1, 2), dtype=np.int64)
        first = len(ids)
        for index, (start, end) in enumerate(offsets):
            if end > shift:  # the token holds at least one character of the text
                first = min(first, index)
                spans[index] = (max(start - shift, 0), end - shift)
        return Encoded(
            ids=np.asarray(ids + [self.eos_id], dtype=np.int64),
            spans=spans,
            content=slice(first, len(ids)),
            truncated=truncated,
        )
