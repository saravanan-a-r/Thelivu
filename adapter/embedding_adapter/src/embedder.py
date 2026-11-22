"""
Using the embedding adapter: text in, vectors out.

    embedder = TextEmbedder.from_pretrained(base_folder, adapter_folder)
    queries = embedder.encode_queries(["how do I reset my password"])
    passages = embedder.encode_passages(["Open Settings, then ..."])
    scores = queries @ passages.T

Only the encoder of the base model is loaded. `load_decoder=True` loads the
whole base model instead (`embedder.model`), for callers that also generate
text with it; `embedder.base_behaviour()` then switches the embedding adapter
off around the calls that must see the plain base encoder.

Late chunking is built in. `token_vectors` returns the final-layer vector of
every token of a document together with the characters it covers, and
`late_chunk` turns them into one vector per chunk: the document is encoded
whole, so each chunk's vector knows the text around it, and a chunk is the
mean of its own tokens only. A document longer than one window is encoded in
overlapping windows and every token takes its vector from the window in
which it lies furthest from an edge.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import torch.nn as nn

from .adapter_io import load_adapter
from .base_model import BaseModel, load_base
from .chunking import chunk_spans
from .encoding import encode_tokens
from .errors import AdapterError, DataError
from .lora import EncoderAdapter
from .pooling import hidden_states, normalize
from .text import TextTokenizer, format_passage

# The contract of an encoder without an adapter (plain mean pooling of the base).
BASE_CONTRACT: dict[str, Any] = {
    "query_prefix": "query: ",
    "passage_prefix": "passage: ",
    "max_query_tokens": 64,
    "eval_max_tokens": 512,
    "matryoshka_dims": [],
    "late_chunking": {"window_tokens": 2048, "overlap_tokens": 256, "chunk_tokens": 256},
}


@dataclass(frozen=True)
class Runtime:
    """Where and how the encoder runs at inference."""

    device: torch.device
    bf16: bool
    encode_tokens: int  # padded tokens per batch


@dataclass
class TokenVectors:
    """The final-layer vectors of one document's own tokens (prefix, title and EOS left out)."""

    text: str
    vectors: torch.Tensor  # float32 [tokens, d_model], not normalised
    spans: np.ndarray  # int64 [tokens, 2], characters of `text` each token covers
    windows: int  # encoder passes it took

    def __len__(self) -> int:
        return len(self.spans)

    def pool(self, start: int, end: int) -> torch.Tensor:
        """Mean vector of the tokens whose last character lies in text[start:end]."""
        ends = self.spans[:, 1]
        first = int(np.searchsorted(ends, start, side="right"))
        last = int(np.searchsorted(ends, end, side="right"))
        if last <= first:
            raise DataError(f"no token ends inside characters {start}:{end}")
        return self.vectors[first:last].mean(dim=0)


@dataclass
class TextChunk:
    text: str
    start: int  # characters of the document
    end: int
    tokens: int
    vector: torch.Tensor  # float32 [dim]


class TextEmbedder:
    def __init__(
        self,
        base: BaseModel,
        adapter: EncoderAdapter | None,
        tokenizer: TextTokenizer,
        contract: dict[str, Any],
        runtime: Runtime,
    ) -> None:
        self.base = base
        self.adapter = adapter
        self.tokenizer = tokenizer
        self.contract = contract
        self.runtime = runtime
        self.base.encoder.eval()

    @classmethod
    def from_pretrained(
        cls,
        base_checkpoint: str | Path,
        adapter: str | Path | None = None,
        *,
        load_decoder: bool = False,
        device: str = "auto",
        batch_tokens: int | None = None,
    ) -> "TextEmbedder":
        """
        `adapter=None` gives the plain base encoder (the zero-training baseline).
        `load_decoder=False` loads the encoder only; True loads the whole base model.
        `device`: "auto" (GPU when there is one), "gpu"/"cuda", or "cpu".
        """
        if device == "auto":
            device = "gpu" if torch.cuda.is_available() else "cpu"
        if device not in ("gpu", "cuda", "cpu"):
            raise AdapterError(f"device must be auto, gpu or cpu, got {device!r}")
        on_gpu = device != "cpu"
        if on_gpu and not torch.cuda.is_available():
            raise AdapterError("no CUDA device is visible; use device='cpu'")
        runtime = Runtime(
            device=torch.device("cuda", 0) if on_gpu else torch.device("cpu"),
            bf16=on_gpu,
            encode_tokens=batch_tokens or (48_000 if on_gpu else 4_000),
        )
        base = load_base(base_checkpoint, load_decoder=load_decoder, device=runtime.device)
        tokenizer = TextTokenizer.from_checkpoint(
            base_checkpoint, eos_id=base.config.eos_token_id, pad_id=base.config.pad_token_id
        )
        if adapter is None:
            return cls(base, None, tokenizer, dict(BASE_CONTRACT), runtime)
        loaded, document = load_adapter(adapter, base)
        return cls(base, loaded, tokenizer, dict(document["embedding"]), runtime)

    # -- the model ---------------------------------------------------------

    @property
    def encoder(self) -> nn.Module:
        return self.base.encoder

    @property
    def model(self) -> nn.Module:
        """The whole base model (encoder + decoder); needs `load_decoder=True`."""
        if self.base.model is None:
            raise AdapterError("only the encoder is loaded; pass load_decoder=True to load the whole model")
        return self.base.model

    @property
    def dimension(self) -> int:
        return self.base.config.d_model

    @property
    def supported_dims(self) -> list[int]:
        """The vector sizes the adapter was trained for, largest first; any `dim` argument must be one of them."""
        trained = {int(dim) for dim in self.contract.get("matryoshka_dims", [])}
        return sorted(trained | {self.dimension}, reverse=True)

    @property
    def late_chunking_defaults(self) -> dict[str, int]:
        """The window, overlap and chunk size (in tokens) that late chunking uses when a call sets none."""
        window, overlap = self._late_settings()
        chunk = int(self.contract["late_chunking"]["chunk_tokens"])
        return {"window_tokens": window, "overlap_tokens": overlap, "chunk_tokens": chunk}

    @contextmanager
    def base_behaviour(self) -> Iterator[None]:
        """Inside the block the encoder is the plain base encoder (adapter switched off)."""
        if self.adapter is None:
            yield
            return
        with self.adapter.disabled():
            yield

    # -- one vector per text -----------------------------------------------

    def encode_queries(
        self, texts: Sequence[str], *, dim: int | None = None, max_tokens: int | None = None
    ) -> torch.Tensor:
        limit = max_tokens or self.contract["max_query_tokens"]
        return self._encode(list(texts), self.contract["query_prefix"], limit, dim)

    def encode_passages(
        self,
        texts: Sequence[str],
        titles: Sequence[str] | None = None,
        *,
        dim: int | None = None,
        max_tokens: int | None = None,
    ) -> torch.Tensor:
        texts = list(texts)
        if titles is not None:
            if len(titles) != len(texts):
                raise DataError(f"{len(texts)} texts but {len(titles)} titles")
            texts = [format_passage(title, text) for title, text in zip(titles, texts)]
        limit = max_tokens or self.contract["eval_max_tokens"]
        return self._encode(texts, self.contract["passage_prefix"], limit, dim)

    def _encode(self, texts: list[str], prefix: str, max_tokens: int, dim: int | None) -> torch.Tensor:
        """Unit vectors, float32 [texts, dim], on the CPU."""
        self._check_dim(dim)
        if max_tokens > self.base.config.max_encoder_length:
            raise DataError(
                f"max_tokens {max_tokens} exceeds the model's context {self.base.config.max_encoder_length}"
            )
        if not texts:
            return torch.empty((0, dim or self.dimension))
        tokens = self.tokenizer.encode_batch(texts, prefix, max_tokens)
        vectors = encode_tokens(self.encoder, tokens, self.runtime, unit=False)
        return normalize(vectors, dim).cpu()

    def truncate(self, vectors: torch.Tensor | np.ndarray, dim: int) -> torch.Tensor:
        """
        Cut vectors you already hold to a smaller trained size, unit length again.

        For an index or cache built at a larger size, such as the full 1,024:
        keeps the first `dim` components and normalises them, which is what
        `encode_*(dim=...)` does. Works on one vector or a batch, and on vectors
        that were already cut once (512 to 256). `dim` must be in `supported_dims`.
        """
        self._check_dim(_whole_number("dim", dim))
        tensor = torch.as_tensor(vectors, dtype=torch.float32)
        if tensor.ndim == 0 or tensor.shape[-1] < dim:
            raise DataError(f"the vectors are only {tensor.shape[-1] if tensor.ndim else 0} wide, cannot cut to {dim}")
        return normalize(tensor, dim)

    def matryoshka(
        self, vectors: torch.Tensor | np.ndarray, dims: Sequence[int] | None = None
    ) -> dict[int, torch.Tensor]:
        """
        Several sizes of the same vectors at once, each one unit length: {dim: vectors}.

        `dims` picks the sizes, in the order given. Without it, every supported
        size that fits the vectors is returned, largest first, which is handy for
        keeping a coarse and a fine index side by side.
        """
        tensor = torch.as_tensor(vectors, dtype=torch.float32)
        if dims is None:
            dims = [dim for dim in self.supported_dims if tensor.ndim and dim <= tensor.shape[-1]]
        return {dim: self.truncate(tensor, dim) for dim in dims}

    def _check_dim(self, dim: int | None) -> None:
        if dim is not None and dim not in self.supported_dims:
            raise DataError(f"dim {dim} was not trained; this adapter supports {self.supported_dims}")

    # -- one vector per token (late chunking) ------------------------------

    def _late_settings(
        self, window_tokens: int | None = None, overlap_tokens: int | None = None
    ) -> tuple[int, int]:
        """
        The window and the overlap, in tokens, that late chunking runs with: the
        caller's values where given, the adapter's own otherwise.
        """
        settings = self.contract["late_chunking"]
        limit = self.base.config.max_encoder_length
        window = min(int(settings["window_tokens"]), limit)
        if window_tokens is not None:
            window = _whole_number("window_tokens", window_tokens)
            if not 32 <= window <= limit:
                raise DataError(f"window_tokens must be from 32 to {limit} (the model's context), got {window}")
        if overlap_tokens is None:
            overlap = int(settings["overlap_tokens"])
            if window_tokens is not None and not 0 <= overlap < window // 2:
                raise DataError(
                    f"the adapter's overlap of {overlap} tokens is too large for a window of {window} tokens; "
                    f"pass an overlap_tokens below {window // 2}"
                )
            return window, overlap
        overlap = _whole_number("overlap_tokens", overlap_tokens)
        if not 0 <= overlap < window // 2:
            raise DataError(
                f"overlap_tokens must be from 0 to {window // 2 - 1} for a window of {window} tokens, got {overlap}"
            )
        return window, overlap

    @torch.no_grad()
    def token_vectors(
        self,
        text: str,
        *,
        title: str = "",
        kind: str = "passage",
        window_tokens: int | None = None,
        overlap_tokens: int | None = None,
    ) -> TokenVectors:
        """
        Final-layer hidden vectors of every token of `text`, however long it is.

        The title, when given, is context for the encoder but not part of the
        returned tokens. Vectors are on the CPU, float32, not normalised.
        The encoder reads `window_tokens` tokens at once, and a longer text is
        encoded in windows that share `overlap_tokens` tokens. Both default to
        the adapter's own settings; the window can be at most the model's context.
        """
        if kind not in ("passage", "query"):
            raise DataError(f"kind must be 'passage' or 'query', got {kind!r}")
        if not text.strip():
            raise DataError("the text is empty")
        window, overlap = self._late_settings(window_tokens, overlap_tokens)
        head_text = self.contract[f"{kind}_prefix"] + (f"{title}. " if title else "")
        encoded = self.tokenizer.encode_with_spans(text, head_text)
        head = encoded.ids[: encoded.content.start]
        content = encoded.ids[encoded.content]
        spans = encoded.spans[encoded.content]
        capacity = window - len(head) - 1  # content tokens per window (EOS takes one)
        if capacity <= 2 * overlap:
            raise DataError(
                f"a window of {window} tokens holds {capacity} tokens of text after the {len(head)} tokens of "
                f"prefix and title, and that must be more than twice the overlap ({overlap}); "
                f"use a smaller overlap_tokens or a shorter title"
            )
        eos = np.asarray([self.tokenizer.eos_id], dtype=np.int64)
        starts = window_starts(len(content), capacity, overlap)
        vectors = torch.empty((len(content), self.dimension), dtype=torch.float32)
        encoder = self.encoder
        encoder.eval()
        for number, start in enumerate(starts):
            ids = np.concatenate([head, content[start : start + capacity], eos])
            input_ids = torch.from_numpy(ids).unsqueeze(0).to(self.runtime.device)
            hidden = hidden_states(encoder, input_ids, torch.ones_like(input_ids), self.runtime.bf16)[0]
            own = hidden[len(head) : -1]
            # Of the tokens two windows share, the first half stays with the earlier window.
            first = 0 if number == 0 else (starts[number - 1] + capacity + start) // 2
            last = len(content) if number == len(starts) - 1 else (start + capacity + starts[number + 1]) // 2
            vectors[first:last] = own[first - start : last - start].cpu()
        return TokenVectors(text=text, vectors=vectors, spans=spans, windows=len(starts))

    def late_chunk(
        self,
        text: str,
        *,
        title: str = "",
        chunk_tokens: int | None = None,
        spans: Sequence[tuple[int, int]] | None = None,
        dim: int | None = None,
        window_tokens: int | None = None,
        overlap_tokens: int | None = None,
    ) -> list[TextChunk]:
        """
        Split a document into chunks and embed each one in the context of the whole.

        `spans` are the caller's own chunk boundaries as (start, end) character
        offsets; without them the built-in splitter makes chunks of whole
        sentences, about `chunk_tokens` tokens each. Vectors are unit length
        and comparable with `encode_queries` output. `window_tokens` and
        `overlap_tokens` set the encoder windows of a long document, as in
        `token_vectors`.
        """
        self._check_dim(dim)
        tokens = self.token_vectors(text, title=title, window_tokens=window_tokens, overlap_tokens=overlap_tokens)
        ends = tokens.spans[:, 1]
        if spans is None:
            size = chunk_tokens or int(self.contract["late_chunking"]["chunk_tokens"])
            if size < 1:
                raise DataError("chunk_tokens must be positive")
            spans = chunk_spans(text, ends, size)
        chunks = []
        for start, end in spans:
            if not 0 <= start < end <= len(text):
                raise DataError(f"chunk span ({start}, {end}) is outside the text (length {len(text)})")
            count = int(np.searchsorted(ends, end, side="right") - np.searchsorted(ends, start, side="right"))
            if count == 0:  # white space only: nothing to embed
                continue
            vector = normalize(tokens.pool(start, end), dim)
            chunks.append(TextChunk(text[start:end], start, end, count, vector))
        return chunks


def _whole_number(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise DataError(f"{name} must be a whole number, got {value!r}")
    return int(value)


def window_starts(tokens: int, capacity: int, overlap: int) -> list[int]:
    """Start of every window over `tokens` content tokens; the last window ends at the end."""
    if tokens <= capacity:
        return [0]
    stride = capacity - overlap
    starts = list(range(0, tokens - capacity, stride))
    starts.append(tokens - capacity)
    return starts
