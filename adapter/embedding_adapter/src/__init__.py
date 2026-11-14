"""
Embedding adapter for the Thelivu base model.

An encoder-only LoRA that turns the base encoder into a text-embedding model:
mean-pooled, L2-normalised 1024-d vectors (usable at 512 / 256 / 128 too), the
final-layer token vectors, and late chunking built on them.

The module depends on the base model package (`model/src`) and nothing else in
this repo.
"""
