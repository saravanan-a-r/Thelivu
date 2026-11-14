"""Errors raised by the embedding adapter module."""

from __future__ import annotations


class EmbeddingAdapterError(Exception):
    """Base class: anything this module refuses to do."""


class ConfigError(EmbeddingAdapterError):
    """The YAML config is missing a key, has an unknown one, or holds an invalid value."""


class DataError(EmbeddingAdapterError):
    """The training data or its prepared cache is missing, stale or malformed."""


class AdapterError(EmbeddingAdapterError):
    """An adapter does not fit the base model it is being attached to or loaded for."""


class ResourceError(EmbeddingAdapterError):
    """The requested device cannot be used safely (busy GPU, CUDA missing, ...)."""
