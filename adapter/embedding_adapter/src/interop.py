"""
Importing the base model package.

`model/` calls its code directory `src`, like every module of this repo, so it
is loaded by path and registered under the name the other modules use
(`thelivu_model`). A process that has already loaded it reuses that copy, which
keeps `isinstance` checks valid across modules.

This is the only file of the embedding adapter that knows where the base model
code lives; nothing else in the repo is imported.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

# adapter/embedding_adapter/src/interop.py -> src -> embedding_adapter -> adapter -> root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_SRC = PROJECT_ROOT / "model" / "src"


def _load(name: str, source: Path) -> ModuleType:
    """Load a `src` directory as a top-level package under `name`."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, source / "__init__.py", submodule_search_locations=[str(source)]
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {source}")
    module = importlib.util.module_from_spec(spec)
    # Registered before execution so the package's relative imports resolve
    # against this name instead of triggering a second load.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


base_model_pkg = _load("thelivu_model", MODEL_SRC)
