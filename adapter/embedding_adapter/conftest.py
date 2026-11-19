"""
Shared fixtures for the embedding adapter test suite.

Run pytest **from this directory**: every module in this repo calls its code
directory `src`, so each is its own test root.

The tests run a tiny base (2+2 layers, 64 dims, 256-token vocabulary) saved in
the release layout (`config.json`, `model.safetensors`, `tokenizer.json`) and a
few dozen made-up records per source. What is tested (the adapter is a no-op
when new, the base never changes, GradCache equals one big batch, a resumed run
continues with identical tensors, pooling ignores padding) does not depend on
size.

CUDA is hidden before torch loads and PyTorch is capped at 2 threads, so the
suite can run next to a training job without touching the GPU.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cache import prepare_cache  # noqa: E402
from src.config import build_config  # noqa: E402
from src.interop import base_model_pkg  # noqa: E402
from src.resources import DevicePlan  # noqa: E402

REAL_CONFIG = ROOT / "embedding_config.yml"

TINY_MODEL = dict(
    d_model=64,
    d_ff=176,
    num_layers=2,
    num_decoder_layers=2,
    num_heads=4,
    d_kv=16,
    vocab_size=256,
    max_encoder_length=128,
    max_decoder_length=128,
    relative_attention_num_buckets=8,
    relative_attention_max_distance=16,
    layer_norm_epsilon=1e-6,
    feed_forward_proj="gated-gelu",
    tie_word_embeddings=True,
    scale_output_for_tied_embeddings=True,
    scale_attention_scores=False,
    dropout_rate=0.0,
    label_smoothing=0.0,
    initializer_factor=1.0,
    pad_token_id=0,
    decoder_start_token_id=0,
    eos_token_id=1,
    label_pad_token_id=-100,
    profile="tiny",
    expected_parameters=None,
)

WORDS = (
    "query passage the a of in on is are was what who where when why how does do can which "
    "capital city river mountain lake country planet star moon ocean forest desert island bridge "
    "largest smallest oldest newest tallest longest deepest fastest first last famous known "
    "red blue green yellow black white north south east west river bank "
    "alpha beta gamma delta omega sigma kappa lambda theta zeta "
    "cat dog bird fish horse lion tiger bear wolf fox "
    "one two three four five six seven eight nine ten "
    "built found named called located lies flows stands grows lives "
    "paris london rome berlin madrid oslo cairo delhi tokyo lima "
    "title about answer question fact story note record number"
).split()
PUNCTUATION = [".", ",", ":", "?", "!", "(", ")", "-", "'"]


def build_tokenizer() -> Tokenizer:
    """Word-level tokenizer: pad 0, EOS 1, unk 2, then punctuation and words."""
    vocabulary = {"<pad>": 0, "</s>": 1, "<unk>": 2}
    for token in PUNCTUATION + sorted(set(WORDS)):
        vocabulary.setdefault(token, len(vocabulary))
    assert len(vocabulary) <= TINY_MODEL["vocab_size"]
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return tokenizer


@pytest.fixture(scope="session", autouse=True)
def bounded_torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def write_base(directory: Path, seed: int = 0) -> Path:
    """A tiny base model in the release layout."""
    torch.manual_seed(seed)
    model = base_model_pkg.build_model(base_model_pkg.ModelConfig(**TINY_MODEL), verify=False)
    base_model_pkg.save_pretrained(model, directory)
    build_tokenizer().save(str(directory / "tokenizer.json"))
    return directory


CITIES = "paris london rome berlin madrid oslo cairo delhi tokyo lima".split()
THINGS = "river mountain lake forest desert island bridge ocean star moon".split()
COLOURS = "red blue green yellow black white".split()
GREEK = "alpha beta gamma delta omega sigma kappa lambda theta zeta".split()
ANIMALS = "cat dog bird fish horse lion tiger bear wolf fox".split()
NUMBERS = "one two three four five six seven eight nine ten".split()


def make_records(source: str, count: int, *, negatives: int = 0, positives: int = 1) -> list[dict]:
    """Records whose query shares rare words with its positive, so a tiny model can learn them."""
    rows = []
    for index in range(count):
        city = CITIES[index % 10]
        thing = THINGS[(index // 10) % 10]
        greek = GREEK[(index * 7 + index // 100) % 10]
        animal = ANIMALS[(index * 3) % 10]
        number = NUMBERS[(index // 3) % 10]
        query = f"what is the {thing} of {city} {greek} {number}?"
        own = [
            {
                "title": f"{city} {thing}" if k == 0 and index % 2 == 0 else "",
                "text": f"the {COLOURS[(index + k) % 6]} {thing} in {city} is named {greek} {number} "
                        f"and the {animal} lives on the bank. note {k}." + (" it flows north." * (index % 4)),
            }
            for k in range(positives)
        ]
        hard = [
            {
                "title": "",
                "text": f"the {thing} in {CITIES[(index + 1 + k) % 10]} is called {GREEK[(index + 3 + k) % 10]} "
                        f"{number}, known for the {animal}. record {index} {k}.",
            }
            for k in range(negatives)
        ]
        rows.append({"id": f"{source}-{index}", "query": query, "positives": own, "negatives": hard,
                     "meta": {}, "ref": ""})  # fmt: skip
    return rows


def write_source(data_root: Path, name: str, rows: list[dict]) -> None:
    directory = data_root / name
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "train.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def config_document(tmp_path: Path, **section_overrides) -> dict:
    """A complete, small config; `section_overrides` patch sections key by key."""
    document = yaml.safe_load(REAL_CONFIG.read_text(encoding="utf-8"))
    document.update(
        base_checkpoint=str(tmp_path / "base"),
        data_root=str(tmp_path / "data"),
        cache_root=str(tmp_path / "cache"),
        eval_root=str(tmp_path / "eval"),
    )
    document["adapter"].update(rank=4, alpha=8.0, dropout=0.0)
    document["text"].update(max_query_tokens=16, max_passage_tokens=48, eval_max_tokens=64)
    document["loss"].update(temperature=0.05, matryoshka_dims=[64, 32, 16])
    document["dev"].update(
        queries_per_source=6, queries_overrides={"squad": 4}, distractors_per_source=20,
        benchmark_tasks=[], benchmark_every=0,
    )  # fmt: skip
    small = dict(batch_queries=8, warmup_steps=2, eval_every=4, save_every=2, keep_last=2)
    document["stage_1"].update(small, lr=5e-3, epochs_webfaq=1, epochs_curated=2)
    document["stage_2"].update(small, lr=2e-3, hard_negatives=2, webfaq_sample=40, epochs_webfaq=1, epochs_curated=1)
    document["mining"].update(top_k=12, skip_top=2, max_score_ratio=0.99, per_query=2)
    document["late_chunking"].update(window_tokens=64, overlap_tokens=8, chunk_tokens=12)
    document["resources"].update(chunk_tokens=400, encode_tokens=800, cpu_threads=2, cpu_chunk_tokens=400, seed=7)
    for section, values in section_overrides.items():
        document[section].update(values)
    return document


def make_config(tmp_path: Path, **section_overrides):
    return build_config(config_document(tmp_path, **section_overrides))


def cpu_plan(config) -> DevicePlan:
    """The CPU plan of `config` without changing the process's thread count."""
    resources = config.resources
    return DevicePlan(
        device=torch.device("cpu"), bf16=False, threads=2, memory_gb=0.0,
        chunk_tokens=resources.cpu_chunk_tokens, encode_tokens=resources.cpu_chunk_tokens,
    )  # fmt: skip


@pytest.fixture(scope="session")
def workspace(tmp_path_factory) -> Path:
    """A tiny base, three sources and their token cache, built once for the session (read-only)."""
    root = tmp_path_factory.mktemp("workspace")
    write_base(root / "base")
    write_source(root / "data", "webfaq-retrieval-english", make_records("webfaq", 70))
    write_source(root / "data", "squad", make_records("squad", 44, positives=2))
    write_source(root / "data", "hotpotqa", make_records("hotpot", 38, positives=2, negatives=3))
    (root / "data" / "squad_v2").mkdir()
    (root / "data" / "squad_v2" / "train.jsonl").write_text("", encoding="utf-8")
    prepare_cache(make_config(root), workers=1, log=lambda _: None)
    return root


@pytest.fixture
def config(workspace):
    return make_config(workspace)
