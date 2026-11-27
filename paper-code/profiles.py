"""
profiles.py -- the hardware setting of the text pipeline, and a test profile.

'laptop': the setting of the paper's runs. An NVIDIA RTX 4070 Laptop GPU with 8 GB of memory, under Windows: the
          Qwen3-Embedding-0.6B and Qwen3-Reranker-0.6B pair in bfloat16.
'cpu-test': a tiny random Qwen3 model on the CPU, so that the code path can be tested anywhere (tests/run_toy.py);
          its results mean nothing.

Model choice: the rule in model_selection/RULES.md, applied by model_selection/pareto_simple.py to the saved MTEB(eng,
v2) leaderboard file; for an 8 GB card it picks Qwen3-Embedding-0.6B, and the reranker is the Qwen3-Reranker of the
same size, released with it. The revisions are the Hugging Face commit hashes of the models used (see MODELS.md).

Memory: 0.6B parameters take about 1.2 GB in bfloat16 (2 bytes per parameter), which leaves room on an 8 GB card for a
batch of 64 texts. The embedder and the reranker are never loaded at the same time (run_all.py runs 'embed', then
'rerank'). On Windows there are no flash-attn wheels and no bitsandbytes, so the attention implementation is 'sdpa'
(built into PyTorch) and nothing is quantised.
"""
from __future__ import annotations

REVISIONS = {
    "Qwen/Qwen3-Embedding-0.6B": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
    "Qwen/Qwen3-Reranker-0.6B": "e61197ed45024b0ed8a2d74b80b4d909f1255473",
}

# One-sentence task instruction for the query side (the Qwen3 model cards say that leaving it out costs 1 to 5
# percent; documents carry no instruction). It is the instruction the paper's runs used.
QUERY_INSTRUCTION = ("Given a description of a disinformation narrative, retrieve "
                     "fact-checked claims that are instances or variants of it")

PROFILES = {
    "laptop": {
        "description": "the paper's runs: RTX 4070 Laptop GPU, 8 GB, Windows; the 0.6B pair in bfloat16",
        "embedder": "Qwen/Qwen3-Embedding-0.6B",
        "reranker": "Qwen/Qwen3-Reranker-0.6B",
        "device": "cuda", "dtype": "bfloat16", "attn": "sdpa",
        "embed_batch": 64, "rerank_batch": 32, "max_length": 512,
    },
    "cpu-test": {
        "description": "code test only: a tiny random Qwen3 model on the CPU; results are meaningless",
        "embedder": "yujiepan/qwen3-tiny-random",
        "reranker": "yujiepan/qwen3-tiny-random",
        "device": "cpu", "dtype": "float32", "attn": "sdpa",
        "embed_batch": 8, "rerank_batch": 8, "max_length": 128,
    },
}


def get_profile(name: str, **overrides) -> dict:
    if name not in PROFILES:
        raise KeyError(f"unknown profile {name!r}; choose from {sorted(PROFILES)}")
    p = dict(PROFILES[name])
    p["name"] = name
    for k, v in overrides.items():
        if v is not None:
            p[k] = v
    p["embedder_revision"] = p.get("embedder_revision") or REVISIONS.get(p["embedder"])
    p["reranker_revision"] = p.get("reranker_revision") or REVISIONS.get(p["reranker"])
    p["query_instruction"] = QUERY_INSTRUCTION
    return p
