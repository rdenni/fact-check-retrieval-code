"""
dense.py -- the dense retriever of the 2026 pipeline: a Qwen3-Embedding model
through Hugging Face Transformers, and exact nearest-neighbour search.

How the embedding is computed (copied from the 'Transformers Usage' part of the
Qwen3-Embedding model cards, the same for every size, read on 2026-09-18; the cards and the
models are under the Apache License 2.0, https://huggingface.co/Qwen/Qwen3-Embedding-0.6B):
  - tokenizer with padding_side='left';
  - the query is 'Instruct: {task}\\nQuery:{query}'; documents get no instruction;
  - the embedding is the hidden state of the LAST token (last_token_pool);
  - L2-normalised, so the dot product is the cosine similarity.
We use max_length 512: longer texts are cut there (the model allows 8192).

Search is exact: the query vector is multiplied with every document vector
(about 80,000 explanations x 1,024 dimensions for the 0.6B model, about 160 MB in float16, kept in RAM, processed in chunks on
the GPU or CPU). An approximate index, such as Qdrant's HNSW, is
approximate; exactness removes one source of noise from the comparison.

Files written by `embed`: WORK/dense_<profile>/doc_emb.npy (float16, one row
per document of corpus.jsonl, same order), doc_ids.json, query_emb.npy,
query_ids.json, meta.json (model, revision, dtype, instruction, timing).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _torch_dtype(name: str):
    import torch
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def load_pretrained(cls, model_id: str, revision: str | None, dtype: str, attn: str | None):
    """transformers 5 calls the precision argument 'dtype', 4.x called it
    'torch_dtype'; try the new name first."""
    kwargs = {}
    if attn:
        kwargs["attn_implementation"] = attn
    try:
        return cls.from_pretrained(model_id, revision=revision, dtype=_torch_dtype(dtype), **kwargs)
    except TypeError:
        return cls.from_pretrained(model_id, revision=revision, torch_dtype=_torch_dtype(dtype), **kwargs)


class Qwen3Embedder:
    def __init__(self, model_id: str, revision: str | None, device: str = "cuda",
                 dtype: str = "bfloat16", attn: str = "sdpa", max_length: int = 512):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.model_id, self.revision, self.device, self.dtype_name = model_id, revision, device, dtype
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, padding_side="left")
        self.model = load_pretrained(AutoModel, model_id, revision, dtype, attn).to(device).eval()
        self.torch = torch

    @staticmethod
    def last_token_pool(last_hidden_states, attention_mask):
        # Verbatim logic of the Qwen3-Embedding model card.
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        import torch
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    @staticmethod
    def with_instruction(task: str, query: str) -> str:
        return f"Instruct: {task}\nQuery:{query}"  # exact format of the model card (no space after 'Query:')

    def encode(self, texts: list[str], batch_size: int = 32, progress: bool = False) -> np.ndarray:
        """Returns an (n, dim) float32 array of L2-normalised embeddings."""
        torch = self.torch
        F = torch.nn.functional
        out = []
        it = range(0, len(texts), batch_size)
        if progress:
            from tqdm import tqdm
            it = tqdm(it, desc=f"embed[{self.model_id.split('/')[-1]}]", unit="batch")
        with torch.no_grad():
            for i in it:
                batch = texts[i:i + batch_size]
                enc = self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
                enc = {k: v.to(self.device) for k, v in enc.items()}
                hidden = self.model(**enc).last_hidden_state
                emb = self.last_token_pool(hidden, enc["attention_mask"])
                emb = F.normalize(emb.float(), p=2, dim=1)
                out.append(emb.cpu().numpy().astype(np.float32))
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.model.config.hidden_size), dtype=np.float32)


def embed_corpus_and_queries(profile: dict, corpus: list[dict], queries: list[dict], out_dir: Path,
                             batch_size: int | None = None, progress: bool = True) -> dict:
    """Encode everything once with the embedder loaded once; the reranker is
    loaded later by a separate process, so the two never share the GPU."""
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    enc = Qwen3Embedder(profile["embedder"], profile.get("embedder_revision"), profile["device"],
                        profile["dtype"], profile.get("attn"), profile["max_length"])
    t_load = time.time() - t0
    bs = batch_size or profile["embed_batch"]
    t0 = time.time()
    from common import kg as _kg
    mode = profile.get("text_mode", "claim")
    n_all = len(corpus)
    corpus = _kg.text_corpus(corpus, mode)  # explanation mode: documents without an explanation are not embedded (not retrievable)
    doc_emb = enc.encode([_kg.doc_text(d, mode) for d in corpus], bs, progress)
    t_docs = time.time() - t0
    t0 = time.time()
    q_texts = [enc.with_instruction(profile["query_instruction"], q["free_text"]) for q in queries]
    q_emb = enc.encode(q_texts, min(bs, 16), False)
    t_q = time.time() - t0
    np.save(out_dir / "doc_emb.npy", doc_emb.astype(np.float16))
    np.save(out_dir / "query_emb.npy", q_emb.astype(np.float32))
    (out_dir / "doc_ids.json").write_text(json.dumps([d["cr_iri"] for d in corpus]), encoding="utf-8")
    (out_dir / "query_ids.json").write_text(json.dumps([q["query_id"] for q in queries]), encoding="utf-8")
    meta = {"embedder": profile["embedder"], "embedder_revision": profile.get("embedder_revision"),
            "device": profile["device"], "dtype": profile["dtype"], "attn": profile.get("attn"),
            "max_length": profile["max_length"], "batch_size": bs, "query_instruction": profile["query_instruction"],
            "pooling": "last token", "normalised": True, "dim": int(doc_emb.shape[1]), "text_mode": profile.get("text_mode", "claim"),
            "n_docs": len(corpus), "n_left_out_no_text": n_all - len(corpus), "n_queries": len(queries),
            "seconds": {"load": round(t_load, 1), "docs": round(t_docs, 1), "queries": round(t_q, 1)},
            "docs_per_second": round(len(corpus) / t_docs, 1) if t_docs > 0 else None}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_embeddings(out_dir: Path):
    doc_emb = np.load(out_dir / "doc_emb.npy")
    q_emb = np.load(out_dir / "query_emb.npy")
    doc_ids = json.loads((out_dir / "doc_ids.json").read_text(encoding="utf-8"))
    q_ids = json.loads((out_dir / "query_ids.json").read_text(encoding="utf-8"))
    return doc_emb, doc_ids, q_emb, q_ids


def exact_search(q_emb: np.ndarray, doc_emb: np.ndarray, doc_ids: list[str], k: int,
                 chunk: int = 20000) -> list[list[tuple[str, float]]]:
    """Exact top-k by cosine (vectors are normalised). Chunked so a 1 GB matrix
    never has to be cast to float32 at once."""
    k = min(k, len(doc_ids))
    q = q_emb.astype(np.float32)
    best_scores = np.full((q.shape[0], k), -np.inf, dtype=np.float32)
    best_idx = np.zeros((q.shape[0], k), dtype=np.int64)
    for start in range(0, doc_emb.shape[0], chunk):
        d = doc_emb[start:start + chunk].astype(np.float32)
        s = q @ d.T
        cand_s = np.concatenate([best_scores, s], axis=1)
        cand_i = np.concatenate([best_idx, np.arange(start, start + d.shape[0])[None, :].repeat(q.shape[0], 0)], axis=1)
        part = np.argpartition(-cand_s, k - 1, axis=1)[:, :k]
        best_scores = np.take_along_axis(cand_s, part, axis=1)
        best_idx = np.take_along_axis(cand_i, part, axis=1)
    out = []
    for r in range(q.shape[0]):
        order = np.argsort(-best_scores[r], kind="stable")
        out.append([(doc_ids[int(best_idx[r, j])], float(best_scores[r, j])) for j in order if np.isfinite(best_scores[r, j])])
    return out
