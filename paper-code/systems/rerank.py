"""
rerank.py -- the cross-encoder of the 2026 pipeline: a Qwen3-Reranker model.

How the score is computed (copied from the 'Transformers Usage' part of the
Qwen3-Reranker model cards, the same for every size, read on 2026-09-18; the cards and the
models are under the Apache License 2.0, https://huggingface.co/Qwen/Qwen3-Reranker-0.6B):
  - the input is  prefix + '<Instruct>: {task}\\n<Query>: {query}\\n<Document>: {doc}' + suffix
    where prefix is the system message that says the answer can only be
    "yes" or "no", and suffix opens the assistant turn with an empty <think> block;
  - the model is a causal language model; we read the logits of the last
    position for the tokens 'yes' and 'no';
  - score = softmax over (no, yes) of those two logits, taken at 'yes':
    a number between 0 and 1, the model's probability that the document
    meets the query.
The instruction is the same sentence as the embedder's query instruction.
"""
from __future__ import annotations

import time
from pathlib import Path

import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
          "and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


from systems.dense import load_pretrained  # noqa: E402


class Qwen3Reranker:
    def __init__(self, model_id: str, revision: str | None, device: str = "cuda",
                 dtype: str = "bfloat16", attn: str = "sdpa", max_length: int = 512):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_id, self.revision, self.device = model_id, revision, device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, padding_side="left")
        self.model = load_pretrained(AutoModelForCausalLM, model_id, revision, dtype, attn).to(device).eval()
        self.torch = torch
        self._logits_kwarg = None  # decided on the first call
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.prefix_tokens = self.tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(SUFFIX, add_special_tokens=False)

    @staticmethod
    def format_pair(instruction: str, query: str, doc: str) -> str:
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

    def _last_logits(self, enc):
        """Logits of the last position only. Newer transformers accept
        logits_to_keep=1 (older: num_logits_to_keep), which avoids building the
        logits of every position; fall back to the full tensor otherwise."""
        if self._logits_kwarg is None:
            for name in ("logits_to_keep", "num_logits_to_keep"):
                try:
                    out = self.model(**enc, **{name: 1}).logits
                    self._logits_kwarg = name
                    return out[:, -1, :]
                except TypeError:
                    continue
            self._logits_kwarg = ""
        if self._logits_kwarg:
            return self.model(**enc, **{self._logits_kwarg: 1}).logits[:, -1, :]
        return self.model(**enc).logits[:, -1, :]

    def score(self, instruction: str, query: str, docs: list[str], batch_size: int = 16,
              progress: bool = False) -> np.ndarray:
        """P(yes) for every (query, doc) pair, as a float32 array."""
        torch = self.torch
        pairs = [self.format_pair(instruction, query, d) for d in docs]
        out = []
        it = range(0, len(pairs), batch_size)
        if progress:
            from tqdm import tqdm
            it = tqdm(it, desc="rerank", unit="batch", leave=False)
        body_max = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        with torch.no_grad():
            for i in it:
                batch = pairs[i:i + batch_size]
                enc = self.tokenizer(batch, padding=False, truncation="longest_first",
                                     return_attention_mask=False, max_length=body_max)
                for j, ids in enumerate(enc["input_ids"]):
                    enc["input_ids"][j] = self.prefix_tokens + ids + self.suffix_tokens
                enc = self.tokenizer.pad(enc, padding=True, return_tensors="pt", max_length=self.max_length)
                enc = {k: v.to(self.device) for k, v in enc.items()}
                logits = self._last_logits(enc)
                two = torch.stack([logits[:, self.token_false_id], logits[:, self.token_true_id]], dim=1)
                p_yes = torch.nn.functional.log_softmax(two.float(), dim=1)[:, 1].exp()
                out.append(p_yes.cpu().numpy().astype(np.float32))
        return np.concatenate(out) if out else np.zeros((0,), dtype=np.float32)


def rerank_candidates(profile: dict, queries: list[dict], candidates: dict[str, list[str]],
                      texts: dict[str, str], batch_size: int | None = None, progress: bool = True):
    """candidates: query_id -> candidate doc ids. Returns (query_id -> [(doc_id, p_yes)] best first, meta)."""
    t0 = time.time()
    rr = Qwen3Reranker(profile["reranker"], profile.get("reranker_revision"), profile["device"],
                       profile["dtype"], profile.get("attn"), profile["max_length"])
    t_load = time.time() - t0
    bs = batch_size or profile["rerank_batch"]
    t0 = time.time()
    n_pairs = 0
    runs = {}
    it = queries
    if progress:
        from tqdm import tqdm
        it = tqdm(queries, desc=f"rerank[{profile['reranker'].split('/')[-1]}]", unit="q")
    for q in it:
        ids = candidates.get(q["query_id"], [])
        if not ids:
            runs[q["query_id"]] = []
            continue
        scores = rr.score(profile["query_instruction"], q["free_text"], [texts.get(d, "") for d in ids], bs)
        n_pairs += len(ids)
        order = np.argsort(-scores, kind="stable")
        runs[q["query_id"]] = [(ids[int(j)], float(scores[j])) for j in order]
    t_score = time.time() - t0
    meta = {"reranker": profile["reranker"], "reranker_revision": profile.get("reranker_revision"),
            "device": profile["device"], "dtype": profile["dtype"], "attn": profile.get("attn"),
            "max_length": profile["max_length"], "batch_size": bs, "instruction": profile["query_instruction"],
            "score": "P(yes) from the yes/no logits at the last position (model card recipe)", "text_mode": profile.get("text_mode", "claim"),
            "n_pairs": n_pairs, "seconds": {"load": round(t_load, 1), "score": round(t_score, 1)},
            "pairs_per_second": round(n_pairs / t_score, 1) if t_score > 0 else None}
    return runs, meta
