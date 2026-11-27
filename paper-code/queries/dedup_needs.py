"""
dedup_needs.py -- merge the candidates of one country (all its cells) that
describe the same claim, so that the random draw picks among distinct
information needs and a claim reported by several documents, or by one
document read by two cells (a Freedom on the Net edition covers two years),
is not drawn several times. A merged cluster belongs to the cell of the year
its passages name (their "period"), or to the earliest cell among its members
when they name none, and lists all of them ("cells").

Two methods. 'embed' (the real run): sentence embeddings from a small model
that is NOT part of the evaluated pipeline (default sentence-transformers
all-MiniLM-L6-v2 through transformers, mean pooling), cosine similarity of the
"need" sentences, union-find over pairs above --threshold. 'lexical' (tests,
or a check): Jaccard overlap of the word sets above --threshold. Candidates
whose passage was not found in the text (draft_needs.py's passage_found =
false) are set aside first, as rule D4 of REVIEW-RULES.md, and listed in
dropped_auto_<cell>.jsonl.

Output: WORK/drafting/merged_<cell>.jsonl, one line per cluster: cluster_id,
the representative candidate (the one with the longest passage), the member
ids, the documents that mention it (mentions), and the fields the review sheet
shows; WORK/drafting/dedup.md with the counts per cell.

Usage:
  python queries/dedup_needs.py --work WORK [--method embed|lexical] [--model MODEL] [--threshold 0.85] [--only cell,...]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.jsonl import read_jsonl, write_jsonl  # noqa: E402

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# the model on the Hugging Face hub changes over time (last changed 2026-06-01); the merge uses this exact commit
DEFAULT_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a or b else 0.0


class Embedder:
    def __init__(self, model_id: str, device: str = "cpu", revision: str | None = None):
        import torch  # noqa: F401
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModel.from_pretrained(model_id, revision=revision).to(device).eval()
        self.device = device

    def __call__(self, texts: list[str]):
        import numpy as np
        import torch
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), 64):
                batch = self.tok(texts[i:i + 64], padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
                h = self.model(**batch).last_hidden_state
                m = batch["attention_mask"].unsqueeze(-1).float()
                e = (h * m).sum(1) / m.sum(1).clamp(min=1)
                e = torch.nn.functional.normalize(e, dim=-1)
                out.append(e.cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 1))


def clusters_of(n: int, pairs: list[tuple[int, int]]) -> list[list[int]]:
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: g[0])


def home_cell(members: list[dict]) -> str:
    """The cell a cluster belongs to. Its members can sit in several cells of the country (a report on 2020-2025 is
    read by six cells). The cluster goes to the cell of the year its passages name ("period", the month and year the
    passage gives), the most named one if several, the earliest of them on a tie; to the earliest cell if no member
    names a year among its cells."""
    by_year = {int(m["cell"].rsplit("-", 1)[1]): m["cell"] for m in members}
    named = [int(y) for m in members for y in re.findall(r"\b(?:19|20)\d\d\b", str(m.get("period") or "")) if int(y) in by_year]
    year = max(sorted(set(named)), key=named.count) if named else min(by_year)
    return by_year[year]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--method", choices=["embed", "lexical"], default="embed")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--revision", default=DEFAULT_REVISION, help="the model's commit on the hub (pinned so the merge is repeatable)")
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    d = args.work / "drafting"
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    files = sorted(p for p in d.glob("candidates_*.jsonl") if not only or p.stem[len("candidates_"):] in only)
    embed = Embedder(args.model, revision=args.revision) if args.method == "embed" else None
    report = ["# Deduplication of the candidates", "",
              f"method: {args.method}" + (f" ({args.model}, revision {args.revision})" if embed else "") + f", threshold {args.threshold}; "
              "merged within each country across its cells (a document read by two cells gives the same claim twice); "
              "a cluster belongs to the cell of the year its passages name, else to the earliest cell among its members", "",
              "| cell | candidates | passage not found (D4) | clusters assigned to the cell | merged away |", "|---|---|---|---|---|"]
    by_country: dict[str, list[tuple[str, list[dict]]]] = {}
    for f in files:
        cell = f.stem[len("candidates_"):]
        cands = list(read_jsonl(f))
        country = cands[0]["country"] if cands else cell.rsplit("-", 1)[0]
        by_country.setdefault(country, []).append((cell, cands))
    for country, cells in sorted(by_country.items()):
        good, counts = [], {}
        for cell, cands in cells:
            bad = [c for c in cands if not c.get("passage_found")]
            write_jsonl(d / f"dropped_auto_{cell}.jsonl", [{**c, "rule": "D4", "reason": "passage not found verbatim in the text"} for c in bad])
            good.extend(c for c in cands if c.get("passage_found"))
            counts[cell] = [len(cands), len(bad), 0, 0]
        pairs: list[tuple[int, int]] = []
        if len(good) > 1:
            if embed:
                E = embed([c["need"] for c in good])
                S = E @ E.T
                pairs = [(i, j) for i in range(len(good)) for j in range(i + 1, len(good)) if S[i, j] >= args.threshold]
            else:
                W = [words(c["need"] + " " + c["claim"]) for c in good]
                pairs = [(i, j) for i in range(len(good)) for j in range(i + 1, len(good)) if jaccard(W[i], W[j]) >= args.threshold]
        merged_by_cell: dict[str, list[dict]] = {cell: [] for cell, _ in cells}
        for g in clusters_of(len(good), pairs):
            members = [good[i] for i in g]
            rep = max(members, key=lambda c: len(c["passage"]))
            home = home_cell(members)
            docs = sorted({m["doc"] for m in members})
            n = len(merged_by_cell[home])
            merged_by_cell[home].append({
                "cluster_id": f"{home}:c{n:04d}", "cell": home, "cells": sorted({m["cell"] for m in members}),
                "country": rep["country"], "year": int(home.rsplit("-", 1)[1]),
                "need": rep["need"], "claim": rep["claim"], "spreader": rep.get("spreader", ""),
                "verdict_words": rep.get("verdict_words", ""), "verdict_found": rep.get("verdict_found", False),
                "passage": rep["passage"], "entities": rep["entities"],
                "period": rep["period"], "topic": rep["topic"], "country_fit": rep["country_fit"],
                "representative": rep["candidate_id"], "members": [m["candidate_id"] for m in members],
                "mentions": len(docs), "docs": docs, "doc": rep["doc"], "tier": rep["tier"], "publisher_type": rep["publisher_type"],
                "url": rep["url"], "file": rep["file"], "sha256": rep["sha256"], "model": rep["model"], "precision": rep["precision"],
                "prompt_sha256": rep["prompt_sha256"]})
        n_good = {cell: sum(1 for c in good if c["cell"] == cell) for cell, _ in cells}
        for cell, _ in cells:
            write_jsonl(d / f"merged_{cell}.jsonl", merged_by_cell[cell])
            total, bad, _, _ = counts[cell]
            report.append(f"| {cell} | {total} | {bad} | {len(merged_by_cell[cell])} | {n_good[cell] - len(merged_by_cell[cell])} |")
            print(f"[dedup] {cell:22s} {total:5d} candidates, {bad:3d} without passage, {len(merged_by_cell[cell]):5d} clusters")
    (d / "dedup.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
