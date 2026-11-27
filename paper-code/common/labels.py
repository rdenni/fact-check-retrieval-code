"""
labels.py -- the one place that reads the annotators' label files.

Rule: for every (query, document) an annotator's LAST answer of kind 'pool'
counts (undo and re-labelling append a new line; the tool never deletes).
'gold' and 'repeat' answers are kept apart for the agreement report.
"""
from __future__ import annotations

from pathlib import Path

from .jsonl import read_jsonl


def load_annotator(path: str | Path) -> dict:
    """Returns {"pool": {(qid, doc): label}, "gold": [(label, gold_label)], "repeat": [(qid, doc, label)], "seconds": [float]}."""
    out = {"pool": {}, "gold": [], "repeat": [], "seconds": []}
    path = Path(path)
    if not path.exists():
        return out
    for r in read_jsonl(path):
        lab = r.get("label")
        if lab not in ("y", "n"):
            continue
        if r.get("seconds") is not None:  # a label read from a sheet has no time (pool/judges_sheet.py)
            out["seconds"].append(float(r["seconds"]))
        kind = r.get("kind", "pool")
        key = (r["query_id"], r["doc_id"])
        if kind == "gold":
            out["gold"].append((lab, r.get("gold_label")))
        elif kind == "repeat":
            out["repeat"].append((key[0], key[1], lab))
        else:
            out["pool"][key] = lab
    return out


def final_check_labels(work: str | Path, annotators: list[str]) -> tuple[dict, list]:
    """The human label of every document of the check sample (pool/build_pool.py): the label of the query's owner, the
    judge who reads the whole query, also for a shared document. The other judge's label of a shared document only
    measures the agreement; there is no discussion. This is TREC's primary assessor: "The author of a topic is its
    primary assessor", and the secondary assessors' labels serve only to study agreement (Voorhees 2000).
    Returns (labels, shared documents that only the other judge labelled: they have no final label)."""
    from .jsonl import read_json
    work = Path(work)
    check = read_json(work / "pool" / "check.json")
    shared = {tuple(x) for x in read_json(work / "pool" / "shared.json")}
    human = {j: load_annotator(work / "labels" / f"labels_{j}.jsonl")["pool"] for j in annotators}
    final, undecided = {}, []
    for q in check["queries"]:
        pq = check["per_query"][q]
        for d in pq["docs"]:
            k = (q, d)
            if k in human.get(pq["owner"], {}):
                final[k] = human[pq["owner"]][k]
            elif k in shared and any(k in human[j] for j in annotators):
                undecided.append(k)
    return final, undecided
