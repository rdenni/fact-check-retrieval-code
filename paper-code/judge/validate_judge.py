"""
validate_judge.py -- how good each AI judge is, measured on the human check sample, and the rule that picks the main
judge (model_selection/RULES.md, section 4; written before any label).

THE CANDIDATES: every file WORK/judge/labels_llm_<name>.jsonl (one per model) and, with three models, their majority
(a pair is relevant if at least two models say YES), written by this script as labels_llm_majority.jsonl.

THE PASS RULE. A candidate passes if both hold:
  (a) its Cohen's kappa with EACH human judge (on all the documents that judge labelled, the labels given
      independently) reaches the lower end of the 95 percent bootstrap interval of the two judges' kappa with each
      other on the shared documents (the labels given independently). The interval: the
      shared documents are drawn again with replacement --bootstrap times (seed --seed), the kappa is computed each
      time, and the 2.5th and 97.5th percentiles are the ends. Alaofi et al. 2026, p. 9: an AI judge is good when it
      is 'within the human-to-human agreement range';
  (b) on the check queries the systems come out in the same order under the human labels and under its labels:
      no pair of systems in the opposite order. The measure of a system on a check query is the share of its checked
      documents that are relevant: its top --depth for BM25 and the pipeline, the random sample of the strict set for
      a writer's graph run; the system's value is the mean over the check queries where it returned something. The
      human label of a document is the label of the query's owner, also for a shared document (no discussion).
THE MAIN JUDGE: the passing candidate with the highest mean kappa with the two judges; the others are reported as a
robustness table. If no candidate passes, the one with the highest mean kappa is used and the paper says it did not
pass (no time for more human judging before the deadline).
Also reported, per candidate: kappa and observed agreement split by the system that returned the document (an AI
judge may favour one kind of system: Clarke and Dietz 2024; Balog et al. 2025), and its YES rate next to the humans'.

Outputs: WORK/judge/validation.json, validation.md, labels_llm_majority.jsonl (with three models).

Usage: python judge/validate_judge.py --work WORK --annotators R,M [--majority gptoss120b,llama33_70b,deepseekv32]
"""
from __future__ import annotations

import argparse
import itertools
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.jsonl import read_jsonl, read_json, write_json, write_jsonl  # noqa: E402
from common.labels import load_annotator, final_check_labels  # noqa: E402
from judge.agreement import cohen_kappa  # noqa: E402


def bootstrap_interval(pairs: list[tuple[str, str]], reps: int, seed: str) -> tuple[float | None, float | None]:
    if not pairs:
        return None, None
    rng = random.Random(seed)
    ks = sorted(cohen_kappa(rng.choices(pairs, k=len(pairs)))["kappa"] for _ in range(reps))
    return round(ks[int(0.025 * reps)], 4), round(ks[min(reps - 1, int(0.975 * reps))], 4)


def kendall_tau_b(x: list[float], y: list[float]) -> float | None:
    conc = disc = tx = ty = 0
    for i, j in itertools.combinations(range(len(x)), 2):
        dx, dy = x[i] - x[j], y[i] - y[j]
        if dx == 0 and dy == 0:
            continue
        if dx == 0:
            tx += 1
        elif dy == 0:
            ty += 1
        elif dx * dy > 0:
            conc += 1
        else:
            disc += 1
    den = ((conc + disc + tx) * (conc + disc + ty)) ** 0.5
    return round((conc - disc) / den, 4) if den else None


def system_means(labels: dict, docs_of: dict[str, dict[str, list[str]]], keys: set) -> dict[str, float | None]:
    """per system: the mean over check queries of the share of its checked documents labelled 'y' (only documents in keys)."""
    out = {}
    for s, per_q in docs_of.items():
        vals = []
        for q, docs in per_q.items():
            ds = [d for d in docs if (q, d) in keys]
            if ds:
                vals.append(sum(1 for d in ds if labels[(q, d)] == "y") / len(ds))
        out[s] = round(statistics.mean(vals), 4) if vals else None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--annotators", default="R,M", help="the two human judges")
    ap.add_argument("--majority", default=None, help="the three models whose majority is a candidate (default: the three label files, if there are exactly three)")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", default="20261116")
    args = ap.parse_args()
    work = Path(args.work)
    a, b = [x.strip() for x in args.annotators.split(",") if x.strip()]
    check = read_json(work / "pool" / "check.json")
    shared = [tuple(x) for x in read_json(work / "pool" / "shared.json")]
    pool = {(r["query_id"], r["doc_id"]): r for r in read_jsonl(work / "pool" / "pool.jsonl")}
    human = {j: load_annotator(work / "labels" / f"labels_{j}.jsonl")["pool"] for j in (a, b)}
    # the final human label of every checked document (common/labels.py)
    final, undecided = final_check_labels(work, [a, b])

    # the human-human agreement and its interval
    hh_pairs = [(human[a][k], human[b][k]) for k in shared if k in human[a] and k in human[b]]
    hh = cohen_kappa(hh_pairs)
    low, high = bootstrap_interval(hh_pairs, args.bootstrap, f"{args.seed}:bootstrap")

    # the candidates
    files = {p.stem.replace("labels_llm_", ""): p for p in sorted((work / "judge").glob("labels_llm_*.jsonl")) if p.stem != "labels_llm_majority"}
    llm = {n: {(r["query_id"], r["doc_id"]): r["label"] for r in read_jsonl(p) if r.get("label") in ("y", "n")} for n, p in files.items()}
    models = {n: next((r.get("model") for r in read_jsonl(p)), None) for n, p in files.items()}
    trio = [x.strip() for x in args.majority.split(",")] if args.majority else (sorted(llm) if len(llm) == 3 else [])
    if trio:
        if len(trio) != 3 or any(n not in llm for n in trio):
            raise SystemExit(f"--majority needs three of the label files {sorted(llm)}")
        common = set.intersection(*(set(llm[n]) for n in trio))
        maj = {k: Counter(llm[n][k] for n in trio).most_common(1)[0][0] for k in common}
        write_jsonl(work / "judge" / "labels_llm_majority.jsonl",
                    [{"query_id": q, "doc_id": d, "label": lab, "model": "majority of " + ", ".join(trio)} for (q, d), lab in sorted(maj.items())])
        llm["majority"], models["majority"] = maj, "majority of " + ", ".join(trio)

    # the documents each system put into the check
    depth = check["depth"]
    docs_of: dict[str, dict[str, list[str]]] = {s: {} for s in check["ranked"] + check["sets"]}
    for q in check["queries"]:
        pq = check["per_query"][q]
        for s in check["ranked"]:
            docs_of[s][q] = [d for d in pq["docs"] if pool.get((q, d), {}).get("ranks", {}).get(s, 10**9) <= depth]
        for s in check["sets"]:
            docs_of[s][q] = pq["set_samples"].get(s, [])
    systems = list(docs_of)

    report = {"human_human": {**hh, "bootstrap_95": [low, high], "bootstrap_reps": args.bootstrap, "seed": args.seed},
              "checked_documents_with_final_label": len(final), "shared_without_owner_label": len(undecided),
              "human_yes_rate": {j: round(sum(1 for v in human[j].values() if v == "y") / len(human[j]), 4) if human[j] else None for j in (a, b)},
              "candidates": {}}
    for n, lab in llm.items():
        e = {"model": models.get(n), "n_labels": len(lab)}
        e["kappa_vs"] = {j: cohen_kappa([(lab[k], v) for k, v in human[j].items() if k in lab]) for j in (a, b)}
        ks = [e["kappa_vs"][j]["kappa"] for j in (a, b)]
        e["mean_kappa"] = round(statistics.mean(ks), 4) if None not in ks else None
        e["pass_a"] = bool(low is not None and None not in ks and all(k >= low for k in ks))
        keys = {k for k in final if k in lab}
        hum, mach = system_means(final, docs_of, keys), system_means(lab, docs_of, keys)
        both = [s for s in systems if hum[s] is not None and mach[s] is not None]
        opposite = [f"{s} / {t}" for s, t in itertools.combinations(both, 2)
                    if (hum[s] - hum[t]) * (mach[s] - mach[t]) < 0]
        e["order"] = {"human": hum, "ai": mach, "opposite_pairs": opposite,
                      "kendall_tau_b": kendall_tau_b([hum[s] for s in both], [mach[s] for s in both])}
        e["pass_b"] = not opposite and len(both) >= 2
        e["passes"] = e["pass_a"] and e["pass_b"]
        e["yes_rate"] = round(sum(1 for k in keys if lab[k] == "y") / len(keys), 4) if keys else None
        e["by_system"] = {}
        for s in systems:
            ks_s = {(q, d) for q, docs in docs_of[s].items() for d in docs if (q, d) in keys}
            e["by_system"][s] = cohen_kappa([(lab[k], final[k]) for k in sorted(ks_s)])
        report["candidates"][n] = e
    passing = {n: e for n, e in report["candidates"].items() if e["passes"]}
    pool_c = passing or {n: e for n, e in report["candidates"].items() if e["mean_kappa"] is not None}
    report["main_judge"] = max(pool_c, key=lambda n: pool_c[n]["mean_kappa"]) if pool_c else None
    report["main_judge_passed"] = bool(passing)
    write_json(work / "judge" / "validation.json", report)

    lines = ["# AI judge validation (model_selection/RULES.md, section 4)", "",
             f"human judges {a} and {b}: kappa {hh['kappa']} on {hh['n']} shared documents (labels given independently), "
             f"95 percent bootstrap interval {low} to {high} ({args.bootstrap} draws, seed {args.seed}); the pass mark is {low}",
             f"checked documents with a final human label (the owner's): {len(final)}; shared documents labelled only by the second judge: {len(undecided)}",
             f"YES rate: {a} {report['human_yes_rate'][a]}, {b} {report['human_yes_rate'][b]}", "",
             f"| candidate | model | kappa vs {a} | kappa vs {b} | mean | (a) | systems in the same order | tau-b | (b) | passes |",
             "|---|---|---:|---:|---:|---|---|---:|---|---|"]
    for n, e in report["candidates"].items():
        lines.append(f"| {n} | {e['model']} | {e['kappa_vs'][a]['kappa']} | {e['kappa_vs'][b]['kappa']} | {e['mean_kappa']} | "
                     f"{'yes' if e['pass_a'] else 'no'} | {'yes' if not e['order']['opposite_pairs'] else 'no: ' + '; '.join(e['order']['opposite_pairs'])} | "
                     f"{e['order']['kendall_tau_b']} | {'yes' if e['pass_b'] else 'no'} | {'YES' if e['passes'] else 'no'} |")
    lines += ["", f"main judge: {report['main_judge']}" + ("" if report["main_judge_passed"] else " (NO CANDIDATE PASSED: the paper says so)"), "",
              "## The systems on the check queries: share of checked documents relevant (human labels, then each candidate)", "",
              "| system | human | " + " | ".join(report["candidates"]) + " |", "|---|---:|" + "---:|" * len(report["candidates"])]
    first = next(iter(report["candidates"].values()), None)
    for s in systems:
        lines.append(f"| {s} | {first['order']['human'][s] if first else '-'} | " + " | ".join(str(e["order"]["ai"][s]) for e in report["candidates"].values()) + " |")
    lines += ["", "## Agreement with the final human labels, by the system that returned the document (kappa, n)", "",
              "| candidate | " + " | ".join(systems) + " |", "|---|" + "---:|" * len(systems)]
    for n, e in report["candidates"].items():
        lines.append(f"| {n} | " + " | ".join(f"{e['by_system'][s]['kappa']} ({e['by_system'][s]['n']})" for s in systems) + " |")
    (work / "judge" / "validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
