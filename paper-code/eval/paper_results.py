"""
paper_results.py -- every number, table and figure of the paper's Results section (and of the matching part of the
Supplementary Material), read from the files that the evaluation scripts wrote. Nothing is typed by hand: when a
label or a run changes, re-run the evaluation scripts and then this script, and the paper follows.

What it reads (WORK is the work folder of the paper's runs; every file is written by a script of this repository):
  WORK/eval/metrics_<name>.json          eval/metrics.py; <name> is ai (the main judge), ai_before (the documents
                                         published before entity linking stopped, or undated), human and ai_check
                                         (the check sample: human labels, and the main judge on the same documents),
                                         and ai_<judge> for each single model judge (robustness)
  WORK/judge/validation.json             judge/validate_judge.py: the judges' agreement, the pass rule, the main judge
  WORK/judge/labels_llm_<judge>.jsonl    each model's labels (agreement, providers, cost)
  WORK/labels/labels_<H>.jsonl           each human judge's labels; WORK/pool/shared.json and check.json
  WORK/pool/pool.jsonl, extension_before.json   the pool (pool/build_pool.py, pool/extend_pool_before.py)
  WORK/qrels/qrels_ai.txt, WORK/runs/<system>.trec   the pool-depth curve and the precision and recall behind the
                                         oracle's F1
  WORK/runs/<set>_sets.jsonl             the writers' result sets (the worked example)
  WORK/eval/failures_<w>.csv, oracle_<w>.jsonl      eval/failure_protocol.py and eval/oracle_formula.py, per writer
  WORK/eval/stale_names_<name>.json      eval/stale_names.py
  WORK/eval/writer_agreement_<name>.json eval/writer_agreement.py with --qrels
  the query file (--queries)             only for the text of the worked example
A file that is missing is reported and its numbers are printed as "--" (the toy test has no single-model tables).

What it writes (--out, default latex/figures):
  results_numbers.tex    one macro per number the text quotes (\\resBMRTwenty ...)
  results_*.tex          the tables (one tabular each; the caption is written in the section)
  results_*.pdf          the figures
  results.json           the numbers behind every table and figure, for checking

Usage:
  python eval/paper_results.py --work WORK --queries Q.json [--out latex/figures]
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import random
import re
import statistics
import sys
from collections import Counter
from itertools import combinations, groupby
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common.jsonl import read_jsonl, write_json  # noqa: E402
from common.labels import load_annotator  # noqa: E402
from common.trec import read_qrels, read_run  # noqa: E402
from judge.agreement import cohen_kappa  # noqa: E402
import metrics as mt  # noqa: E402  (the same Wilcoxon test and pool-depth curve as the tables)
from stats.plot_archive import AQUA, BLUE, GREY, ORANGE, YELLOW, save, style  # noqa: E402

csv.field_size_limit(10 ** 8)  # the failure files carry long entity lists

# How the tables name things. Only names, no numbers.
SYSTEM_NAME = {"bm25": "BM25", "pipeline": "standard IR pipeline", "bm25_claimart": "BM25, claim and explanation",
               "pipeline_claimart": "standard IR pipeline, claim and explanation"}
JUDGE_NAME = {"gptoss120b": "gpt-oss-120b", "llama33_70b": "Llama-3.3-70B", "deepseekv32": "DeepSeek-V3.2",
              "majority": "majority of the three"}
WRITERS = ["A", "B"]          # the writers in the order of --sets
HUMANS = ["H1", "H2"]         # the human judges in the order of --humans
TEST_MEASURE = {"set_P vs P@B": "precision, same size", "set_R vs R@B": "recall, same size", "set_F1 vs F1@B": "F1, same size"}

MACROS: dict[str, str] = {}
NOTES: list[str] = []


def mac(name: str, value) -> None:
    """One LaTeX macro. Its name must be letters only (LaTeX allows no digits in a command name)."""
    if not re.fullmatch(r"[A-Za-z]+", name):
        raise SystemExit(f"bad macro name {name!r}: letters only")
    if name in MACROS:
        raise SystemExit(f"macro {name} written twice")
    MACROS[name] = "--" if value is None else str(value)


def esc(s) -> str:
    """Text for LaTeX: escape the special characters."""
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    return s.replace("“", "``").replace("”", "''").replace('"', "''")


def formula_tex(formula: str) -> str:
    """A formula for the text: every page title between LaTeX quotes, the rest escaped."""
    return re.sub(r'"([^"]*)"', lambda m: "``" + esc(m.group(1)) + "''", formula)


def f3(x):
    return None if x is None else f"{x:.3f}"


def f2(x):
    return None if x is None else f"{x:.2f}"


def signed(x):
    """A difference with its sign, and a true minus sign in LaTeX."""
    return None if x is None else ("$+$" if x >= 0 else "$-$") + f"{abs(x):.3f}"


def pct(x):
    return None if x is None else f"{100 * x:.0f}"


def pct1(x):
    return None if x is None else f"{100 * x:.1f}"


def num(n):
    return None if n is None else f"{n:,}"


def cell(x) -> str:
    return "--" if x is None else str(x)


def p_table(p) -> str:
    if p is None:
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def p_text(p) -> str | None:
    """A p-value for a sentence, with its relation: $p<0.001$ or $p=0.021$."""
    if p is None:
        return None
    return "$p<0.001$" if p < 0.001 else f"$p={p:.3f}$"


def load(path: Path):
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    NOTES.append(f"missing: {path}")
    return None


def stack(*parts: str) -> str:
    """A header cell on several lines (plain LaTeX, no package)."""
    return "\\shortstack{" + "\\\\".join(parts) + "}"


ROW_RULE = "\\midrule[0.2pt]"  # a thin rule between two rows (booktabs), thinner than the rule under the header


def tabular(spec: str, header: list[str], rows: list, out: Path) -> None:
    """rows: a list of cell lists, or the string 'mid' for a \\midrule, or ('block', text) for a full-width line.
    A thin rule separates every row from the next, except before a 'mid' and after the last row."""
    lines = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    ncol = len(header)
    for k, r in enumerate(rows):
        if r == "mid":
            lines.append("\\midrule")
        elif isinstance(r, tuple) and r[0] == "block":
            lines.append(f"\\multicolumn{{{ncol}}}{{l}}{{\\emph{{{r[1]}}}}} \\\\")
        else:
            lines.append(" & ".join(cell(c) for c in r) + " \\\\")
        if r != "mid" and k + 1 < len(rows) and rows[k + 1] != "mid":
            lines.append(ROW_RULE)
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    out.write_text("\n".join(lines), encoding="utf-8")


BOOT_REPS = 10000  # draws of the bootstrap intervals of the robustness figure (\resBootReps)


def boot_ci(vals: list, seed: str, reps: int = BOOT_REPS) -> tuple:
    """The 95 percent percentile bootstrap interval of a mean over queries: the queries drawn again with replacement
    reps times, the 2.5th and 97.5th percentiles of the means. A string seed makes it the same at every run."""
    if not vals:
        return (None, None)
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(vals, k=len(vals))) for _ in range(reps))
    return (means[int(0.025 * reps)], means[int(0.975 * reps) - 1])


def paired_exact(xa: list, xb: list) -> dict:
    """metrics.paired (the same Wilcoxon test, wins and losses), but with the mean difference NOT rounded to four
    decimals first: rounding twice (to 4, then to 3 for the text) can move the last digit (-0.0115 -> -0.011 instead of
    -0.012, the value oracle_formula.py prints)."""
    t = mt.paired(xa, xb)
    diffs = [x - y for x, y in zip(xa, xb) if x is not None and y is not None]
    t["mean_diff"] = statistics.mean(diffs) if diffs else None
    return t


def sname(s: str, sets: list[str], generous: list[str]) -> str:
    if s in sets:
        return f"graph, formula writer {WRITERS[sets.index(s)]}"
    if s in generous:
        return f"generous version, formula writer {WRITERS[generous.index(s)]}"
    return SYSTEM_NAME.get(s, esc(s))


# ---------------------------------------------------------------- the main tables (also before the linking stop)

def ranked_table(m: dict, ranked: list[str], extra: list[str], generous: list[str], sets: list[str], out: Path) -> None:
    S = m["summary"]
    rows = []
    for k_, (block, tags) in enumerate((("", ranked), ("the text systems on the claim and the explanation (second row, no test)", extra),
                                        ("", generous))):
        tags = [t for t in tags if t in S]
        if not tags:
            continue
        if k_:  # a rule between the blocks; the generous version needs no title (the caption says it has no test)
            rows.append("mid")
            if block:
                rows.append(("block", block))
        for t in tags:
            rows.append([sname(t, sets, generous)] + [f3(S[t].get(k)) for k in ("P@10", "P@20", "nDCG@20", "R@20")])
    tabular("lrrrr", ["system", "P@10", "P@20", "nDCG@20", "R@20"], rows, out)


def sets_table(m: dict, sets: list[str], out: Path) -> None:
    S = m["summary"]
    rows = []
    for i, w in enumerate(sets):
        if w not in S:
            continue
        s = S[w]
        nonempty = s["queries"] - s["queries_B_zero"]
        rows.append([f"formula writer {WRITERS[i]}", s["queries"], s["queries_B_zero"], f"{s['median_B']:g}",
                     f"{f3(s['set_P'])} ({nonempty})", f3(s["set_R"]), f3(s["set_F1"])])
    rows.append("mid")
    for label, key in (("mean of the two formula writers", "graph (mean of the writers)"), ("better formula writer per query (a ceiling)", "graph (better writer per query: a ceiling)")):
        if key in S:
            rows.append([label, "", "", "", f3(S[key]["set_P"]), f3(S[key]["set_R"]), f3(S[key]["set_F1"])])
    tabular("lrrrrrr", ["result set", "queries", "empty", stack("median", "size"), stack("precision", "(queries)"), "recall", "F1"], rows, out)


def bold(x: str) -> str:
    return f"\\textbf{{{x}}}"


def best_idx(values: list) -> set:
    """The positions of the best (highest) value of a column; equal values (at three decimals) are all best."""
    vals = [None if v is None else round(v, 3) for v in values]
    top = max((v for v in vals if v is not None), default=None)
    return {i for i, v in enumerate(vals) if v is not None and v == top}


def main_results_table(m: dict, ranked: list[str], generous: list[str], sets: list[str], out: Path) -> None:
    """The main table: what each system returns. A ranked system is scored on its first 20 results, a formula writer on its whole set.
    Recall is R@20 or the recall of the set (0 for an empty set); the second column counts the queries for which at least one relevant
    fact-check is returned. The best value of each column among the four systems is in bold; the mean of the two writers and the better
    writer per query (which needs the labels, so it is not realistic) are below a rule and take no part."""
    S, pq = m["summary"], m["per_query"]
    measured = sorted(q for q, n in m["n_relevant_per_query"].items() if n > 0)
    n = len(measured)
    systems = []  # (label, recall, queries with a relevant fact-check returned)
    for r in ranked:
        if r in S:
            found = sum(1 for q in measured if q in pq.get(r, {}) and pq[r][q]["R@20"] > 0)
            systems.append((f"{SYSTEM_NAME.get(r, esc(r))}, first 20 results", S[r].get("R@20"), found))
    for i, w in enumerate(sets):
        if w in S:
            found = sum(1 for q in measured if q in pq.get(w, {}) and pq[w][q]["set_R"] > 0)
            systems.append((f"formula writer {WRITERS[i]}, whole set", S[w]["set_R"], found))
    for i, g in enumerate(generous):  # the generous version is a ranked list too: its first 20 results
        if g in S:
            found = sum(1 for q in measured if q in pq.get(g, {}) and pq[g][q]["R@20"] > 0)
            systems.append((f"formula writer {WRITERS[i]}, generous version, first 20 results", S[g].get("R@20"), found))
    best_r, best_f = best_idx([x[1] for x in systems]), best_idx([x[2] for x in systems])
    rows = []
    for k, (lab, rec, found) in enumerate(systems):
        c_rec, c_found = f3(rec), f"{found} of {n}"
        rows.append([lab, bold(c_rec) if k in best_r else c_rec, bold(c_found) if k in best_f else c_found])
    rows.append("mid")
    key, key_best = "graph (mean of the writers)", "graph (better writer per query: a ceiling)"
    if key in S:
        gq = m["graph_mean_per_query"]
        found = sum(1 for q in measured if (gq.get(q, {}).get("set_R") or 0) > 0)
        rows.append(["the two formula writers, whole sets, mean", f3(S[key]["set_R"]), f"{found} of {n}"])
    if key_best in S:
        found = 0
        for q in measured:
            per = [pq[w][q] for w in sets if q in pq.get(w, {})]
            if per and max(per, key=lambda r_: (r_["set_F1"] or 0.0))["set_R"] > 0:
                found += 1
        rows.append(["the better of the two for each query, whole sets (not realistic)", f3(S[key_best]["set_R"]), f"{found} of {n}"])
    tabular("lrr", ["system", "recall", stack("queries with at least one", "relevant fact-check found")], rows, out)


def at_b_table(m: dict, ranked: list[str], sets: list[str], out: Path) -> None:
    """The same systems with the same number of results: BM25 and the pipeline return as many results as the writer's set holds
    (at most 100; none when the set is empty). The best value of each column within a writer's block is in bold."""
    S = m["summary"]
    rows = []
    for i, w in enumerate(sets):
        if w not in S:
            continue
        if rows:
            rows.append("mid")
        block = [(f"formula writer {WRITERS[i]}'s set", S[w]["set_P"], S[w]["set_R"], S[w]["set_F1"])]
        for r in ranked:
            block.append((f"{SYSTEM_NAME.get(r, esc(r))}, as many results as {WRITERS[i]}'s set",
                          S[r].get(f"P@B[{w}]"), S[r].get(f"R@B[{w}]"), S[r].get(f"F1@B[{w}]")))
        best = [best_idx([b_[c] for b_ in block]) for c in (1, 2, 3)]
        for j, b_ in enumerate(block):
            rows.append([b_[0]] + [bold(f3(b_[c])) if j in best[c - 1] else f3(b_[c]) for c in (1, 2, 3)])
    tabular("lrrr", ["", "precision", "recall", "F1"], rows, out)


def test_label(key: str) -> tuple[str, str]:
    left, measure = key.split(": ", 1)
    a, b = left.split(" vs ", 1)
    nm = lambda s: "graph" if s.startswith("graph") else SYSTEM_NAME.get(s, esc(s))  # noqa: E731
    return f"{nm(a)} vs {nm(b)}", TEST_MEASURE.get(measure, esc(measure))


def tests_table(m: dict, out: Path) -> None:
    rows = []
    for k, v in m["tests"].items():
        comp, meas = test_label(k)
        rows.append([comp, meas, v["n"], signed(v["mean_diff"]), v["wins"], v["losses"],
                     v["ties"], p_table(v["p_wilcoxon"]), p_table(v["p_BH"])])
    tabular("llrrrrrrr", ["comparison", "measure", "queries", "difference", stack("first", "higher"), stack("second", "higher"), "ties", "$p$", "$p$ (BH)"], rows, out)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--queries", default=None, help="the query file (only for the text of the worked example)")
    ap.add_argument("--out", default=str(HERE.parent.parent / "latex" / "figures"))
    ap.add_argument("--ranked", default="bm25,pipeline")
    ap.add_argument("--extra", default="", help="other ranked runs, listed as a second block of the ranked tables without a test (the paper lists none; the runs on the claim followed by the explanation are pooled and judged but not reported)")
    ap.add_argument("--generous", default="relaxed_annotator_a,relaxed_annotator_b")
    ap.add_argument("--sets", default="graph_annotator_a_cr,graph_annotator_b_cr", help="writer A first")
    ap.add_argument("--judges", default="gptoss120b,llama33_70b,deepseekv32", help="the model judges, as in WORK/judge/labels_llm_<judge>.jsonl")
    ap.add_argument("--humans", default="R,M", help="the human judges, as in WORK/labels/labels_<H>.jsonl; printed as H1, H2")
    ap.add_argument("--writer-files", default="ai_a,ai_b", help="suffix of failures_<x>.csv and oracle_<x>.jsonl, one per writer")
    ap.add_argument("--stale-name", default="ai", help="WORK/eval/stale_names_<name>.json")
    ap.add_argument("--writers-name", default="ai", help="WORK/eval/writer_agreement_<name>.json (empty: writer_agreement.json)")
    ap.add_argument("--example-query", default="india-2019-01", help="the worked example of the oracle")
    args = ap.parse_args()
    work, out = Path(args.work), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    nm = lambda s: [x.strip() for x in s.split(",") if x.strip()]  # noqa: E731
    ranked, extra, generous, sets = nm(args.ranked), nm(args.extra), nm(args.generous), nm(args.sets)
    judges, humans, wfiles = nm(args.judges), nm(args.humans), nm(args.writer_files)
    data: dict = {}

    M = {n: load(work / "eval" / f"metrics_{n}.json") for n in ["ai", "ai_before", "human", "ai_check"] + [f"ai_{j}" for j in judges]}
    main_m = M["ai"]
    if main_m is None:
        raise SystemExit("WORK/eval/metrics_ai.json is needed (eval/metrics.py --name ai)")
    S = main_m["summary"]
    measured = sorted(q for q, n in main_m["n_relevant_per_query"].items() if n > 0)
    measured_set = set(measured)

    # ------------------------------------------------------------ 1. what was judged
    nrel = main_m["n_relevant_per_query"]
    rel_counts = sorted(nrel[q] for q in measured)
    mac("resQueriesAll", len(nrel))
    mac("resQueriesMeasured", main_m["n_queries"])
    mac("resQueriesExcluded", len(main_m["excluded_queries_without_relevant"]))
    mac("resExcludedList", ", ".join(main_m["excluded_queries_without_relevant"]))
    mac("resRelevantTotal", num(main_m["total_relevant"]))
    mac("resRelMedian", f"{statistics.median(rel_counts):g}")
    mac("resRelMin", rel_counts[0])
    mac("resRelMax", rel_counts[-1])
    mac("resRelMean", f"{statistics.mean(rel_counts):.1f}")
    mac("resQueriesFewRel", sum(1 for n in rel_counts if n < 5))
    pool_rows = list(read_jsonl(work / "pool" / "pool.jsonl"))
    mac("resPoolPairs", num(len(pool_rows)))
    ext = load(work / "pool" / "extension_before.json")
    mac("resPoolMainPairs", num(ext["main_pairs"]) if ext else num(len(pool_rows)))
    mac("resPoolAddedPairs", num(ext["added_pairs"]) if ext else None)
    mac("resPoolAddedQueries", len(ext["added_by_query"]) if ext else None)
    # the pairs that only a run the paper does not report brought into the pool: their documents are judged and so enter the known
    # relevant documents of every system (printed in the Supplementary Material only)
    reported_runs = set(ranked) | set(generous) | set(sets)
    qrels_main = read_qrels(work / "qrels" / "qrels_ai.txt")
    only_other = {(r["query_id"], r["doc_id"]) for r in pool_rows if not (set(r.get("ranks") or {}) & reported_runs)}
    other_rel = {(q, d) for q, d in only_other if qrels_main.get(q, {}).get(d) == 1}
    measured_without_other = [q for q in measured if any(v == 1 and (q, d) not in other_rel for d, v in qrels_main.get(q, {}).items())]
    mac("resPoolOtherPairs", num(len(only_other)))
    mac("resPoolOtherRelevant", num(len(other_rel)))
    mac("resPoolOtherRelevantPct", pct(len(other_rel) / main_m["total_relevant"]) if main_m["total_relevant"] else None)
    mac("resMeasuredWithoutOther", len(measured_without_other))
    data["relevant_per_query"] = {q: nrel[q] for q in measured}

    # each model's labels: how many, YES share, providers, cost
    labels = {}
    judging_rows = []
    total_cost = 0.0
    for j in judges + ["majority"]:
        p = work / "judge" / f"labels_llm_{j}.jsonl"
        if not p.exists():
            NOTES.append(f"missing: {p}")
            continue
        rows = [r for r in read_jsonl(p) if r.get("label") in ("y", "n")]
        labels[j] = {(r["query_id"], r["doc_id"]): r["label"] for r in rows}
        if j == "majority":
            continue
        prov = Counter(r.get("provider") or "not recorded" for r in rows)
        cost = sum(float(r["cost"]) for r in rows if r.get("cost") is not None)
        total_cost += cost
        secs = [float(r["seconds"]) for r in rows if r.get("seconds") is not None]
        judging_rows.append([JUDGE_NAME.get(j, esc(j)), num(len(rows)), pct(sum(1 for r in rows if r["label"] == "y") / len(rows)),
                             f"{statistics.median(secs):.1f}" if secs else None, f"{cost:.2f}",
                             "; ".join(f"{esc(k)} {num(v)}" for k, v in prov.most_common())])
        data.setdefault("judging", {})[j] = {"labels": len(rows), "providers": dict(prov), "cost_usd": round(cost, 4)}
    if judging_rows:
        tabular("lrrrrL{4.6cm}", ["model", "labels", "YES (\\%)", stack("median", "seconds"), stack("cost", "(USD)"), "provider (labels)"], judging_rows, out / "results_judging.tex")
    mac("resCostTotal", f"{total_cost:.2f}")
    mac("resLabelsPerModel", num(len(next(iter(labels.values())))) if labels else None)

    # The queries left out by country: the drawn set follows the archive's countries; the measured set keeps only the
    # queries with at least one relevant document, and the countries the archive covers thinly lose most of theirs.
    excluded = set(main_m["excluded_queries_without_relevant"])
    qfile = json.load(open(args.queries, encoding="utf-8")) if args.queries and Path(args.queries).exists() else None
    qlist = (qfile.get("queries", []) if isinstance(qfile, dict) else qfile) or []
    country_of = {q["query_id"]: (q.get("stratum") or {}).get("country") for q in qlist if q.get("query_id") in nrel}
    if country_of:
        drawn, kept = Counter(country_of.values()), Counter(c for q, c in country_of.items() if q not in excluded)
        order = sorted(drawn, key=lambda c: (-drawn[c], c))
        c_rows = [[esc(c), drawn[c], kept[c], drawn[c] - kept[c]] for c in order]
        c_rows += ["mid", ["all", sum(drawn.values()), sum(kept.values()), sum(drawn.values()) - sum(kept.values())]]
        tabular("lrrr", ["country of the query", "drawn", stack("measured", "(a relevant", "document)"), "left out"], c_rows, out / "results_countries.tex")
        top2 = order[:2]
        names = lambda cs: " and ".join(cs) if len(cs) <= 2 else ", ".join(cs[:-1]) + ", and " + cs[-1]  # noqa: E731  (the serial comma)
        the = lambda c: ("the " + c) if c in ("United States", "United Kingdom") else c  # noqa: E731
        mac("resTopTwoCountries", names([the(c) for c in top2]))
        mac("resDesignTopTwo", sum(drawn[c] for c in top2))
        mac("resDesignTopTwoPct", pct(sum(drawn[c] for c in top2) / sum(drawn.values())))
        mac("resMeasuredTopTwo", sum(kept[c] for c in top2))
        mac("resMeasuredTopTwoPct", pct(sum(kept[c] for c in top2) / max(1, sum(kept.values()))))
        mac("resCountriesNoneLeft", names(sorted(c for c in drawn if kept[c] == 0)) or "none")
        mac("resCountriesOneLeft", names(sorted(c for c in drawn if kept[c] == 1)) or "none")
        data["queries_by_country"] = {c: {"drawn": drawn[c], "measured": kept[c]} for c in order}
    else:
        (out / "results_countries.tex").write_text("(no query file)\n", encoding="utf-8")
        for n_ in ("resTopTwoCountries", "resDesignTopTwo", "resDesignTopTwoPct", "resMeasuredTopTwo", "resMeasuredTopTwoPct",
                   "resCountriesNoneLeft", "resCountriesOneLeft"):
            mac(n_, None)
    # the left-out queries on which at least one model said yes to some document (a more generous judge keeps them)
    some_yes = {q for j in judges if j in labels for (q, d), v in labels[j].items() if q in excluded and v == "y"}
    mac("resExcludedSomeYes", len(some_yes))
    mac("resQueriesUnderFive", sum(1 for q in nrel if nrel[q] < 5))
    mac("resQueriesExcludedPct", pct(len(excluded) / len(nrel)) if nrel else None)

    # ------------------------------------------------------------ 2. agreement and the main judge
    v = load(work / "judge" / "validation.json")
    hh = v["human_human"] if v else {}
    mac("resKappaHH", f2(hh.get("kappa")))
    lo, hi = (hh.get("bootstrap_95") or [None, None])
    mac("resKappaHHLow", f2(lo))
    mac("resKappaHHHigh", f2(hi))
    mac("resHHObserved", f2(hh.get("observed")))
    mac("resHHExpected", f2(hh.get("expected")))
    mac("resHHShared", hh.get("n"))
    mac("resHHBothYes", hh.get("yy"))
    mac("resHOneYesOnly", hh.get("yn"))
    mac("resHTwoYesOnly", hh.get("ny"))
    mac("resHHBothNo", hh.get("nn"))
    mac("resHOneYesShared", (hh.get("yy") or 0) + (hh.get("yn") or 0) if hh else None)
    mac("resHTwoYesShared", (hh.get("yy") or 0) + (hh.get("ny") or 0) if hh else None)
    hy = (v or {}).get("human_yes_rate", {})
    mac("resHOneYesRate", pct(hy.get(humans[0])) if len(humans) > 0 else None)
    mac("resHTwoYesRate", pct(hy.get(humans[1])) if len(humans) > 1 else None)
    mac("resCheckDocsLabelled", num((v or {}).get("checked_documents_with_final_label")))
    cands = (v or {}).get("candidates", {})
    main_judge = (v or {}).get("main_judge")
    mac("resMainJudge", JUDGE_NAME.get(main_judge, esc(main_judge)) if main_judge else None)
    mac("resMainJudgePassed", "passed" if (v or {}).get("main_judge_passed") else "did not pass")
    mac("resNPassing", sum(1 for c in cands.values() if c.get("passes")))
    mac("resNPassA", sum(1 for c in cands.values() if c.get("pass_a")))
    val_rows = []
    for c, e in cands.items():
        kv = e.get("kappa_vs", {})
        opp = e["order"]["opposite_pairs"]
        val_rows.append([JUDGE_NAME.get(c, esc(c))] + [f2(kv.get(h, {}).get("kappa")) for h in humans]
                        + ["yes" if e["pass_a"] else "no", "yes" if not opp else f"no ({len(opp)} swapped)", f2(e["order"]["kendall_tau_b"]),
                           "\\textbf{yes}" if e["passes"] else "no"])
    if val_rows:
        tabular("lrrcccc", ["candidate"] + [f"$\\kappa$ with {h}" for h in HUMANS[:len(humans)]]
                + ["(a)", "(b) same order", "$\\tau_b$", "passes"], val_rows, out / "results_validation.tex")
    if main_judge in cands:
        kv = cands[main_judge]["kappa_vs"]
        mac("resKappaMainHOne", f2(kv.get(humans[0], {}).get("kappa")))
        mac("resKappaMainHTwo", f2(kv.get(humans[1], {}).get("kappa")))
        order = cands[main_judge]["order"]
        data["order"] = order
    else:
        mac("resKappaMainHOne", None)
        mac("resKappaMainHTwo", None)
    # the order test: the share of checked documents judged relevant, per system and candidate
    if cands:
        first = next(iter(cands.values()))
        systems = list(first["order"]["human"])
        o_rows = [[sname(s, sets, generous), f3(first["order"]["human"][s])] + [f3(e["order"]["ai"][s]) for e in cands.values()] for s in systems]
        tabular("L{3.2cm}r" + "r" * len(cands), ["system", "humans"] + [(stack(*JUDGE_NAME[c].split(" ", 1)) if " " in JUDGE_NAME.get(c, "") else JUDGE_NAME.get(c, esc(c))) for c in cands], o_rows, out / "results_order.tex")
        by_sys = {s: cands[main_judge]["by_system"][s]["n"] for s in systems} if main_judge in cands else {}
        for i, w in enumerate(sets):
            mac(f"resCheckDocsGraph{WRITERS[i]}", by_sys.get(w))
    # every pair of judges on the shared documents, and the models with each other on every pair
    shared = [tuple(x) for x in json.load(open(work / "pool" / "shared.json", encoding="utf-8"))]
    hl = {h: load_annotator(work / "labels" / f"labels_{h}.jsonl")["pool"] for h in humans}
    every = {**{HUMANS[i]: hl[h] for i, h in enumerate(humans)}, **{JUDGE_NAME.get(j, j): labels[j] for j in judges if j in labels}}
    names_ = list(every)
    grid_rows = []
    grid = {}
    for a in names_:
        ks = [k for k in shared if k in every[a]]
        yes = sum(1 for k in ks if every[a][k] == "y")
        row = [esc(a), f"{yes} of {len(ks)}"]
        for b in names_:
            if names_.index(b) >= names_.index(a):
                row.append("")
                continue
            kk = [k for k in shared if k in every[a] and k in every[b]]
            r = cohen_kappa([(every[a][k], every[b][k]) for k in kk])
            grid[f"{a}|{b}"] = r["kappa"]
            row.append(f2(r["kappa"]))
        grid_rows.append(row[:-1] if len(row) > 2 else row)
    if len(names_) > 1:
        tabular("lr" + "r" * (len(names_) - 1), ["judge", "YES"] + [esc(b) for b in names_[:-1]], grid_rows, out / "results_agreement.tex")
    data["kappa_shared"] = grid
    # the main judge on the same shared documents, so that it is compared with the humans on the same documents
    for i, h in enumerate(humans[:2]):
        mj = labels.get(main_judge, {})
        kk = [k for k in shared if k in mj and k in hl[h]]
        mac(f"resKappaMainShared{['HOne', 'HTwo'][i]}", f2(cohen_kappa([(mj[k], hl[h][k]) for k in kk])["kappa"]) if kk else None)
    hum_model = [grid[k] for k in grid if k.split("|")[1] in HUMANS and k.split("|")[0] not in HUMANS]
    mac("resKappaHumanModelMin", f2(min(hum_model)) if hum_model else None)
    mac("resKappaHumanModelMax", f2(max(hum_model)) if hum_model else None)
    mm = []
    for a, b in combinations([j for j in judges if j in labels], 2):
        common = labels[a].keys() & labels[b].keys()
        r = cohen_kappa([(labels[a][k], labels[b][k]) for k in common])
        mm.append((a, b, r["kappa"], r["observed"], len(common)))
    data["kappa_models_all_pairs"] = [{"a": a, "b": b, "kappa": k, "same_answer": o, "n": n} for a, b, k, o, n in mm]
    mac("resKappaModelsMin", f2(min(x[2] for x in mm)) if mm else None)
    mac("resKappaModelsMax", f2(max(x[2] for x in mm)) if mm else None)
    mac("resSameAnswerMin", pct(min(x[3] for x in mm)) if mm else None)
    mac("resSameAnswerMax", pct(max(x[3] for x in mm)) if mm else None)
    allj = [j for j in judges if j in labels]
    if allj:
        common = set.intersection(*(set(labels[j]) for j in allj))
        mac("resAllModelsSame", pct(sum(1 for k in common if len({labels[j][k] for j in allj}) == 1) / len(common)))
        yes_rates = {j: sum(1 for k in common if labels[j][k] == "y") / len(common) for j in allj}
        mac("resModelYesMin", pct(min(yes_rates.values())))
        mac("resModelYesMax", pct(max(yes_rates.values())))
        mac("resModelYesMaxName", JUDGE_NAME.get(max(yes_rates, key=yes_rates.get), ""))
    else:
        for n_ in ("resAllModelsSame", "resModelYesMin", "resModelYesMax", "resModelYesMaxName"):
            mac(n_, None)

    # the check sample: owners and counts
    chk = load(work / "pool" / "check.json")
    owners = Counter(chk["per_query"][q]["owner"] for q in chk["queries"]) if chk else Counter()
    mac("resCheckQueries", len(chk["queries"]) if chk else None)
    mac("resCheckOwnedHOne", owners.get(humans[0]))
    mac("resCheckOwnedHTwo", owners.get(humans[1]) if len(humans) > 1 else None)
    hm = M["human"]
    if hm and chk:
        hm_measured = [q for q, n in hm["n_relevant_per_query"].items() if n > 0]
        mac("resHumanMeasured", len(hm_measured))
        mac("resHumanMeasuredHOne", sum(1 for q in hm_measured if chk["per_query"][q]["owner"] == humans[0]))
        mac("resHumanMeasuredHTwo", sum(1 for q in hm_measured if chk["per_query"][q]["owner"] == humans[1]))
        mac("resHumanRelevant", hm["total_relevant"])
        mac("resHumanRelPerQuery", f"{hm['total_relevant'] / max(1, hm['n_queries']):.1f}")
    else:
        for n_ in ("resHumanMeasured", "resHumanMeasuredHOne", "resHumanMeasuredHTwo", "resHumanRelevant", "resHumanRelPerQuery"):
            mac(n_, None)
    mac("resMainRelPerQuery", f"{main_m['total_relevant'] / max(1, main_m['n_queries']):.1f}")
    ac = M["ai_check"]
    mac("resCheckAIRelevant", ac["total_relevant"] if ac else None)
    mac("resCheckAIMeasured", ac["n_queries"] if ac else None)

    # ------------------------------------------------------------ 3. the main tables
    ranked_table(main_m, ranked, extra, generous, sets, out / "results_ranked.tex")
    sets_table(main_m, sets, out / "results_sets.tex")
    main_results_table(main_m, ranked, generous, sets, out / "results_main.tex")
    at_b_table(main_m, ranked, sets, out / "results_atb.tex")
    tests_table(main_m, out / "results_tests.tex")
    if M["ai_before"]:
        ranked_table(M["ai_before"], ranked, extra, generous, sets, out / "results_ranked_before.tex")
        sets_table(M["ai_before"], sets, out / "results_sets_before.tex")
        main_results_table(M["ai_before"], ranked, generous, sets, out / "results_main_before.tex")
        at_b_table(M["ai_before"], ranked, sets, out / "results_atb_before.tex")
        tests_table(M["ai_before"], out / "results_tests_before.tex")
    short = {"bm25": "BM", "pipeline": "Pipe", "bm25_claimart": "BMClaim", "pipeline_claimart": "PipeClaim"}
    meas = {"P@10": "PTen", "P@20": "PTwenty", "nDCG@20": "nDCG", "R@20": "RTwenty", "R@10": "RTen"}
    for s in ranked + extra:
        if s in short:
            for k, kn in meas.items():
                mac(f"res{short[s]}{kn}", f3(S.get(s, {}).get(k)))
    for i, g in enumerate(generous):
        mac(f"resGen{WRITERS[i]}RTwenty", f3(S.get(g, {}).get("R@20")))
        mac(f"resGen{WRITERS[i]}PTen", f3(S.get(g, {}).get("P@10")))
    for i, w in enumerate(sets):
        s = S.get(w, {})
        W = WRITERS[i]
        mac(f"resGraph{W}Recall", f3(s.get("set_R")))
        mac(f"resGraph{W}Prec", f3(s.get("set_P")))
        mac(f"resGraph{W}Fone", f3(s.get("set_F1")))
        mac(f"resGraph{W}Empty", s.get("queries_B_zero"))
        mac(f"resGraph{W}NonEmpty", s["queries"] - s["queries_B_zero"] if s else None)
        mac(f"resGraph{W}EmptyPct", pct(s["queries_B_zero"] / s["queries"]) if s else None)
        for r in ranked:
            if r in short:
                mac(f"res{short[r]}AtB{W}Prec", f3(S[r].get(f"P@B[{w}]")))
                mac(f"res{short[r]}AtB{W}Recall", f3(S[r].get(f"R@B[{w}]")))
                mac(f"res{short[r]}AtB{W}Fone", f3(S[r].get(f"F1@B[{w}]")))
    gm = S.get("graph (mean of the writers)", {})
    gb = S.get("graph (better writer per query: a ceiling)", {})
    mac("resGraphMeanRecall", f3(gm.get("set_R")))
    mac("resGraphMeanFone", f3(gm.get("set_F1")))
    mac("resGraphMeanPrec", f3(gm.get("set_P")))
    mac("resGraphBestRecall", f3(gb.get("set_R")))
    # the same three numbers as percentages with one decimal, for the abstract and the introduction: with whole
    # percentages (4, 43, 61) a reader who divides them gets ratios (11, 15) other than the two ratios below (10, 14),
    # which are computed from the unrounded means
    mac("resGraphMeanRecallPct", pct1(gm.get("set_R")))
    mac("resBMRTwentyPct", pct1(S[ranked[0]]["R@20"]))
    mac("resPipeRTwentyPct", pct1(S[ranked[1]]["R@20"]))
    ratio = S[ranked[0]]["R@20"] / gm["set_R"] if gm.get("set_R") else None
    mac("resRatioBMGraph", f"{ratio:.0f}" if ratio else None)
    ratio = S[ranked[1]]["R@20"] / gm["set_R"] if gm.get("set_R") else None
    mac("resRatioPipeGraph", f"{ratio:.0f}" if ratio else None)
    judged = [S[s].get(f"judged@{k}") for s in ranked + extra + generous if s in S for k in (10, 20)]
    mac("resJudgedMin", f3(min(x for x in judged if x is not None)) if judged else None)
    tie = [abs(S[s][f"P@{k}"] - S[s][f"P@{k} tie-aware"]) for s in ranked + extra + generous if s in S for k in (10, 20)]
    mac("resTieMaxDiff", f3(max(tie)) if tie else None)
    T = main_m["tests"]
    bp = [k for k in T if k.startswith(f"{ranked[0]} vs {ranked[1]}:")]
    mac("resPBMPipeMax", p_text(max(T[k]["p_BH"] for k in bp)) if bp else None)
    k20 = f"{ranked[0]} vs {ranked[1]}: R@20"
    if k20 in T:
        mac("resRTwentyPipeHigher", T[k20]["losses"])
        mac("resRTwentyBMHigher", T[k20]["wins"])
        mac("resRTwentyTies", T[k20]["ties"])
    gb_bm = [k for k in T if k.startswith("graph") and k.split(": ")[0].endswith(f"vs {ranked[0]}")]
    gb_pipe = [k for k in T if k.startswith("graph") and k.split(": ")[0].endswith(f"vs {ranked[1]}")]
    mac("resPGraphBMMin", p_text(min(T[k]["p_BH"] for k in gb_bm)) if gb_bm else None)
    mac("resPGraphPipeMax", p_text(max(T[k]["p_BH"] for k in gb_pipe)) if gb_pipe else None)
    for k in gb_bm + gb_pipe:
        tag = ("BM" if k in gb_bm else "Pipe") + {"set_P vs P@B": "Prec", "set_R vs R@B": "Recall", "set_F1 vs F1@B": "Fone"}[k.split(": ")[1]]
        mac(f"resTestGraph{tag}N", T[k]["n"])
        mac(f"resTestGraph{tag}GraphHigher", T[k]["wins"])
        mac(f"resTestGraph{tag}OtherHigher", T[k]["losses"])
        mac(f"resTestGraph{tag}Ties", T[k]["ties"])
    # queries where both writers' sets are empty: there the ranked systems are cut at 0 and every comparison is a tie
    pq = main_m["per_query"]
    both_empty = [q for q in measured if all(pq.get(w, {}).get(q, {}).get("B", 0) == 0 for w in sets)]
    mac("resBothEmpty", len(both_empty))
    mac("resAtLeastOneSet", len(measured) - len(both_empty))
    # per-query recall: how many queries a system finds nothing for
    def per_q(s, key):
        return [pq[s][q][key] for q in measured if q in pq.get(s, {})]
    recall_cols = [(SYSTEM_NAME[ranked[0]] + ", top 20", per_q(ranked[0], "R@20"), ORANGE),
                   (SYSTEM_NAME[ranked[1]] + ", top 20", per_q(ranked[1], "R@20"), BLUE)]
    for i, w in enumerate(sets):
        recall_cols.append((f"graph, formula writer {WRITERS[i]}, whole set", per_q(w, "set_R"), AQUA if i == 0 else YELLOW))
    for (label, vals, _), tag in zip(recall_cols, ["BM", "Pipe"] + [f"Graph{W}" for W in WRITERS[:len(sets)]]):
        mac(f"resZero{tag}", sum(1 for x in vals if x == 0))
    gmean_q = main_m["graph_mean_per_query"]
    mac("resZeroGraphMean", sum(1 for q in measured if (gmean_q.get(q, {}).get("set_R") or 0) == 0))
    mac("resGraphAboveBM", sum(1 for q in measured if (gmean_q.get(q, {}).get("set_R") or 0) > pq[ranked[0]][q]["R@20"]))
    mac("resGraphAbovePipe", sum(1 for q in measured if (gmean_q.get(q, {}).get("set_R") or 0) > pq[ranked[1]][q]["R@20"]))
    mac("resPipeAboveBM", sum(1 for q in measured if pq[ranked[1]][q]["R@20"] > pq[ranked[0]][q]["R@20"]))
    data["per_query_recall"] = {label: vals for label, vals, _ in recall_cols}
    # Pooled over the documents instead of averaged per query: how many of all the relevant documents the sets hold,
    # and how many of those come from the two queries where the set found the most (a few very large sets).
    for i, w in enumerate(sets):
        W = WRITERS[i]
        hits = {q: pq.get(w, {}).get(q, {}).get("hits", 0) for q in measured}
        found = sum(hits.values())
        top2 = sum(sorted(hits.values(), reverse=True)[:2])
        mac(f"resGraph{W}FoundDocs", num(found))
        mac(f"resGraph{W}FoundDocsPct", pct(found / main_m["total_relevant"]) if main_m["total_relevant"] else None)
        mac(f"resGraph{W}FoundTopTwo", num(top2))
        mac(f"resGraph{W}FoundTopTwoPct", pct(top2 / found) if found else None)
    # The same pooled count for the top 20 of each ranked system (R@20 times the query's number of relevant documents),
    # so that the pooled view compares like with like. The ratios resRatioBMGraph and resRatioPipeGraph are ratios of
    # per-query MEAN recalls, not of these counts, and the text must say so.
    nrel = main_m["n_relevant_per_query"]
    for s, tag in zip(ranked, ["BM", "Pipe"]):
        found = sum(round(pq[s][q]["R@20"] * nrel[q]) for q in measured if q in pq.get(s, {}))
        mac(f"res{tag}FoundDocs", num(found))
        mac(f"res{tag}FoundDocsPct", pct(found / main_m["total_relevant"]) if main_m["total_relevant"] else None)
    # every query where a writer's set beats a text system at the same size (the paper shows every such query)
    beats = []
    for i, w in enumerate(sets):
        W = WRITERS[i]
        nb = {r: 0 for r in ranked}
        for q in measured:
            g = pq.get(w, {}).get(q)
            if not g:
                continue
            others = {r: pq[r][q].get(f"F1@B[{w}]") for r in ranked}
            won = [r for r in ranked if others[r] is not None and g["set_F1"] > others[r]]
            for r in won:
                nb[r] += 1
            if won:
                beats.append([esc(q), W, g["B"], nrel[q], f3(g["set_F1"])] + [f3(others[r]) for r in ranked])
        for r in ranked:
            mac(f"resBeats{short.get(r, 'X')}{W}", nb[r])
    if beats:
        tabular("lcrrr" + "r" * len(ranked), ["query", "formula writer", "set size", "relevant", "graph F1"] + [f"{SYSTEM_NAME.get(r, esc(r))} F1" for r in ranked],
                beats, out / "results_beats.tex")
    else:
        (out / "results_beats.tex").write_text("(no query)\n", encoding="utf-8")

    # ------------------------------------------------------------ 4. robustness: every judge, before the stop, the humans
    rob = [("ai", f"{JUDGE_NAME.get(main_judge, 'main judge')} (main)")] + [(f"ai_{j}", f"{JUDGE_NAME.get(j, esc(j))} alone") for j in judges] \
        + [("ai_before", "main judge, documents before \\entityStopMonth"), ("ai_check", "main judge, check-sample documents"), ("human", "human judges, check-sample documents")]
    rob_rows, rob_fig = [], []
    pipe_p10, graph_r, p_pipe_bm, p_graph_pipe, p_graph_bm = [], [], [], [], []
    for n_, label in rob:
        m = M.get(n_)
        if not m:
            continue
        s_ = m["summary"]
        t_ = m["tests"]
        g_ = s_.get("graph (mean of the writers)", {}).get("set_R")
        pt = t_.get(f"{ranked[0]} vs {ranked[1]}: R@20", {}).get("p_BH")
        gp = t_.get(f"graph (writers' mean) vs {ranked[1]}: set_R vs R@B", {}).get("p_BH")
        gbm = t_.get(f"graph (writers' mean) vs {ranked[0]}: set_R vs R@B", {}).get("p_BH")
        rob_rows.append([label, m["n_queries"], f3(s_[ranked[0]]["P@10"]), f3(s_[ranked[1]]["P@10"]), f3(s_[ranked[0]]["R@20"]),
                         f3(s_[ranked[1]]["R@20"]), f3(g_), p_table(pt), p_table(gp), p_table(gbm)])
        # 95 percent bootstrap intervals of the three means (the queries drawn again with replacement, a fixed seed per
        # set of labels and system, so the figure is the same at every run); descriptive only, the tests are in the table
        qs_ = [q for q, nr in m["n_relevant_per_query"].items() if nr > 0]
        vals_ = {"bm25": [m["per_query"][ranked[0]][q]["R@20"] for q in qs_],
                 "pipeline": [m["per_query"][ranked[1]][q]["R@20"] for q in qs_],
                 "graph": [m["graph_mean_per_query"][q]["set_R"] for q in qs_ if q in m.get("graph_mean_per_query", {})]}
        ci_ = {k: boot_ci(v, f"20261125:{n_}:{k}") for k, v in vals_.items()}
        short_label = {"ai": "main judge", "ai_before": "main judge, before the stop", "ai_check": "main judge, check sample",
                       "human": "humans, check sample"}.get(n_, f"{JUDGE_NAME.get(n_[3:], n_[3:])} alone")
        rob_fig.append((short_label, m["n_queries"], s_[ranked[0]]["R@20"], s_[ranked[1]]["R@20"], g_, ci_))
        if n_ == "ai" or n_ in {f"ai_{j}" for j in judges}:  # the ranges "under every model judge": the main judge and each model
            pipe_p10.append(s_[ranked[1]]["P@10"])
            graph_r.append(g_)
            # The text claims these over EVERY test of the kind, not only the R@20 and recall shown in the table:
            # the pipeline above BM25 on all four measures, the graph below the pipeline on precision, recall and F1
            # at the same size, and the graph never significantly different from BM25 at the same size.
            pb = [v for k, v in t_.items() if k.startswith(f"{ranked[0]} vs {ranked[1]}:")]
            gpi = [v for k, v in t_.items() if k.startswith("graph") and k.split(": ")[0].endswith(f"vs {ranked[1]}")]
            gbi = [v for k, v in t_.items() if k.startswith("graph") and k.split(": ")[0].endswith(f"vs {ranked[0]}")]
            p_pipe_bm += [v["p_BH"] for v in pb]
            p_graph_pipe += [v["p_BH"] for v in gpi]
            p_graph_bm += [v["p_BH"] for v in gbi]
            # the direction the text states: BM25 minus pipeline and graph minus pipeline are negative on average
            wrong = [k for k, v in t_.items() if v in pb + gpi and v["mean_diff"] is not None and v["mean_diff"] >= 0]
            if wrong:
                NOTES.append(f"{n_}: a difference has the other sign than the text states: {wrong}")
    tabular("L{3.4cm}rrrrrrrrr", ["labels", "queries", stack("BM25", "P@10"), stack("standard IR", "pipeline", "P@10"), stack("BM25", "R@20"), stack("standard IR", "pipeline", "R@20"),
                                  stack("graph", "recall"), stack("$p$ standard IR", "pipeline", "vs BM25"), stack("$p$ graph", "vs standard IR", "pipeline"),
                                  stack("$p$ graph", "vs BM25")],
            rob_rows, out / "results_judges.tex")
    data["robustness"] = rob_fig
    mac("resBootReps", num(BOOT_REPS))
    mac("resRobPipePTenMin", f3(min(pipe_p10)) if pipe_p10 else None)
    mac("resRobPipePTenMax", f3(max(pipe_p10)) if pipe_p10 else None)
    mac("resRobGraphRecallMin", f3(min(x for x in graph_r if x is not None)) if graph_r else None)
    mac("resRobGraphRecallMax", f3(max(x for x in graph_r if x is not None)) if graph_r else None)
    mac("resRobPPipeBMMax", p_text(max(x for x in p_pipe_bm if x is not None)) if p_pipe_bm else None)
    mac("resRobPGraphPipeMax", p_text(max(x for x in p_graph_pipe if x is not None)) if p_graph_pipe else None)
    mac("resRobPGraphBMMin", p_text(min(x for x in p_graph_bm if x is not None)) if p_graph_bm else None)
    for j in judges:
        m = M.get(f"ai_{j}")
        mac("resRobQueries" + {"gptoss120b": "Gpt", "llama33_70b": "Llama", "deepseekv32": "Deepseek"}.get(j, "X" * (judges.index(j) + 1)), m["n_queries"] if m else None)
    mb = M["ai_before"]
    mac("resBeforeQueries", mb["n_queries"] if mb else None)
    mac("resBeforeRemovedPairs", num(mb["qrels_pairs_removed_by_before"]) if mb else None)
    for i, w in enumerate(sets):
        mac(f"resBeforeGraph{WRITERS[i]}Recall", f3(mb["summary"][w]["set_R"]) if mb else None)
        mac(f"resBeforeGraph{WRITERS[i]}Empty", mb["summary"][w]["queries_B_zero"] if mb else None)
    mac("resBeforeBMRTwenty", f3(mb["summary"][ranked[0]]["R@20"]) if mb else None)
    mac("resBeforePipeRTwenty", f3(mb["summary"][ranked[1]]["R@20"]) if mb else None)
    if hm:
        mac("resHumanBMRTwenty", f3(hm["summary"][ranked[0]]["R@20"]))
        mac("resHumanPipeRTwenty", f3(hm["summary"][ranked[1]]["R@20"]))
        mac("resHumanPMin", p_text(min(t["p_BH"] for t in hm["tests"].values())))
        for i, w in enumerate(sets):
            mac(f"resHumanGraph{WRITERS[i]}Judged", f3(hm["summary"][w]["judged_share"]))
            mac(f"resHumanGraph{WRITERS[i]}PrecJudged", f3(hm["summary"][w]["set_P_judged_only"]))
    else:
        for n_ in ["resHumanBMRTwenty", "resHumanPipeRTwenty", "resHumanPMin"] + [f"resHumanGraph{W}{x}" for W in WRITERS[:len(sets)] for x in ("Judged", "PrecJudged")]:
            mac(n_, None)
    mac("resCheckAIBMRTwenty", f3(ac["summary"][ranked[0]]["R@20"]) if ac else None)
    mac("resCheckAIPipeRTwenty", f3(ac["summary"][ranked[1]]["R@20"]) if ac else None)

    # ------------------------------------------------------------ 5. pool completeness (Zobel), every depth 1 to 20
    qrels = read_qrels(work / "qrels" / "qrels_ai.txt")
    runs = {r: read_run(work / "runs" / f"{r}.trec") for r in ranked if (work / "runs" / f"{r}.trec").exists()}
    depths = list(range(1, int(main_m.get("depth", 20)) + 1))
    zc = mt.zobel_curve(runs, qrels, measured, depths)
    for d, k in zip(main_m["zobel_depths"], range(len(main_m["zobel_depths"]))):
        if zc["union"][d - 1] != main_m["zobel_curve"]["union"][k]:
            raise SystemExit(f"the pool-depth curve at depth {d} differs from metrics_ai.json: {zc['union'][d - 1]} vs {main_m['zobel_curve']['union'][k]}")
    data["pool_depth"] = zc
    mac("resZobelTen", num(zc["union"][9]) if len(depths) >= 10 else None)
    mac("resZobelTwenty", num(zc["union"][19]) if len(depths) >= 20 else None)
    mac("resZobelGrowth", pct(zc["union"][19] / zc["union"][9] - 1) if len(depths) >= 20 and zc["union"][9] else None)
    mac("resZobelLastStep", zc["union"][19] - zc["union"][18] if len(depths) >= 20 else None)
    # tied scores: the tables use trec_eval's order (document id descending inside a tie); this is R@20 under any other order
    for r_ in ranked:
        if r_ not in runs:
            continue
        tag_ = {"bm25": "BM", "pipeline": "Pipe"}.get(r_, "Run")
        lo_, hi_, ex_, straddle_ = [], [], [], 0
        for q in measured:
            n_rel_ = sum(1 for v in qrels.get(q, {}).values() if v == 1)
            slots, e, lo, hi, cut = 20, 0.0, 0, 0, False
            for _, grp in groupby(runs[r_].get(q, []), key=lambda t: t[1]):
                if slots <= 0:
                    break
                g = [d for d, _ in grp]
                nr = sum(1 for d in g if qrels.get(q, {}).get(d) == 1)
                if slots >= len(g):
                    e, lo, hi, slots = e + nr, lo + nr, hi + nr, slots - len(g)
                else:
                    e, lo, hi, slots, cut = e + nr * slots / len(g), lo + max(0, nr - (len(g) - slots)), hi + min(nr, slots), 0, True
            straddle_ += cut
            ex_.append(e / n_rel_)
            lo_.append(lo / n_rel_)
            hi_.append(hi / n_rel_)
        mac(f"resTie{tag_}RTwentyExpected", f3(statistics.mean(ex_)) if ex_ else None)
        mac(f"resTie{tag_}RTwentyLow", f3(statistics.mean(lo_)) if lo_ else None)
        mac(f"resTie{tag_}RTwentyHigh", f3(statistics.mean(hi_)) if hi_ else None)
        mac(f"resTie{tag_}Straddle", straddle_)
    # the generous version: how many fact-checks it chooses from (those with at least one entity of the formula), how much of the relevant
    # set that list holds, and what its first 20 and its first 100 results reach
    ent_path = work / "entities_cr.jsonl"
    if ent_path.exists() and generous:
        inv = {}
        for r_ in read_jsonl(ent_path):
            for e_ in r_.get("entities") or []:
                inv.setdefault(e_, set()).add(r_["cr_iri"])

        def term_set(rec):
            t_ = rec.get("terms") or {}
            if isinstance(t_, str):
                t_ = ast.literal_eval(t_)
            out_ = set()

            def walk(x):
                if isinstance(x, str):
                    out_.add(x)
                elif isinstance(x, dict):
                    for v in x.values():
                        walk(v)
                elif isinstance(x, (list, tuple, set)):
                    for v in x:
                        walk(v)
            walk(t_)
            walk(rec.get("alias_terms") or [])
            return out_
        for i, (g_, w_) in enumerate(zip(generous, sets)):
            sp_, gp_ = work / "runs" / f"{w_}_sets.jsonl", work / "runs" / f"{g_}.trec"
            if not (sp_.exists() and gp_.exists()):
                continue
            terms_ = {r_["query_id"]: term_set(r_) for r_ in read_jsonl(sp_)}
            grun_ = read_run(gp_)
            sizes_, reach_, r100_ = [], [], []
            for q in measured:
                rel_ = {d for d, v in qrels.get(q, {}).items() if v == 1}
                if not rel_:
                    continue
                cand_ = set().union(*[inv.get(e_, set()) for e_ in terms_.get(q, set())]) if terms_.get(q) else set()
                sizes_.append(len(cand_))
                reach_.append(len(rel_ & cand_) / len(rel_))
                r100_.append(sum(1 for d, _ in grun_.get(q, [])[:100] if d in rel_) / len(rel_))
            mac(f"resGenCand{WRITERS[i]}", num(int(statistics.median(sizes_))) if sizes_ else None)
            mac(f"resGenReach{WRITERS[i]}", f3(statistics.mean(reach_)) if reach_ else None)
            mac(f"resGenRHundred{WRITERS[i]}", f3(statistics.mean(r100_)) if r100_ else None)

    # ------------------------------------------------------------ 6. why the graph misses: the failure boxes
    fail = {}
    fail_q = {}  # writer -> query -> Counter of the missed documents' boxes (for the shares averaged per query)
    for i, wf in enumerate(wfiles):
        p = work / "eval" / f"failures_{wf}.csv"
        if not p.exists():
            NOTES.append(f"missing: {p}")
            continue
        c = Counter()
        cq = {}
        with open(p, encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                if r["box"] == "FALSE_HIT":
                    c["FALSE_HIT_ALL"] += 1  # every query, also those without a relevant document
                    if r["query_id"] not in measured_set:
                        continue  # the table counts the false hits on the measured queries, like the misses
                c[r["box"]] += 1
                if r["box"] in ("NO_ENTITIES", "ENTITY_MISMATCH", "PARTIAL_MATCH"):
                    qc = cq.setdefault(r["query_id"], Counter())
                    qc[r["box"]] += 1
                    if r["box"] == "NO_ENTITIES":
                        qc["NE:" + r["no_entities_kind"]] += 1
                if r["box"] == "NO_ENTITIES":
                    c["NE:" + r["no_entities_kind"]] += 1
                if r["box"] == "PARTIAL_MATCH" and r["author_flagged_not_expressible"] == "yes":
                    c["PM:flagged"] += 1
        fail[WRITERS[i]] = c
        fail_q[WRITERS[i]] = cq
    f_rows = []
    keys = [("missed relevant documents", "MISSED"), ("no entities", "NO_ENTITIES"), ("\\quad no explanation text", "NE:no_article"),
            ("\\quad published after linking stopped", "NE:after_link_stop"), ("\\quad never linked", "NE:never_linked"),
            ("entity mismatch (none of the formula's entities)", "ENTITY_MISMATCH"), ("partial match (some, not all)", "PARTIAL_MATCH"),
            ("\\quad of which the formula writer noted not expressible", "PM:flagged")]
    for c in fail.values():
        c["MISSED"] = c["NO_ENTITIES"] + c["ENTITY_MISMATCH"] + c["PARTIAL_MATCH"]
    for label, k in keys:
        row = [label]
        for W, c in fail.items():
            share = f" ({pct(c[k] / c['MISSED'])}\\,\\%)" if k != "MISSED" and c["MISSED"] else ""
            row.append(f"{num(c[k])}{share}")
        f_rows.append(row)
    f_rows += ["mid", ["documents in the set judged not relevant (false hits)"] + [num(c["FALSE_HIT"]) for c in fail.values()],
               ["\\quad on all queries, also those without a relevant document"] + [num(c["FALSE_HIT_ALL"]) for c in fail.values()]]
    if fail:
        tabular("l" + "r" * len(fail), [""] + [f"formula writer {W}" for W in fail], f_rows, out / "results_failures.tex")
    data["failures"] = {W: dict(c) for W, c in fail.items()}
    for W, c in fail.items():
        mac(f"resFail{W}Missed", num(c["MISSED"]))
        for tag, k in (("NoEnt", "NO_ENTITIES"), ("Mismatch", "ENTITY_MISMATCH"), ("Partial", "PARTIAL_MATCH")):
            mac(f"resFail{W}{tag}", num(c[k]))
            mac(f"resFail{W}{tag}Pct", pct(c[k] / c["MISSED"]) if c["MISSED"] else None)
        mac(f"resFail{W}Flagged", num(c["PM:flagged"]))
        mac(f"resFail{W}FalseHits", num(c["FALSE_HIT"]))
        for tag, k in (("NoText", "NE:no_article"), ("AfterStop", "NE:after_link_stop"), ("NeverLinked", "NE:never_linked")):
            mac(f"resFail{W}{tag}", num(c[k]))
    # The counts above pool the documents, so a query with many relevant documents weighs more. The same shares
    # averaged per query (each query with a missed document counts once) show whether a few large queries drive them.
    fail_mean = {}
    for W, cq in fail_q.items():
        tot = {q: qc["NO_ENTITIES"] + qc["ENTITY_MISMATCH"] + qc["PARTIAL_MATCH"] for q, qc in cq.items()}
        qs = [q for q in cq if tot[q]]
        fail_mean[W] = {k: statistics.mean(cq[q][k] / tot[q] for q in qs) for k in
                        ("PARTIAL_MATCH", "ENTITY_MISMATCH", "NO_ENTITIES", "NE:no_article", "NE:after_link_stop", "NE:never_linked")} if qs else {}
        fail_mean[W]["queries"] = len(qs)
        for tag, k in (("NoEnt", "NO_ENTITIES"), ("Mismatch", "ENTITY_MISMATCH"), ("Partial", "PARTIAL_MATCH")):
            mac(f"resFail{W}{tag}PctQuery", pct(fail_mean[W].get(k)))
        mac(f"resFail{W}Queries", len(qs))
        top2 = sorted(tot.values(), reverse=True)[:2]
        mac(f"resFail{W}TopTwo", num(sum(top2)))
        mac(f"resFail{W}TopTwoPct", pct(sum(top2) / sum(tot.values())) if tot else None)
    data["failures_mean_per_query"] = fail_mean

    # ------------------------------------------------------------ 7. could better formulas do better? the oracle
    orc = {}
    for i, wf in enumerate(wfiles):
        p = work / "eval" / f"oracle_{wf}.jsonl"
        if p.exists():
            orc[WRITERS[i]] = list(read_jsonl(p))
        else:
            NOTES.append(f"missing: {p}")

    def omean(rows, get):
        xs = [get(r) for r in rows]
        xs = [x for x in xs if x is not None]
        return statistics.mean(xs) if xs else None

    def at(r, fam, t, budget):
        a = r["oracle"][f"{fam}/{t}"][budget]
        if budget == "at_writer_size" and not r["writer_terms"]:
            return None
        return (a or {}).get("F1", 0.0) if a is not None else 0.0

    o_rows = []
    spec = [("the formula writer's formula", lambda r, t: (r["writer"].get(t) if r["writer"] else 0.0)),
            ("oracle, single entities, as many as the formula writer used", lambda r, t: at(r, "singles", t, "at_writer_size")),
            ("oracle, single entities, up to 10", lambda r, t: at(r, "singles", t, "at_k_max")),
            ("oracle, pairs of entities, as many as the formula writer used", lambda r, t: at(r, "pairs", t, "at_writer_size")),
            ("oracle, pairs of entities, up to 10", lambda r, t: at(r, "pairs", t, "at_k_max")),
            ("control: the oracle on random documents, as many as the formula writer used", lambda r, t: r["control_random"]["at_writer_size"] if t == "strict" else None),
            ("control: the same, up to 10", lambda r, t: r["control_random"]["at_k_max"] if t == "strict" else None)]
    for r_ in ranked:
        spec.append((f"{SYSTEM_NAME.get(r_, esc(r_))}, best cutoff", lambda r, t, r_=r_: r["cutoff"][r_][t]["F1"] if r_ in r["cutoff"] else None))
    for label, get in spec:
        row = [label]
        for W, rows in orc.items():
            for t in ("strict", "judged_only"):
                row.append(f3(omean(rows, lambda r, t=t: get(r, t))))
        o_rows.append(row)
    if orc:
        tabular("L{6.2cm}" + "rr" * len(orc), ["mean F1 over the queries"] + [stack("as many entities", f"as writer {W} used", t) for W in orc for t in ("strict", "judged only")], o_rows, out / "results_oracle.tex")
    for W, rows in orc.items():
        prim = [at(r, "singles", "strict", "at_writer_size") for r in rows]
        mac(f"resOracle{W}", f3(omean(rows, lambda r: at(r, "singles", "strict", "at_writer_size"))))
        mac(f"resOracle{W}TenTerms", f3(omean(rows, lambda r: at(r, "singles", "strict", "at_k_max"))))
        mac(f"resOraclePairs{W}", f3(omean(rows, lambda r: at(r, "pairs", "strict", "at_writer_size"))))
        mac(f"resOraclePairs{W}TenTerms", f3(omean(rows, lambda r: at(r, "pairs", "strict", "at_k_max"))))
        mac(f"resOracle{W}JudgedOnly", f3(omean(rows, lambda r: at(r, "singles", "judged_only", "at_writer_size"))))
        tot_, un_ = 0, 0  # the share of the fact-checks that the oracle returns which nobody labelled
        for r in rows:
            a_ = r["oracle"]["singles/strict"]["at_writer_size"]
            if a_ and a_.get("B"):
                tot_ += a_["B"]
                un_ += a_["B"] - (a_.get("judged") or 0)
        mac(f"resOracleUnlabelled{W}", pct(un_ / tot_) if tot_ else None)
        mac(f"resControl{W}", f3(omean(rows, lambda r: r["control_random"]["at_writer_size"])))
        mac(f"resControl{W}TenTerms", f3(omean(rows, lambda r: r["control_random"]["at_k_max"])))
        mac(f"resWriterFormula{W}", f3(omean(rows, lambda r: r["writer"].get("strict") if r["writer"] else 0.0)))
        comps = [("Control", [r["control_random"]["at_writer_size"] for r in rows]),
                 ("Writer", [r["writer"].get("strict") if r["writer"] else 0.0 for r in rows])]
        for r_ in ranked:
            comps.append((f"Cut{short.get(r_, 'X')}", [r["cutoff"][r_]["strict"]["F1"] if r_ in r["cutoff"] else None for r in rows]))
        for tag, other in comps:
            t = paired_exact(prim, other)
            mac(f"resOracle{W}Minus{tag}", signed(t["mean_diff"]))
            mac(f"resOracle{W}Minus{tag}P", p_text(t["p_wilcoxon"]))
            mac(f"resOracle{W}Minus{tag}Higher", t["wins"])
            mac(f"resOracle{W}Minus{tag}Lower", t["losses"])
    # the other budgets against the pipeline's best cutoff (context; the primary number is above)
    for W, rows in orc.items():
        cut = [r["cutoff"][ranked[1]]["strict"]["F1"] if ranked[1] in r["cutoff"] else None for r in rows]
        ten = [at(r, "singles", "strict", "at_k_max") for r in rows]
        for tag, a_vals, b_vals in (("TenMinusCutPipe", ten, cut),
                                    ("PairsMinusCutPipe", [at(r, "pairs", "strict", "at_writer_size") for r in rows], cut),
                                    ("TenMinusControlTen", ten, [r["control_random"]["at_k_max"] for r in rows])):
            t = paired_exact(a_vals, b_vals)
            mac(f"resOracle{W}{tag}", signed(t["mean_diff"]))
            mac(f"resOracle{W}{tag}P", p_text(t["p_wilcoxon"]))
    if orc:
        rowsA = next(iter(orc.values()))
        for r_ in ranked:
            mac(f"resCut{short.get(r_, 'X')}", f3(omean(rowsA, lambda r: r["cutoff"][r_]["strict"]["F1"] if r_ in r["cutoff"] else None)))
            mac(f"resCut{short.get(r_, 'X')}JudgedOnly", f3(omean(rowsA, lambda r: r["cutoff"][r_]["judged_only"]["F1"] if r_ in r["cutoff"] else None)))
        mac("resMaxRecallEntities", f3(omean(rowsA, lambda r: r["max_recall_by_entities"])))
        mac("resNoEntityRelevantPct", pct(1 - omean(rowsA, lambda r: r["max_recall_by_entities"])))
        # the precision and recall behind the F1: the oracle (writer A's budget) and the ranked systems at their best cutoff
        pr = {}
        for r in rowsA:
            o = r["oracle"]["singles/strict"]["at_writer_size"] if r["writer_terms"] else None
            b, h = ((o or {}).get("B") or 0), ((o or {}).get("hits") or 0)
            pr.setdefault("oracle", []).append((h / b if b else 0.0, h / r["n_relevant"], b))
            rel = {d for d, x in qrels.get(r["query_id"], {}).items() if x == 1}
            for r_ in runs:
                k = r["cutoff"][r_]["strict"]["k"]
                docs = [d for d, _ in runs[r_].get(r["query_id"], [])][:k]
                hits = sum(1 for d in docs if d in rel)
                pr.setdefault(r_, []).append((hits / k if k else 0.0, hits / r["n_relevant"], k))
        for key, tag in [("oracle", "Oracle")] + [(r_, f"Cut{short.get(r_, 'X')}") for r_ in runs]:
            xs = pr.get(key, [])
            mac(f"res{tag}MeanPrec", f3(statistics.mean(x[0] for x in xs)) if xs else None)
            mac(f"res{tag}MeanRecall", f3(statistics.mean(x[1] for x in xs)) if xs else None)
            mac(f"res{tag}MeanSize", f"{statistics.mean(x[2] for x in xs):.1f}" if xs else None)
        data["oracle_precision_recall"] = {k: {"precision": statistics.mean(x[0] for x in xs), "recall": statistics.mean(x[1] for x in xs),
                                               "size": statistics.mean(x[2] for x in xs)} for k, xs in pr.items() if xs}
        curves = {f"{fam}/{t}": [statistics.mean(r["oracle"][f"{fam}/{t}"]["curve"][k] for r in rowsA) for k in range(len(rowsA[0]["oracle"]["singles/strict"]["curve"]))]
                  for fam in ("singles", "pairs") for t in ("strict", "judged_only") if f"{fam}/{t}" in rowsA[0]["oracle"]}
        if all("curve" in r["control_random"] for r in rowsA):
            curves["control"] = [statistics.mean(r["control_random"]["curve"][k] for r in rowsA) for k in range(len(curves["singles/strict"]))]
        data["oracle_curves"] = curves
        # the curve at one page and at the last budget: the oracle against its control
        mac("resOracleCurveOne", f3(curves["singles/strict"][0]))
        mac("resOracleCurveLast", f3(curves["singles/strict"][-1]))
        mac("resControlCurveOne", f3(curves["control"][0]) if "control" in curves else None)
        mac("resControlCurveLast", f3(curves["control"][-1]) if "control" in curves else None)
        # the worked example
        ex = next((r for r in rowsA if r["query_id"] == args.example_query), None)
        qtext = None
        if args.queries and Path(args.queries).exists():
            qs = json.load(open(args.queries, encoding="utf-8"))
            qs = qs.get("queries", qs) if isinstance(qs, dict) else qs
            it = qs.get(args.example_query) if isinstance(qs, dict) else next((x for x in qs if x.get("query_id") == args.example_query), None)
            qtext = (it or {}).get("free_text")
        mac("resExQueryId", esc(args.example_query))
        mac("resExQueryText", esc(qtext) if qtext else None)
        if ex:
            o = ex["oracle"]["singles/strict"]["at_writer_size"] or ex["oracle"]["singles/strict"]["at_k_max"]
            mac("resExRelevant", ex["n_relevant"])
            mac("resExOracleFormula", formula_tex(o["formula"]) if o else None)
            mac("resExOracleSize", o["B"] if o else None)
            mac("resExOracleHits", o["hits"] if o else None)
            mac("resExOracleFone", f2(o["F1"]) if o else None)
            mac("resExControl", f2(ex["control_random"]["at_writer_size"]))
            pc = ex["cutoff"].get(ranked[1], {}).get("strict", {})
            mac("resExPipeBestK", pc.get("k"))
            mac("resExPipeBestFone", f2(pc.get("F1")))
            rel = {d for d, x in qrels.get(args.example_query, {}).items() if x == 1}
            docs = [d for d, _ in runs.get(ranked[1], {}).get(args.example_query, [])]
            ranks = [i + 1 for i, d in enumerate(docs) if d in rel]
            mac("resExPipeRanks", ", ".join(str(x) for x in ranks) if ranks else "none")
            ex_rows = []
            h = 0
            for k in range(1, min(20, len(docs)) + 1):
                h += docs[k - 1] in rel
                p_, r_ = h / k, h / len(rel)
                ex_rows.append([k, h, f2(p_), f2(r_), f2(2 * p_ * r_ / (p_ + r_) if h else 0.0)])
            pick = {1, 3, pc.get("k") or 0, 10, 20}
            tabular("rrrrr", ["stop after", "relevant found", "precision", "recall", "F1"], [r for r in ex_rows if r[0] in pick], out / "results_example.tex")
            for i, w in enumerate(sets):
                sp = work / "runs" / f"{w}_sets.jsonl"
                size = next((len(r["doc_ids"]) for r in read_jsonl(sp) if r["query_id"] == args.example_query), None) if sp.exists() else None
                mac(f"resExSetSize{WRITERS[i]}", size)
        else:
            for n_ in ["resExRelevant", "resExOracleFormula", "resExOracleSize", "resExOracleHits", "resExOracleFone", "resExControl",
                       "resExPipeBestK", "resExPipeBestFone", "resExPipeRanks"] + [f"resExSetSize{W}" for W in WRITERS[:len(sets)]]:
                mac(n_, None)
            (out / "results_example.tex").write_text("(no example)\n", encoding="utf-8")

    # ------------------------------------------------------------ 8. old entity names, the two writers together
    st = load(work / "eval" / f"stale_names_{args.stale_name}.json")
    for i, w in enumerate(sets):
        s = (st or {}).get("writers", {}).get(w)
        W = WRITERS[i]
        mac(f"resStale{W}RecallMain", f3(s["mean_recall_main"]) if s else None)
        mac(f"resStale{W}RecallCurrent", f3(s["mean_recall_current"]) if s else None)
        mac(f"resStale{W}RelLost", s["relevant_lost"] if s else None)
        mac(f"resStale{W}RelFound", s["relevant_found_main"] if s else None)
        mac(f"resStale{W}QueriesLosing", s["queries_losing_relevant"] if s else None)
        mac(f"resStale{W}EmptyCurrent", s["empty_sets_current"] if s else None)
        mac(f"resStale{W}EmptyMain", s["empty_sets_main"] if s else None)
    wa = load(work / "eval" / (f"writer_agreement_{args.writers_name}.json" if args.writers_name else "writer_agreement.json"))
    # The two writers' sets joined by OR, on the measured queries, measured as the sets of Table 'sets' are (an empty
    # set: recall and F1 zero; precision over the non-empty sets). writer_agreement.py also gives these, but its
    # precision averages over every query with a non-empty set, also those without a relevant document, so it is not
    # comparable with the table; its recall and F1 must equal ours, and are checked.
    union = {}
    for w in sets:
        sp = work / "runs" / f"{w}_sets.jsonl"
        if sp.exists():
            for r in read_jsonl(sp):
                if r.get("formula"):
                    union.setdefault(r["query_id"], set()).update(r["doc_ids"])
    ps, rs, fs = [], [], []
    for q in measured:
        rel = {d for d, x in qrels.get(q, {}).items() if x == 1}
        u = union.get(q, set())
        h = len(u & rel)
        p_, r_ = (h / len(u) if u else None), h / len(rel)
        if p_ is not None:
            ps.append(p_)
        rs.append(r_)
        fs.append(2 * p_ * r_ / (p_ + r_) if h else 0.0)
    mac("resWritersOrRecall", f3(statistics.mean(rs)) if rs else None)
    mac("resWritersOrPrec", f3(statistics.mean(ps)) if ps else None)
    mac("resWritersOrNonEmpty", len(ps))
    mac("resWritersOrFone", f3(statistics.mean(fs)) if fs else None)
    aob = (wa or {}).get("summary", {}).get("A_or_B", {})
    for mine, theirs, what in ((rs, aob.get("mean_set_recall"), "recall"), (fs, aob.get("mean_set_F1"), "F1")):
        if theirs is not None and mine and abs(statistics.mean(mine) - theirs) > 0.0006:
            raise SystemExit(f"the two writers together: {what} {statistics.mean(mine):.4f} here, {theirs} in writer_agreement")
    pw = (wa or {}).get("summary", {}).get("per_writer", {})
    for W in WRITERS[:len(sets)]:
        mac(f"resWriter{W}Concepts", f"{pw[W]['mean_and_groups']:.1f}" if W in pw else None)

    # ------------------------------------------------------------ 9. the figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style(plt)

    # (a) per-query recall, one column per system
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    rng = random.Random(20261125)
    for x, (label, vals, colour) in enumerate(recall_cols):
        xs = [x + rng.uniform(-0.28, 0.28) for _ in vals]
        ax.scatter(xs, vals, s=9, color=colour, alpha=0.7, linewidths=0)
        mval = statistics.mean(vals) if vals else 0
        ax.plot([x - 0.36, x + 0.36], [mval, mval], color="black", linewidth=1.6)
        z = sum(1 for v_ in vals if v_ == 0)
        ax.text(x, -0.13, f"{z} of {len(vals)} at 0", ha="center", va="top", fontsize=7, color=GREY)
    ax.set_xticks(range(len(recall_cols)))
    ax.set_xticklabels([c[0].replace(", ", ",\n") + f"\nmean {statistics.mean(c[1]):.2f}" for c in recall_cols], fontsize=7.5)
    ax.set_ylim(-0.2, 1.05)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlim(-0.6, len(recall_cols) - 0.2)
    ax.set_ylabel("recall of one query")
    ax.grid(axis="x", visible=False)
    save(fig, out / "results_recall.pdf")
    plt.close(fig)

    # (b) every labelling gives the same order
    if rob_fig:
        fig, ax = plt.subplots(figsize=(4.9, 0.42 * len(rob_fig) + 0.9))
        ys = list(range(len(rob_fig)))[::-1]
        for y, (label, n_q, bmv, pv, gv, ci) in zip(ys, rob_fig):
            # one row per set of labels; the three systems a little apart so that their intervals do not overlap
            for val, key, colour, dy in ((pv, "pipeline", BLUE, 0.17), (bmv, "bm25", ORANGE, 0.0), (gv, "graph", AQUA, -0.17)):
                if val is None:
                    continue
                lo, hi = ci.get(key, (None, None))
                if lo is not None:
                    ax.plot([lo, hi], [y + dy, y + dy], color=colour, linewidth=1.6, solid_capstyle="round", zorder=2)
                ax.scatter([val], [y + dy], s=30, color=colour, edgecolors="white", linewidths=1.2, zorder=3)
        ax.set_yticks(ys)
        ax.set_yticklabels([f"{lab}\n({n_q} queries)" for lab, n_q, *_ in rob_fig], fontsize=7.5)
        ax.set_xlim(0, 1)
        ax.set_xlabel("mean recall, with its 95% bootstrap interval")
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], marker="o", color=c, markersize=5, linewidth=1.6, label=l_) for c, l_ in
                   ((BLUE, f"{SYSTEM_NAME[ranked[1]]}, top 20"), (ORANGE, f"{SYSTEM_NAME[ranked[0]]}, top 20"), (AQUA, "graph, whole set (mean of the formula writers)"))]
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, fontsize=7.5)
        ax.grid(axis="y", visible=False)
        save(fig, out / "results_judges.pdf")
        plt.close(fig)

    # (c) the failure boxes
    if fail:
        parts = [("partial match", "PARTIAL_MATCH", BLUE, 1.0), ("entity mismatch", "ENTITY_MISMATCH", ORANGE, 1.0),
                 ("no entities: no explanation", "NE:no_article", AQUA, 1.0), ("no entities: after the stop", "NE:after_link_stop", AQUA, 0.65),
                 ("no entities: never linked", "NE:never_linked", AQUA, 0.35)]
        # two bars per writer: the missed documents pooled (each document counts once, so large queries weigh more)
        # and the shares averaged over the queries (each query counts once)
        bars = []
        for W, c in fail.items():
            bars.append((f"formula writer {W}, all missed documents\n({c['MISSED']:,} documents)", {k: (c[k] / c["MISSED"] if c["MISSED"] else 0) for _, k, *_ in parts}))
            fm = fail_mean.get(W, {})
            bars.append((f"formula writer {W}, mean over queries\n({fm.get('queries', 0)} queries)", {k: fm.get(k, 0) for _, k, *_ in parts}))
        fig, ax = plt.subplots(figsize=(5.0, 0.5 * len(bars) + 0.9))
        ys = list(range(len(bars)))[::-1]
        for y, (blabel, shares) in zip(ys, bars):
            left = 0.0
            for label, k, colour, alpha in parts:
                share = shares[k]
                ax.barh(y, share, left=left, color=colour, alpha=alpha, height=0.62, edgecolor="white", linewidth=1.5, label=label if y == ys[0] else None)
                if share >= 0.045:
                    ax.text(left + share / 2, y, f"{100 * share:.0f}%", ha="center", va="center", fontsize=7, color="white" if alpha > 0.5 else "black")
                left += share
        ax.set_yticks(ys)
        ax.set_yticklabels([b[0] for b in bars], fontsize=7.5)
        ax.set_xlim(0, 1)
        ax.set_xlabel("share of the relevant documents the set misses")
        ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=2, fontsize=7)
        ax.grid(axis="y", visible=False)
        save(fig, out / "results_failures.pdf")
        plt.close(fig)

    # (d) the oracle: mean F1 after k pages
    if orc:
        fig, ax = plt.subplots(figsize=(6.6, 3.0))
        ks = list(range(1, len(curves["singles/strict"]) + 1))
        ax.plot(ks, curves["singles/strict"], color=AQUA, marker="o", markersize=4, linewidth=2, label="oracle, single entities")
        if "pairs/strict" in curves:
            ax.plot(ks, curves["pairs/strict"], color=AQUA, marker="s", markersize=4, linewidth=2, linestyle="--", label="oracle, pairs of entities")
        rowsA = next(iter(orc.values()))
        if all("curve" in r["control_random"] for r in rowsA):
            # the control is a curve too: the same greedy search on random sets, after 1 to 10 pages
            ax.plot(ks, curves["control"], color=GREY, marker="o", markersize=3.5, linewidth=1.6,
                    label="control: the single-entity oracle on random documents")
            hlines = []
        else:  # an older oracle file without the control's curve: only its end point is known
            ax.plot([ks[-1]], [omean(rowsA, lambda r: r["control_random"]["at_k_max"])], color=GREY, marker="o", linestyle="",
                    label=f"control: the same oracle on random documents, {ks[-1]} entities")
            hlines = []
        hlines += [(omean(rowsA, lambda r: r["cutoff"][ranked[1]]["strict"]["F1"] if ranked[1] in r["cutoff"] else None), BLUE, "--", f"{SYSTEM_NAME[ranked[1]]}, best cutoff"),
                  (omean(rowsA, lambda r: r["cutoff"][ranked[0]]["strict"]["F1"] if ranked[0] in r["cutoff"] else None), ORANGE, "--", f"{SYSTEM_NAME[ranked[0]]}, best cutoff"),
                  (statistics.mean(omean(rows, lambda r: r["writer"].get("strict") if r["writer"] else 0.0) for rows in orc.values()), YELLOW, "-", "the formula writers' formulas")]
        for val, colour, ls, label in hlines:
            if val is not None:
                ax.axhline(val, color=colour, linestyle=ls, linewidth=1.4, label=label)
        ax.set_xticks(ks)
        ax.set_xlabel("number of entities (or pairs of entities) joined by OR")
        ax.set_ylabel("mean F1 over the queries")
        ax.set_ylim(0, 1)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=7.5)
        save(fig, out / "results_oracle.pdf")
        plt.close(fig)

    # (e) pool completeness: relevant documents found by depth
    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    for key, colour, label in ((ranked[1], BLUE, SYSTEM_NAME[ranked[1]]), (ranked[0], ORANGE, SYSTEM_NAME[ranked[0]]), ("union", GREY, "both together")):
        if key in zc:
            ax.plot(depths, zc[key], color=colour, linewidth=2, label=label)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlabel("depth (documents read from the top of each list)")
    ax.set_ylabel("relevant documents found\n(summed over the queries)")
    ax.legend(fontsize=7.5)
    save(fig, out / "results_pool_depth.pdf")
    plt.close(fig)

    # ------------------------------------------------------------ write
    lines = ["% GENERATED by paper-code/eval/paper_results.py from the evaluation files of the work folder. Do not edit:",
             "% re-run the evaluation scripts and then this script."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in MACROS.items()]
    (out / "results_numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    data["macros"] = MACROS
    write_json(out / "results.json", data)
    print(f"wrote {len(MACROS)} macros, the tables and the figures into {out}")
    for n_ in NOTES:
        print("NOTE:", n_)


if __name__ == "__main__":
    main()
