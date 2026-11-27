"""
writer_agreement.py -- how much the two writers' formulas for the same narrative agree.

The question. Two people read the same narrative and each writes a
Boolean formula over Wikipedia pages. Do they choose the same pages? Do their formulas
find the same fact-checks? Saracevic and Kantor (JASIS 1988, Part III) asked the same of
searchers who worked from the same written question: on average two searchers shared
0.27 of their search terms and 0.17 of the items they retrieved.

Only the queries that BOTH writers formulated count (a rejected row is sent back to its
writer and imported again; until then the query has one formula and is skipped).

Two overlap measures, both reported:
  Jaccard        shared / (in A or in B). Symmetric, one number per query.
                 A finds 30, B finds 50, 12 in both: 12 / (30 + 50 - 12) = 12 / 68 = 0.18.
  S&K overlap    shared / (what one writer found), once from each side, as Saracevic and
                 Kantor computed it (Part III, "Overlap Studies"): 12 / 30 = 0.40 and
                 12 / 50 = 0.24. The paper quotes this one next to their 0.27 and 0.17.
When both sets are empty, nothing can be compared (0 / 0): the query is counted in the
"both empty" row and left out of the means, so two empty formulas never count as agreement.

Pages are compared after the import's redirect resolution (import_formulas.py
--resolve-redirects), so "Global warming" and "Climate change" are one page; upper and
lower case and underscores are ignored ("lie" = "Lie").

Usage:
  python systems/graph_boolean.py --queries Q.json --store STORE --work WORK --node cr --backend index --author annotator_a
  python systems/graph_boolean.py --queries Q.json --store STORE --work WORK --node cr --backend index --author annotator_b
  python eval/writer_agreement.py --work WORK --queries Q.json --authors annotator_a,annotator_b [--qrels WORK/qrels/qrels_ai.txt]

Outputs in WORK/eval/ (with --name N: writer_agreement_N.*):
  writer_agreement.md             the summary, in words
  writer_agreement.json           the summary and every query's numbers
  writer_agreement_per_query.csv  one row per query, numbers only (open it in Excel)
No document, no page title and no narrative is written: counts only. The archive counts
(how many fact-checks a formula finds) are never shown to the writers.

--tex FILE: also writes the numbers the paper quotes as LaTeX macros (latex/figures/writer_numbers.tex),
so that no number about the writers is typed by hand. With --writers DIR (the writers' folder) it adds the repairs:
the titles sent back to each writer as a disambiguation page or not a page (every titoli_da_rivedere*.csv row), the
formulas repaired for a typing slip and for AND and OR mixed without brackets (repair_log.csv). It also counts, from the
query file, the formulas in which a page counts under an older name (alias_kind 'older name', import_formulas.py) and
the formulas with a page typed under another name (alias_kind 'as written').
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.boolean import And, Not, Or, parse  # noqa: E402
from common.jsonl import read_jsonl, write_json  # noqa: E402
from common.queries import load_queries, pick_formula  # noqa: E402
from common.trec import read_qrels  # noqa: E402
from queries.entity_catalog import norm_key  # noqa: E402

SK_TERMS, SK_ITEMS = 0.27, 0.17  # Saracevic and Kantor 1988, Part III, Tables 27 and 28


# ---------------------------------------------------------------------------
# Small measures
# ---------------------------------------------------------------------------
def jaccard(a: set, b: set) -> float | None:
    """shared / union; None when both are empty (nothing to compare)."""
    return len(a & b) / len(a | b) if (a | b) else None


def sk_overlap(base: set, other: set) -> float | None:
    """Saracevic and Kantor's asymmetric overlap: shared / the base searcher's count; None if the base is empty."""
    return len(base & other) / len(base) if base else None


def pages_of(formula: dict) -> set[str]:
    """The Wikipedia pages of a formula, in one spelling: the address the import resolved,
    decoded, underscores as spaces, case ignored (the same key the entity catalog uses)."""
    out = set()
    for e in (formula.get("entities") or {}).values():
        if e.get("alias_of"):
            continue  # the page as typed, kept next to today's page (import rule 'either page'): not a second choice of the writer
        if e.get("dbpedia_uri"):
            out.add(norm_key(e["dbpedia_uri"]))
        elif e.get("wikipedia_title") or e.get("label"):
            out.add((e.get("wikipedia_title") or e.get("label")).replace("_", " ").strip().casefold())
    return out


def and_groups(node) -> int:
    """How many concepts the formula joins with AND at the top: ("a" OR "b") AND "c" -> 2.
    A nested AND counts its own parts, so (a AND b) AND c -> 3."""
    if isinstance(node, And):
        return sum(and_groups(c) if isinstance(c, And) else 1 for c in node.children)
    return 1


def uses_not(node) -> bool:
    if isinstance(node, Not):
        return True
    if isinstance(node, (And, Or)):
        return any(uses_not(c) for c in node.children)
    return False


def case_of(a: set, b: set) -> str:
    if not a and not b:
        return "both empty"
    if not a:
        return "only A empty"
    if not b:
        return "only B empty"
    if a == b:
        return "identical sets"
    if a & b:
        return "both found, some shared"
    return "both found, nothing shared"


CASES = ["both empty", "only A empty", "only B empty", "both found, nothing shared", "both found, some shared", "identical sets"]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 3) if xs else None


def median(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 3) if xs else None


def r_squared(xs, ys):
    """Share of the variation of ys that a straight line on xs explains (Pearson r squared);
    None with fewer than 3 pairs or a constant column."""
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    a, b = zip(*pairs)
    if len(set(a)) < 2 or len(set(b)) < 2:
        return None
    return round(statistics.correlation(a, b) ** 2, 3)


def set_scores(found: set, rel: set):
    """Set precision, recall, F1. An empty set has no precision (None); recall needs a relevant document."""
    hits = len(found & rel)
    p = hits / len(found) if found else None
    r = hits / len(rel) if rel else None
    if p is None or r is None:
        f1 = 0.0 if (r is not None and not found) else None
    else:
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return p, r, f1


def b_band(b: int) -> str:
    return "0" if b == 0 else "1-9" if b < 10 else "10-99" if b < 100 else "100-999" if b < 1000 else "1000+"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--queries", required=True, help="the query file with both writers' formulas")
    ap.add_argument("--authors", required=True, help="the two author ids, writer A first: annotator_a,annotator_b")
    ap.add_argument("--node", default="cr", help="the node the graph runs read (cr in the paper; the toy test uses claim)")
    ap.add_argument("--qrels", default=None, help="optional: adds each writer's set precision, recall and F1")
    ap.add_argument("--name", default=None, help="suffix of the output files, e.g. human or full")
    ap.add_argument("--tex", default=None, help="also write the paper's macros about the writers here (latex/figures/writer_numbers.tex)")
    ap.add_argument("--writers", default=None, help="with --tex: the writers' folder, for the repair counts (titoli_da_rivedere*.csv, repair_log.csv)")
    args = ap.parse_args()

    work = Path(args.work)
    authors = [a.strip() for a in args.authors.split(",") if a.strip()]
    if len(authors) != 2:
        raise SystemExit("--authors takes exactly two author ids, e.g. annotator_a,annotator_b")
    A, B = authors
    queries = load_queries(args.queries)
    qrels = read_qrels(args.qrels) if args.qrels else {}

    sets = {}
    for a in authors:
        p = work / "runs" / f"graph_{a}_{args.node}_sets.jsonl"
        if not p.exists():
            raise SystemExit(f"missing {p}: run systems/graph_boolean.py --author {a} --node {args.node} first")
        sets[a] = {r["query_id"]: set(r["doc_ids"]) for r in read_jsonl(p) if r.get("formula")}

    rows, only = [], {A: [], B: [], "neither": []}
    for q in queries:
        qid = q["query_id"]
        fa, fb = pick_formula(q, A), pick_formula(q, B)
        if fa is None or fb is None:
            only[A if fa else B if fb else "neither"].append(qid)
            continue
        for a in authors:
            if qid not in sets[a]:
                raise SystemExit(f"{qid}: the graph run of {a} has no set for this query; run graph_boolean.py again on this query file")
        pa, pb = pages_of(fa), pages_of(fb)
        sa, sb = sets[A][qid], sets[B][qid]
        na, nb = parse(fa["formula"]), parse(fb["formula"])
        row = {
            "query_id": qid,
            "pages_A": len(pa), "pages_B": len(pb), "pages_shared": len(pa & pb),
            "J_pages": jaccard(pa, pb), "SK_pages_A_base": sk_overlap(pa, pb), "SK_pages_B_base": sk_overlap(pb, pa),
            "B_A": len(sa), "B_B": len(sb), "docs_shared": len(sa & sb), "docs_A_or_B": len(sa | sb),
            "J_sets": jaccard(sa, sb), "SK_sets_A_base": sk_overlap(sa, sb), "SK_sets_B_base": sk_overlap(sb, sa),
            "case": case_of(sa, sb),
            "and_groups_A": and_groups(na), "and_groups_B": and_groups(nb),
            "NOT_A": uses_not(na), "NOT_B": uses_not(nb),
            "minutes_A": fa.get("minutes"), "minutes_B": fb.get("minutes"),
            "refinements_A": fa.get("refinements"), "refinements_B": fb.get("refinements"),
            "not_expressible_A": len(fa.get("not_expressible") or []), "not_expressible_B": len(fb.get("not_expressible") or []),
        }
        if qrels:
            rel = {d for d, r in qrels.get(qid, {}).items() if r >= 1}
            row["relevant_in_pool"] = len(rel)
            for tag, s in (("A", sa), ("B", sb), ("A_or_B", sa | sb)):
                row[f"P_{tag}"], row[f"R_{tag}"], row[f"F1_{tag}"] = set_scores(s, rel)
        rows.append(row)

    def col(k):
        return [r[k] for r in rows]

    sk_pages = col("SK_pages_A_base") + col("SK_pages_B_base")
    sk_sets = col("SK_sets_A_base") + col("SK_sets_B_base")
    both_found = [r for r in rows if r["B_A"] and r["B_B"]]
    summary = {
        "authors": {"A": A, "B": B},
        "queries_in_file": len(queries),
        "queries_both_wrote": len(rows),
        "queries_only_A_wrote": len(only[A]), "queries_only_B_wrote": len(only[B]), "queries_neither_wrote": len(only["neither"]),
        "only_A_wrote": only[A], "only_B_wrote": only[B],
        "pages": {
            "mean_jaccard": mean(col("J_pages")), "median_jaccard": median(col("J_pages")),
            "mean_sk_overlap": mean(sk_pages),
            "queries_no_page_shared": sum(1 for r in rows if r["pages_shared"] == 0),
            "queries_same_pages": sum(1 for r in rows if r["J_pages"] == 1),
        },
        "sets": {
            "cases": {c: sum(1 for r in rows if r["case"] == c) for c in CASES},
            "mean_jaccard_at_least_one_found": mean(col("J_sets")),
            "median_jaccard_at_least_one_found": median(col("J_sets")),
            "mean_jaccard_both_found": mean([r["J_sets"] for r in both_found]),
            "mean_sk_overlap": mean(sk_sets),
            "r2_page_overlap_explains_set_overlap": r_squared(col("J_pages"), col("J_sets")),
        },
        "per_writer": {},
    }
    for tag, a in (("A", A), ("B", B)):
        Bs = col(f"B_{tag}")
        mins = [m for m in col(f"minutes_{tag}") if isinstance(m, (int, float))]
        summary["per_writer"][tag] = {
            "author_id": a,
            "mean_pages": mean(col(f"pages_{tag}")), "mean_and_groups": mean(col(f"and_groups_{tag}")),
            "formulas_with_NOT": sum(1 for x in col(f"NOT_{tag}") if x),
            "empty_sets": sum(1 for b in Bs if b == 0), "median_B": median(Bs), "max_B": max(Bs) if Bs else None,
            "B_bands": {band: sum(1 for b in Bs if b_band(b) == band) for band in ("0", "1-9", "10-99", "100-999", "1000+")},
            "minutes_given": len(mins), "minutes_mean": mean(mins), "minutes_total": sum(mins) if mins else None,
            "rows_with_refinements": sum(1 for x in col(f"refinements_{tag}") if x),
            "rows_with_not_expressible": sum(1 for x in col(f"not_expressible_{tag}") if x),
        }
        if qrels:
            summary["per_writer"][tag].update({"mean_set_precision": mean(col(f"P_{tag}")), "mean_set_recall": mean(col(f"R_{tag}")),
                                               "mean_set_F1": mean(col(f"F1_{tag}"))})
    if qrels:
        summary["A_or_B"] = {"mean_set_precision": mean(col("P_A_or_B")), "mean_set_recall": mean(col("R_A_or_B")),
                             "mean_set_F1": mean(col("F1_A_or_B"))}

    suffix = f"_{args.name}" if args.name else ""
    out = work / "eval"
    write_json(out / f"writer_agreement{suffix}.json", {"summary": summary, "per_query": rows})
    with open(out / f"writer_agreement{suffix}_per_query.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["query_id"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    md = render(summary, bool(qrels))
    (out / f"writer_agreement{suffix}.md").write_text(md, encoding="utf-8")
    print(md)
    if args.tex:
        write_tex(Path(args.tex), summary, queries, A, B, Path(args.writers) if args.writers else None)


def writer_macros(summary: dict, queries: list[dict], A: str, B: str, writers: Path | None) -> dict:
    """The paper's numbers about the two writers, named writerA... and writerB... (A = the first author id)."""
    m = {}
    p, st = summary["pages"], summary["sets"]
    for tag, aid in (("A", A), ("B", B)):
        w = summary["per_writer"][tag]
        fs = [f for q in queries for f in (q.get("boolean_formulas") or []) if f.get("author_id") == aid]
        kinds = [{e.get("alias_kind") for e in (f.get("entities") or {}).values()} for f in fs]
        m.update({
            f"writer{tag}Formulas": len(fs),
            f"writer{tag}PagesMean": f"{w['mean_pages']:.1f}",
            f"writer{tag}ConceptsMean": f"{w['mean_and_groups']:.1f}",
            f"writer{tag}WithNot": w["formulas_with_NOT"],
            f"writer{tag}EmptySets": w["empty_sets"],
            f"writer{tag}MaxB": f"{w['max_B']:,}",
            f"writer{tag}MinutesMean": f"{w['minutes_mean']:.1f}" if w["minutes_mean"] is not None else "n/a",
            f"writer{tag}Hours": f"{w['minutes_total'] / 60:.1f}",
            f"writer{tag}Refinements": w["rows_with_refinements"],
            f"writer{tag}NotExpressible": w["rows_with_not_expressible"],
            f"writer{tag}OlderNames": sum(1 for k in kinds if "older name" in k),
            f"writer{tag}TypedOtherName": sum(1 for k in kinds if "as written" in k),
        })
        for band, n in w["B_bands"].items():
            m[f"writer{tag}Sets" + {"0": "Zero", "1-9": "OneToNine", "10-99": "TenToNinetyNine", "100-999": "HundredPlus", "1000+": "ThousandPlus"}[band]] = n
    m.update({
        "writersQueriesBoth": summary["queries_both_wrote"],
        "writersPagesJaccard": f"{p['mean_jaccard']:.2f}", "writersPagesOverlap": f"{p['mean_sk_overlap']:.2f}",
        "writersSamePages": p["queries_same_pages"], "writersNoPageShared": p["queries_no_page_shared"],
        "writersSetsBothEmpty": st["cases"]["both empty"], "writersSetsIdentical": st["cases"]["identical sets"],
        "writersSetsAtLeastOne": sum(n for case, n in st["cases"].items() if case != "both empty"),
        "writersSetsJaccard": f"{st['mean_jaccard_at_least_one_found']:.2f}", "writersSetsOverlap": f"{st['mean_sk_overlap']:.2f}",
    })
    if writers is not None:
        import csv as _csv
        flagged = {A: 0, B: 0}
        for f in sorted(writers.glob("titoli_da_rivedere*.csv")):
            with open(f, encoding="utf-8-sig", newline="") as fh:
                for r in _csv.DictReader(fh):
                    if r.get("author_id") in flagged:
                        flagged[r["author_id"]] += 1
        slips, andor = {A: 0, B: 0}, {A: 0, B: 0}
        log = writers / "repair_log.csv"
        if log.exists():
            with open(log, encoding="utf-8-sig", newline="") as fh:
                for r in _csv.DictReader(fh):
                    wr = r.get("writer")
                    if wr not in slips:
                        continue
                    if r.get("rule", "").startswith("typing slip"):
                        slips[wr] += 1
                    else:
                        andor[wr] += 1
        for tag, aid in (("A", A), ("B", B)):
            m[f"writer{tag}FlaggedTitles"] = flagged[aid]
            m[f"writer{tag}TypingSlips"] = slips[aid]
            m[f"writer{tag}AndOrRepairs"] = andor[aid]
    return m


def write_tex(path: Path, summary: dict, queries: list[dict], A: str, B: str, writers: Path | None) -> None:
    m = writer_macros(summary, queries, A, B, writers)
    lines = ["% generated by paper-code/eval/writer_agreement.py --tex; do not edit by hand",
             f"% writer A = {A}, writer B = {B}"] + [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(m)} macros -> {path}")


def render(s: dict, with_qrels: bool) -> str:
    p, st, w = s["pages"], s["sets"], s["per_writer"]
    L = ["# Agreement between the two formula writers", "",
         f"Writer A = {s['authors']['A']}, writer B = {s['authors']['B']}. "
         f"Queries in the file: {s['queries_in_file']}; written by both: {s['queries_both_wrote']} "
         f"(only A: {s['queries_only_A_wrote']}, only B: {s['queries_only_B_wrote']}, neither: {s['queries_neither_wrote']}). "
         "Only the queries written by both count below.", "",
         "Counts of the archive: never shown to the writers.", "",
         "## 1. Did they choose the same Wikipedia pages?", "",
         "| measure | value |", "|---|---:|",
         f"| mean Jaccard of the two page sets (shared / either) | {p['mean_jaccard']} |",
         f"| median Jaccard | {p['median_jaccard']} |",
         f"| mean S&K overlap (shared / one writer's pages, both sides) | {p['mean_sk_overlap']} |",
         f"| Saracevic and Kantor 1988, search terms (S&K overlap) | {SK_TERMS} |",
         f"| queries where no page is shared | {p['queries_no_page_shared']} |",
         f"| queries with exactly the same pages | {p['queries_same_pages']} |", "",
         "## 2. Did their formulas find the same fact-checks?", "",
         "| case | queries |", "|---|---:|"]
    L += [f"| {c} | {n} |" for c, n in st["cases"].items()]
    L += ["", "| measure | value |", "|---|---:|",
          f"| mean Jaccard of the two result sets, queries where at least one found something | {st['mean_jaccard_at_least_one_found']} |",
          f"| median Jaccard, same queries | {st['median_jaccard_at_least_one_found']} |",
          f"| mean Jaccard, queries where both found something | {st['mean_jaccard_both_found']} |",
          f"| mean S&K overlap (shared / one writer's set, both sides, non-empty side only) | {st['mean_sk_overlap']} |",
          f"| Saracevic and Kantor 1988, retrieved items (S&K overlap) | {SK_ITEMS} |",
          f"| share of the variation of set overlap explained by page overlap (r squared; S&K found 0.025) | {st['r2_page_overlap_explains_set_overlap']} |",
          "", "## 3. Each writer", "", "| | writer A | writer B |", "|---|---:|---:|"]
    keys = [("mean_pages", "pages per formula (mean)"), ("mean_and_groups", "concepts joined by AND (mean)"),
            ("formulas_with_NOT", "formulas using NOT"), ("empty_sets", "formulas that find nothing"),
            ("median_B", "fact-checks found, median"), ("max_B", "fact-checks found, largest"),
            ("minutes_given", "rows with minutes"), ("minutes_mean", "minutes per formula (mean)"),
            ("minutes_total", "minutes in all"), ("rows_with_refinements", "rows with refinements"),
            ("rows_with_not_expressible", "rows with a not-expressible note")]
    if with_qrels:
        keys += [("mean_set_precision", "set precision (mean)"), ("mean_set_recall", "set recall against the pool (mean)"),
                 ("mean_set_F1", "set F1 (mean)")]
    L += [f"| {label} | {w['A'].get(k)} | {w['B'].get(k)} |" for k, label in keys]
    L += [f"| sets of size {band} | {w['A']['B_bands'][band]} | {w['B']['B_bands'][band]} |" for band in w["A"]["B_bands"]]
    if with_qrels:
        a = s["A_or_B"]
        L += ["", f"A OR B (the two formulas together): set precision {a['mean_set_precision']}, recall {a['mean_set_recall']}, F1 {a['mean_set_F1']}."]
    L += ["", "Means skip the queries where a measure cannot be computed (0 / 0): two empty sets are not agreement.",
          "Per query: writer_agreement_per_query.csv (numbers only)."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
