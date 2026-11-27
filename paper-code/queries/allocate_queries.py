"""
allocate_queries.py -- how many queries each (publisher country, year) cell of
the archive gets: the sampling design of the query set, computed from the
archive statistics and written as a table, never typed.

The design (the paper, Section 5):
  - There is no query log, so the archive's own distribution over publisher
    country and year is the proxy for the population of its users' needs
    (Sanderson 2010, section 3.6.2), and the query set is stratified by it.
  - Allocation is PROPORTIONAL to the number of documents of each stratum
    (Cochran 1977, chapter 5: proportional allocation estimates the
    population quantity, here the effectiveness for a typical need), with a
    FLOOR per country so that no country of the archive is absent: every country first receives the floor
    and only the rest of the total is split in proportion to the documents, so the allocation departs from a
    strictly proportional one; the macros allocAwayN and allocAwayPct count the queries concerned, allocDiffOne,
    allocDiffTwo and allocDiffRest give the difference in percentage points between a country's share of the
    queries and its share of the documents (for the two countries with the most documents and the largest of
    the others), and the paper states it.
  - Inside a country the queries are spread over its years in proportion to
    its documents of each year (largest-remainder rounding, so the totals hold).
  - Countries with fewer than --min-docs documents and years before --since
    get no cell of their own (there are no reports to draw from and the cells
    would be empty); their documents stay in the collection and can be relevant.

Outputs: --out-md (a Markdown table) and --out-tex (a LaTeX tabular
for the paper, latex/figures/allocation.tex), plus a JSON with the cells.

Usage:
  python queries/allocate_queries.py --stats WORK/stats --total 200 --floor 3 --since 2016 --min-docs 50 --out-md query-allocation.md --out-tex allocation.tex
         [--cap Ghana=3]   (a country with fewer usable candidates than queries keeps what it has; the rest is split again: Cochran's procedure)
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def largest_remainder(weights: dict[str, float], total: int, floor: dict[str, int] | None = None) -> dict[str, int]:
    """Split `total` units over keys in proportion to weights, after giving each key its floor; the fractional
    parts are rounded so that the sum is exactly `total` (largest remainder first)."""
    floor = floor or {}
    base = {k: floor.get(k, 0) for k in weights}
    rest = total - sum(base.values())
    if rest < 0:
        raise SystemExit(f"floors ({sum(base.values())}) exceed the total ({total})")
    wsum = sum(weights.values()) or 1.0
    share = {k: rest * w / wsum for k, w in weights.items()}
    alloc = {k: base[k] + int(share[k]) for k in weights}
    for k in sorted(weights, key=lambda k: -(share[k] - int(share[k])))[: rest - sum(int(share[k]) for k in weights)]:
        alloc[k] += 1
    return alloc


def capped(weights: dict[str, float], total: int, floor: int, caps: dict[str, int]) -> dict[str, int]:
    """Cochran's procedure for a stratum that cannot supply its allocation (the allocation would give n_h > N_h): the
    stratum keeps what it can supply (its cap) and the rest of the total is split again over the other strata by the
    same rule (proportional, with the floor, largest remainder); repeated until no stratum exceeds its cap. Described in
    Mergerson 1988 (NASS staff report SSB-88-10, Appendix A, "Cochran's Procedure"); used here after the review, when a
    country turned out to have fewer usable candidates than queries."""
    fixed: dict[str, int] = {}
    while True:
        free = [k for k in weights if k not in fixed]
        alloc = largest_remainder({k: weights[k] for k in free}, total - sum(fixed.values()), {k: floor for k in free})
        over = {k: caps[k] for k in free if k in caps and alloc[k] > caps[k]}
        if not over:
            return {**alloc, **fixed}
        fixed.update(over)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stats", required=True, help="WORK/stats of archive_stats.py (needs country_year.csv)")
    ap.add_argument("--total", type=int, default=200)
    ap.add_argument("--floor", type=int, default=3, help="minimum queries per country")
    ap.add_argument("--since", type=int, default=2016, help="first year with a cell")
    ap.add_argument("--min-docs", type=int, default=50, help="a country needs this many documents (since --since) to get cells")
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--out-tex", default=None)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--cap", action="append", default=[], metavar="COUNTRY=N",
                    help="a country that can supply only N queries (its usable candidates after the review); the rest is split again over the others (Cochran's procedure)")
    args = ap.parse_args()
    cy: dict[str, dict[str, int]] = defaultdict(dict)
    with open(Path(args.stats) / "country_year.csv", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["year"].isdigit() and int(r["year"]) >= args.since:
                cy[r["country"]][r["year"]] = int(r["fact_checks"])
    tot = {c: sum(v.values()) for c, v in cy.items()}
    countries = [c for c in sorted(tot, key=lambda c: -tot[c]) if tot[c] >= args.min_docs]
    n_docs = sum(tot[c] for c in countries)
    caps = {c.split("=")[0]: int(c.split("=")[1]) for c in args.cap}
    per_country = capped({c: float(tot[c]) for c in countries}, args.total, args.floor, caps)
    per_country = {c: per_country[c] for c in countries}
    # the queries that sit in another country than a split in strict proportion to the documents would give them
    # (the sum of the excesses over the strictly proportional numbers, which are not whole numbers)
    away = sum(max(0.0, per_country[c] - args.total * tot[c] / n_docs) for c in countries)
    # the difference in percentage points between each country's share of the queries and its share of the documents;
    # every query counts the same in the mean, so the share of the queries is the weight of the country in the mean;
    # `countries` is sorted by documents, so the first two are the two countries with the most documents
    diffs = [abs(100 * per_country[c] / args.total - 100 * tot[c] / n_docs) for c in countries]
    cells: dict[str, dict[str, int]] = {}
    for c in countries:
        cells[c] = {y: n for y, n in largest_remainder({y: float(v) for y, v in cy[c].items()}, per_country[c]).items() if n}
    years = sorted({y for c in countries for y in cy[c]})
    # Markdown
    cap_note = ("; capped (Cochran's procedure): " + ", ".join(f"{c} {n}" for c, n in sorted(caps.items()))) if caps else ""
    md = [f"# Query allocation: {args.total} queries, floor {args.floor} per country, years {args.since}+ (documents {n_docs:,} in {len(countries)} countries){cap_note}", "",
          "| country | documents | share | queries | " + " | ".join(years) + " |", "|---|---:|---:|---:|" + "---:|" * len(years)]
    for c in countries:
        md.append(f"| {c} | {tot[c]:,} | {100 * tot[c] / n_docs:.1f}% | {per_country[c]} | " + " | ".join(str(cells[c].get(y, 0) or "") for y in years) + " |")
    md.append(f"| total | {n_docs:,} | 100% | {args.total} | " + " | ".join(str(sum(cells[c].get(y, 0) for c in countries) or "") for y in years) + " |")
    text = "\n".join(md) + "\n"
    print(text)
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(text, encoding="utf-8")
    # LaTeX
    if args.out_tex:
        tex = ["% generated by paper-code/queries/allocate_queries.py; do not edit by hand",
               f"% total {args.total}, floor {args.floor}, since {args.since}, min-docs {args.min_docs}{cap_note}",
               "\\begin{tabular}{lrrr" + "r" * len(years) + "}", "\\toprule",
               "country & documents & share & queries & " + " & ".join(years) + " \\\\", "\\midrule"]
        for c in countries:
            tex.append(f"{c} & {tot[c]:,} & {100 * tot[c] / n_docs:.1f}\\,\\% & {per_country[c]} & " + " & ".join(str(cells[c].get(y, 0) or "--") for y in years) + " \\\\")
            if c != countries[-1]:
                tex.append("\\midrule[0.2pt]")  # a thin rule between two rows
        tex += ["\\midrule", f"total & {n_docs:,} & 100\\,\\% & {args.total} & " + " & ".join(str(sum(cells[c].get(y, 0) for c in countries)) for y in years) + " \\\\",
                "\\bottomrule", "\\end{tabular}", ""]
        Path(args.out_tex).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_tex).write_text("\n".join(tex), encoding="utf-8")
        macros = [f"\\newcommand{{\\allocTotal}}{{{args.total}}}", f"\\newcommand{{\\allocFloor}}{{{args.floor}}}", f"\\newcommand{{\\allocSince}}{{{args.since}}}",
                  f"\\newcommand{{\\allocMinDocs}}{{{args.min_docs}}}", f"\\newcommand{{\\allocCountries}}{{{len(countries)}}}", f"\\newcommand{{\\allocDocs}}{{{n_docs:,}}}",
                  f"\\newcommand{{\\allocAwayN}}{{{round(away)}}}", f"\\newcommand{{\\allocAwayPct}}{{{100 * away / args.total:.1f}}}",
                  f"\\newcommand{{\\allocDiffOne}}{{{(diffs[0] if diffs else 0.0):.1f}}}", f"\\newcommand{{\\allocDiffTwo}}{{{(diffs[1] if len(diffs) > 1 else 0.0):.1f}}}",
                  f"\\newcommand{{\\allocDiffRest}}{{{max(diffs[2:], default=0.0):.1f}}}",
                  "\\newcommand{\\allocCaps}{" + (", ".join(f"{c} ({n})" for c, n in sorted(caps.items())) or "none") + "}"]
        Path(args.out_tex).with_name("allocation_numbers.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")
    if args.out_json:
        Path(args.out_json).write_text(json.dumps({"total": args.total, "floor": args.floor, "since": args.since, "min_docs": args.min_docs, "caps": caps,
                                                   "documents": n_docs, "per_country": per_country, "cells": cells}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
