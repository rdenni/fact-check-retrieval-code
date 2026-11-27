"""
prepare_cells.py -- lay the report sources out per (country, year) cell so that
the drafting step (an LLM extracting information needs, then a random draw)
reads exactly the tier-1 material of one cell, with its provenance, and
nothing else.

For every cell of the allocation (latex/figures/allocation.json) it gathers
the saved documents of report_sources.csv whose years cover the cell's year
and whose tier is in --tiers (default: 1 only), plus the EDMO briefs of that
year for France and Ireland, plus the Africa Center regional mapping for the
African cells it covers, and writes:

  WORK/cells/<folder>-<year>.md     one file per cell: a header per document
                                    (label, tier, publisher type, url, sha256)
                                    followed by its text
  WORK/cells/index.json             per cell: queries, documents, characters
  WORK/cells/coverage.md            the coverage table (which cells have how
                                    many tier-1 documents)
  latex/figures/sources_numbers.tex the counts the paper quotes about the
                                    sources (rows per tier, Freedom on the Net
                                    and EDMO counts, cells, covered queries),
                                    as LaTeX macros; --numbers-out changes the
                                    path, --no-numbers skips it

Long reports are cut to the parts that describe narratives: for the Freedom
on the Net country reports only the Overview, Key Developments and section
B5 (manipulation of online content) are kept, because the rest is legal and
infrastructure analysis; every other document is kept whole, with the
navigation lines of saved web pages dropped.

Usage:
  python queries/prepare_cells.py --out reports/cells [--tiers 1] [--list queries/report_sources.csv]
                                  [--reports reports] [--allocation latex/figures/allocation.json]
                                  [--numbers-out latex/figures/sources_numbers.tex | --no-numbers]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FOLDER = {"United States": "us", "India": "india", "United Kingdom": "uk", "Australia": "australia", "Nigeria": "nigeria",
          "Ghana": "ghana", "South Africa": "south-africa", "Ireland": "ireland", "Hong Kong": "hong-kong",
          "France": "france", "Zimbabwe": "zimbabwe"}
EDMO_COUNTRIES = {"france", "ireland"}
REGIONAL = {"africacenter-mapping-disinformation-2024": ({"south-africa", "zimbabwe", "ghana", "nigeria"}, {2022, 2023, 2024})}


def years_of(s: str) -> set[int]:
    s = s.strip()
    if "-" in s:
        a, b = s.split("-")
        return set(range(int(a), int(b) + 1))
    return {int(s)}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_text(txt: str) -> str:
    """Drop the navigation debris of saved pages: very short lines repeated across the file, menus, cookie notices."""
    lines = [l.rstrip() for l in txt.splitlines()]
    counts = defaultdict(int)
    for l in lines:
        if l.strip():
            counts[l.strip()] += 1
    out, blank = [], 0
    for l in lines:
        s = l.strip()
        if s and counts[s] > 2 and len(s) < 60:
            continue  # a menu item or footer repeated through the page
        if re.match(r"^(cookie|accept all|subscribe|sign in|share this|related articles|advertisement)", s, re.I):
            continue
        blank = blank + 1 if not s else 0
        if blank <= 1:
            out.append(s)
    return "\n".join(out).strip()


def fotn_sections(txt: str) -> str:
    """Freedom on the Net: keep Overview, Key Developments and B5; drop the rest."""
    flat = re.sub(r"[ \t]+", " ", txt)
    parts = []
    m = re.search(r"Overview\s*\n(.*?)(?=\n\s*Key Developments|\n\s*A1\b)", flat, re.S)
    if m:
        parts.append("## Overview\n" + m.group(1).strip())
    m = re.search(r"Key Developments.*?\n(.*?)(?=\n\s*A1\b|\n\s*A Obstacles)", flat, re.S)
    if m:
        parts.append("## Key Developments\n" + m.group(1).strip())
    m = re.search(r"\n\s*B5\b(.*?)(?=\n\s*B6\b)", flat, re.S)
    if m:
        parts.append("## B5 Manipulation of online content\n" + m.group(1).strip())
    return "\n\n".join(parts) if parts else clean_text(txt)


def doc_text(row: dict, reports: Path) -> tuple[str, Path | None]:
    base = reports / row["folder"] / row["label"]
    txt = base.with_suffix(".txt")
    src = next((p for p in (base.with_suffix(".pdf"), base.with_suffix(".html")) if p.exists()), None)
    if not txt.exists() or src is None:
        return "", src
    raw = txt.read_text(encoding="utf-8", errors="replace")
    if row["label"].startswith("fotn-"):
        return fotn_sections(raw), src
    return clean_text(raw), src


def edmo_briefs(reports: Path, year: int) -> list[dict]:
    idx = reports / "edmo-briefs" / "index.csv"
    if not idx.exists():
        return []
    out = []
    for r in csv.DictReader(open(idx, encoding="utf-8", newline="")):
        if not r.get("date", "").startswith(str(year)) or not r.get("local"):
            continue
        pdf = reports / "edmo-briefs" / r["local"]
        txt = pdf.with_suffix(".txt")
        if not txt.exists():
            continue
        out.append({"label": "edmo-" + r["label"], "tier": "1", "publisher_type": "research-institute", "url": r["pdf_url"],
                    "years": str(year), "folder": "edmo-briefs", "_text": clean_text(txt.read_text(encoding="utf-8", errors="replace")),
                    "_src": pdf, "note": r.get("focus", "")})
    return out


def macro_name(prefix: str, key: str) -> str:
    """'ngo-research' -> '\\nSourcesNgoResearch': LaTeX macro names allow letters only."""
    return "\\" + prefix + "".join(w[:1].upper() + w[1:] for w in re.split(r"[^A-Za-z]+", key) if w)


def write_numbers(path: Path, all_rows: list[dict], tiers: set[str], edmo_n: int, coverage: list, used_labels: set[str]) -> None:
    """The counts the paper quotes about the sources, as LaTeX macros (the paper never types a number by hand)."""
    tier_count = defaultdict(int)
    for r in all_rows:
        tier_count[r["tier"]] += 1
    strict = [r for r in all_rows if r["tier"] in tiers]
    type_count = defaultdict(int)
    for r in strict:
        type_count[r.get("publisher_type", "") or "unknown"] += 1
    total_q = sum(q for _, _, q, _, _ in coverage)
    covered = sum(q for _, _, q, n, _ in coverage if n)
    uncovered = [f"{c} {y}" for c, y, q, n, _ in coverage if not n]
    nc = "\\newcommand"
    counts = [
        ("nSourceRows", len(all_rows)),
        ("nSourcesTierOne", tier_count.get("1", 0)),
        ("nSourcesTierOneB", tier_count.get("1b", 0)),
        ("nSourcesTierTwo", tier_count.get("2", 0)),
        ("nSourcesTierJ", tier_count.get("J", 0)),
        ("nSourcesExcluded", tier_count.get("X", 0)),
        ("nSourcesUsed", len(used_labels)),
        ("nFotnReports", sum(1 for r in strict if r["label"].startswith("fotn-"))),
        ("nEdmoBriefs", edmo_n),
        ("nCells", len(coverage)),
        ("nCellsCovered", sum(1 for _, _, _, n, _ in coverage if n)),
        ("nQueriesInCells", total_q),
        ("nQueriesCovered", covered),
        ("nQueriesUncovered", total_q - covered),
        ("cellsUncovered", ", ".join(uncovered) if uncovered else "none"),
    ]
    lines = ["% Generated by paper-code/queries/prepare_cells.py from queries/report_sources.csv, reports/edmo-briefs/index.csv and the allocation. Do not edit by hand."]
    lines += [f"{nc}{{\\{name}}}{{{value}}}" for name, value in counts]
    lines += [f"{nc}{{{macro_name('nSources', k)}}}{{{v}}}" for k, v in sorted(type_count.items())]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "cells")
    ap.add_argument("--tiers", default="1", help="comma-separated tiers to include (default 1)")
    ap.add_argument("--list", type=Path, default=HERE / "report_sources.csv")
    ap.add_argument("--reports", type=Path, default=ROOT / "reports")
    ap.add_argument("--allocation", type=Path, default=ROOT / "latex" / "figures" / "allocation.json")
    ap.add_argument("--numbers-out", type=Path, default=ROOT / "latex" / "figures" / "sources_numbers.tex",
                    help="where the LaTeX macros with the source counts go")
    ap.add_argument("--no-numbers", action="store_true", help="do not write the LaTeX macros")
    args = ap.parse_args()
    if not args.reports.is_dir():
        # without the fetched reports every cell would be empty and every count 0: stop instead of writing zeros
        raise SystemExit(f"--reports: {args.reports} is not a folder; fetch the reports first (queries/fetch_pages.py, "
                         f"queries/fetch_edmo_briefs.py) or give the folder that holds them")
    tiers = {t.strip() for t in args.tiers.split(",")}
    alloc = json.load(open(args.allocation, encoding="utf-8"))["cells"]
    all_rows = list(csv.DictReader(open(args.list, encoding="utf-8", newline="")))
    rows = [r for r in all_rows if r["tier"] in tiers]
    args.out.mkdir(parents=True, exist_ok=True)
    used_labels: set[str] = set()

    index, coverage = {}, []
    for country, cells in alloc.items():
        folder = FOLDER[country]
        for year_s, q in cells.items():
            year = int(year_s)
            docs = []
            for r in rows:
                covers = r["folder"] == folder and year in years_of(r["years"])
                reg = REGIONAL.get(r["label"])
                if reg and folder in reg[0] and year in reg[1]:
                    covers = True
                if not covers:
                    continue
                text, src = doc_text(r, args.reports)
                if not text:
                    continue
                docs.append({**r, "_text": text, "_src": src})
            if folder in EDMO_COUNTRIES:
                docs.extend(edmo_briefs(args.reports, year))
            used_labels.update(d["label"] for d in docs)
            name = f"{folder}-{year}"
            with open(args.out / f"{name}.md", "w", encoding="utf-8") as f:
                f.write(f"# Cell {country} {year}: {q} queries, {len(docs)} documents (tiers {','.join(sorted(tiers))})\n\n")
                for d in docs:
                    f.write(f"\n\n---\n\n## DOCUMENT {d['label']}\n")
                    f.write(f"tier: {d['tier']} | publisher type: {d.get('publisher_type','')} | years: {d['years']} | url: {d['url']}\n")
                    f.write(f"file: {d['_src'].relative_to(args.reports).as_posix()} | sha256: {sha256_of(d['_src'])}\n")
                    if d.get("note"):
                        f.write(f"note: {d['note']}\n")
                    f.write("\n" + d["_text"] + "\n")
            index[name] = {"country": country, "year": year, "queries": q,
                           "documents": [{"label": d["label"], "tier": d["tier"], "chars": len(d["_text"])} for d in docs],
                           "chars": sum(len(d["_text"]) for d in docs)}
            coverage.append((country, year, q, len(docs), sum(len(d["_text"]) for d in docs)))
    json.dump(index, open(args.out / "index.json", "w", encoding="utf-8"), indent=1)
    covered = sum(q for _, _, q, n, _ in coverage if n)
    with open(args.out / "coverage.md", "w", encoding="utf-8") as f:
        f.write(f"# Coverage of the allocation by the saved documents of tiers {','.join(sorted(tiers))}: "
                f"{covered} of {sum(q for _, _, q, _, _ in coverage)} queries\n\n| country | year | queries | documents | characters | status |\n|---|---|---|---|---|---|\n")
        for country, year, q, n, chars in coverage:
            f.write(f"| {country} | {year} | {q} | {n} | {chars:,} | {'ok' if n else 'UNCOVERED'} |\n")
    print(f"[cells] {len(coverage)} cells, {covered} of {sum(q for _, _, q, _, _ in coverage)} queries covered; "
          f"uncovered: {[(c, y, q) for c, y, q, n, _ in coverage if not n]}")
    print(f"[cells] written to {args.out}")
    if not args.no_numbers:
        edmo_idx = args.reports / "edmo-briefs" / "index.csv"
        edmo_n = sum(1 for r in csv.DictReader(open(edmo_idx, encoding="utf-8", newline="")) if r.get("local")) if edmo_idx.exists() else 0
        write_numbers(args.numbers_out, all_rows, tiers, edmo_n, coverage, used_labels)
        print(f"[cells] LaTeX macros written to {args.numbers_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
