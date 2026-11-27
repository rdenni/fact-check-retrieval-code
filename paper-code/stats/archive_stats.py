"""
archive_stats.py -- the numbers about the archive that the paper and the query
design need.

From the store and WORK/corpus.jsonl:
  - fact-checks per year and per year-month of publication (min, median, max year);
  - fact-checks per publisher (schema:author name), top 50, and a template
    CSV publishers_to_map.csv (publisher, count, country) to fill by hand;
  - with --country-map: fact-checks per country and the (country x year)
    table that the query collection follows (one cell = one country-year);
  - publisher country: every predicate attached to the publisher nodes is
    listed with its count, so we SEE whether a country predicate exists
    (CimpleKG's paper has a country table, the code never used one); if
    --country-map CSV (publisher,country) is given it is applied;
  - entity coverage: documents with at least one entity on the Claim node,
    on the ClaimReview node, on both, on neither; number of entities per
    claim (median, mean);
  - duplicate claim texts (exact match after lower-casing and whitespace
    normalisation): how many documents share a text with another one;
  - if WORK/runs/graph_sets.jsonl exists: the distribution of B.
  The archive after the language rule (version 2026-10-07):
  - with --country-map also the (country x month) table (last 24 months in the
    report, every month in country_month.csv), per_month.csv, country_year.csv;
  - publishers.csv: every publisher with its country, documents, first and
    last month of publication, documents without a date, documents with an
    explanation text (which feeds are alive and which stopped);
  - documents without a date, by publisher (they stay in the corpus: the date
    only places a document on the query grid, retrieval never uses it);
  - ClaimReview-node entities among the documents that have any: median,
    mean, maximum, and how many documents have at least 2 (an AND of two
    entities needs that) and at least 5;
  - documents sharing a Claim node (the same claim reviewed several times);
  - the language check with the functions of snapshot/language_rule.py:
    claims and explanations in a non-Latin script, mojibake; all should be 0 on a
    store built with the rule; and, if language_decisions.jsonl lies next to
    the store, what the rule removed, by reason, language and publisher.
Outputs: WORK/stats/archive_stats.md and .json, plus the CSV files above

Usage: python stats/archive_stats.py --store STORE --work WORK [--country-map publishers.csv]
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import kg  # noqa: E402
from common.jsonl import read_jsonl, write_json  # noqa: E402
from snapshot import language_rule  # noqa: E402


def month_key(d: dict) -> str:
    ym = kg.date_key(d.get("date_published"))[:7]
    return ym if len(ym) == 7 and ym[:4].isdigit() and ym[5:7].isdigit() else "missing"


def write_csv(path: Path, header: list[str], rows) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--country-map", default=None)
    args = ap.parse_args()
    work = Path(args.work)
    out = work / "stats"
    out.mkdir(parents=True, exist_ok=True)
    corpus = kg.load_corpus(work / "corpus.jsonl")
    store = kg.open_store(args.store)
    rep = {"n_documents": len(corpus)}

    years = Counter()
    for d in corpus:
        y = kg.date_key(d.get("date_published"))[:4]
        years[y if y.isdigit() else "missing"] += 1
    rep["per_year"] = dict(sorted(years.items()))
    months = Counter(month_key(d) for d in corpus)
    rep["per_year_month"] = dict(sorted(months.items()))
    write_csv(out / "per_month.csv", ["month", "fact_checks"], sorted(months.items()))
    undated = Counter(d.get("author_name") or "missing" for d in corpus if month_key(d) == "missing")
    rep["undated_by_publisher"] = undated.most_common()
    ys = sorted(int(y) for y in years if y.isdigit() for _ in range(years[y]))
    rep["year_min_median_max"] = [ys[0], ys[len(ys) // 2], ys[-1]] if ys else None

    pubs = Counter(d.get("author_name") or "missing" for d in corpus)
    rep["n_publishers"] = len(pubs)
    rep["top_publishers"] = pubs.most_common(50)

    # Predicates on publisher nodes.
    q = f"""SELECT ?p (COUNT(*) AS ?n) WHERE {{ ?cr <{kg.P_TYPE}> <{kg.C_CLAIMREVIEW}> ; <{kg.P_AUTHOR}> ?org . ?org ?p ?o }} GROUP BY ?p ORDER BY DESC(?n)"""
    rep["publisher_predicates"] = [(r["p"].value, int(r["n"].value)) for r in store.query(q)]
    country_like = [p for p, _ in rep["publisher_predicates"] if re.search(r"country|location|address|nationality", p, re.I)]
    rep["country_like_predicates"] = country_like
    # Template for the hand-made publisher -> country map (fill the country column, then pass --country-map).
    with open(out / "publishers_to_map.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["publisher", "fact_checks", "country"])
        for name, n in pubs.most_common():
            w.writerow([name, n, ""])
    countries = Counter()
    country_year = Counter()
    country_month = Counter()
    cmap = {}
    if args.country_map:
        with open(args.country_map, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if (row.get("country") or "").strip():
                    cmap[row["publisher"]] = row["country"].strip()
        for d in corpus:
            c = cmap.get(d.get("author_name"), "unmapped")
            countries[c] += 1
            y = kg.date_key(d.get("date_published"))[:4]
            country_year[(c, y if y.isdigit() else "missing")] += 1
            country_month[(c, month_key(d))] += 1
        rep["per_country"] = countries.most_common()
        rep["country_year"] = {f"{c}|{y}": n for (c, y), n in sorted(country_year.items())}
        write_csv(out / "country_year.csv", ["country", "year", "fact_checks"], [(c, y, n) for (c, y), n in sorted(country_year.items())])
        write_csv(out / "country_month.csv", ["country", "month", "fact_checks"], [(c, m, n) for (c, m), n in sorted(country_month.items())])
    # Every publisher: country, documents, first and last month, undated, with explanation (which feeds are alive).
    first_last: dict[str, list] = {}
    for d in corpus:
        p = d.get("author_name") or "missing"
        m = month_key(d)
        fl = first_last.setdefault(p, ["", "", 0, 0])
        if m != "missing":
            fl[0] = min(fl[0] or m, m)
            fl[1] = max(fl[1], m)
        else:
            fl[2] += 1
        fl[3] += 1 if d.get("article_text") else 0
    pub_rows = [(p, cmap.get(p, ""), n, first_last[p][0], first_last[p][1], first_last[p][2], first_last[p][3]) for p, n in pubs.most_common()]
    write_csv(out / "publishers.csv", ["publisher", "country", "fact_checks", "first_month", "last_month", "undated", "with_article"], pub_rows)
    rep["publishers"] = pub_rows

    for node in kg.NODE_CHOICES:
        p = work / f"entities_{node}.jsonl"
        if not p.exists():
            kg.build_entity_index(store, node, p)
    claim_e = kg.load_entity_index(work / "entities_claim.jsonl")
    cr_e = kg.load_entity_index(work / "entities_cr.jsonl")
    ids = [d["cr_iri"] for d in corpus]
    both = sum(1 for i in ids if i in claim_e and i in cr_e)
    only_claim = sum(1 for i in ids if i in claim_e and i not in cr_e)
    only_cr = sum(1 for i in ids if i not in claim_e and i in cr_e)
    neither = len(ids) - both - only_claim - only_cr
    rep["entity_coverage"] = {"claim_node_any": both + only_claim, "cr_node_any": both + only_cr, "both": both,
                              "only_claim": only_claim, "only_cr": only_cr, "neither": neither}
    # The most frequent entities on each node. Reason: on 2026-09-25 the live graph linked EVERY Claim node to
    # the single entity 'Mars' (95,057 English claims, no other entity), a bug in CimpleKG's claim-level linking;
    # this table shows such a thing at once instead of hiding it inside a coverage percentage.
    top_claim = Counter(e for i in ids for e in claim_e.get(i, ())).most_common(5)
    top_cr = Counter(e for i in ids for e in cr_e.get(i, ())).most_common(5)
    rep["top_entities"] = {"claim_node": top_claim, "cr_node": top_cr}
    n_claim_links = sum(len(claim_e.get(i, ())) for i in ids)
    rep["claim_node_top_entity_share"] = round(top_claim[0][1] / n_claim_links, 4) if n_claim_links else None
    counts = [len(claim_e[i]) for i in ids if i in claim_e]
    rep["entities_per_claim"] = {"median": statistics.median(counts) if counts else None, "mean": round(statistics.mean(counts), 2) if counts else None, "max": max(counts) if counts else None}
    cr_counts = [len(cr_e[i]) for i in ids if i in cr_e]
    rep["entities_per_cr"] = {"documents_with_any": len(cr_counts), "median": statistics.median(cr_counts) if cr_counts else None,
                              "mean": round(statistics.mean(cr_counts), 2) if cr_counts else None, "max": max(cr_counts) if cr_counts else None,
                              "at_least_2": sum(1 for c in cr_counts if c >= 2), "at_least_5": sum(1 for c in cr_counts if c >= 5)}
    rep["article_text_present"] = sum(1 for d in corpus if d.get("article_text"))
    # The language check, with the same functions as the rule that built the store (all zero on such a store).
    nl = language_rule.NON_LATIN_MAX
    rep["language_check"] = {"claims_non_latin": sum(1 for d in corpus if (language_rule.non_latin_share(d["claim_text"]) or 0) > nl),
                             "articles_non_latin": sum(1 for d in corpus if (language_rule.non_latin_share(d.get("article_text")) or 0) > nl),
                             "mojibake": sum(1 for d in corpus if max(language_rule.mojibake_share(d["claim_text"]),
                                                                       language_rule.mojibake_share((d.get("article_text") or "")[:language_rule.DETECT_CHARS])) > language_rule.MOJIBAKE_MAX)}
    decisions_path = Path(args.store).resolve().parent / "language_decisions.jsonl"
    if decisions_path.exists():
        rep["language_rule"] = language_rule.summarise(list(read_jsonl(decisions_path)))
    norm = Counter(re.sub(r"\s+", " ", d["claim_text"].lower()).strip() for d in corpus)
    rep["duplicate_claim_texts"] = {"documents_sharing_a_text": sum(n for n in norm.values() if n > 1), "distinct_shared_texts": sum(1 for n in norm.values() if n > 1),
                                    "largest_group": max(norm.values()) if norm else 0}
    shared_claim = Counter(d["claim_iri"] for d in corpus)
    rep["shared_claim_nodes"] = {"documents_sharing_a_claim_node": sum(n for n in shared_claim.values() if n > 1),
                                 "distinct_shared_claim_nodes": sum(1 for n in shared_claim.values() if n > 1), "largest_group": max(shared_claim.values()) if shared_claim else 0}
    p = work / "runs" / "graph_sets.jsonl"
    if p.exists():
        Bs = [r["B"] for r in read_jsonl(p)]
        rep["B"] = {"queries": len(Bs), "zero": sum(1 for b in Bs if b == 0), "median": statistics.median(Bs) if Bs else None, "max": max(Bs) if Bs else None,
                    "over_1000": sum(1 for b in Bs if b > 1000)}
    write_json(out / "archive_stats.json", rep)
    md = ["# Archive statistics", "", f"documents: {rep['n_documents']}; years min/median/max: {rep['year_min_median_max']}", "",
          "| year | fact-checks |", "|---|---:|"] + [f"| {y} | {n} |" for y, n in rep["per_year"].items()]
    md += ["", "| year-month | fact-checks |", "|---|---:|"] + [f"| {ym} | {n} |" for ym, n in rep["per_year_month"].items()]
    md += ["", f"publishers: {rep['n_publishers']} (all of them, with counts, in publishers_to_map.csv: fill the country column by hand, then rerun with --country-map); top 20:", "", "| publisher | fact-checks |", "|---|---:|"] + [f"| {p} | {n} |" for p, n in rep["top_publishers"][:20]]
    md += ["", "predicates on publisher nodes (is there a country?): " + ", ".join(f"{p.split('/')[-1].split('#')[-1]} ({n})" for p, n in rep["publisher_predicates"][:15]),
           f"country-like predicates found: {country_like or 'none -> use --country-map'}"]
    if "per_country" in rep:
        md += ["", "| country | fact-checks |", "|---|---:|"] + [f"| {c} | {n} |" for c, n in rep["per_country"]]
        yrs = sorted({y for (c, y) in country_year})
        md += ["", "country x year (the cells the query collection follows; a cell with many fact-checks needs more queries):", "",
               "| country | " + " | ".join(yrs) + " | total |", "|---|" + "---:|" * (len(yrs) + 1)]
        for c, _ in rep["per_country"]:
            md.append(f"| {c} | " + " | ".join(str(country_year.get((c, y), 0)) for y in yrs) + f" | {countries[c]} |")
        last_months = sorted({m for (_, m) in country_month if m != "missing"})[-24:]
        cs = [c for c, _ in rep["per_country"]]
        md += ["", "country x month, last 24 months (every month in country_month.csv):", "",
               "| month | " + " | ".join(cs) + " | total |", "|---|" + "---:|" * (len(cs) + 1)]
        for m in last_months:
            md.append(f"| {m} | " + " | ".join(str(country_month.get((c, m), 0)) for c in cs) + f" | {months[m]} |")
    md += ["", "publishers: country, fact-checks, first and last month, without a date, with an explanation (publishers.csv has all of them):", "",
           "| publisher | country | fact-checks | first | last | undated | with explanation |", "|---|---|---:|---|---|---:|---:|"]
    md += [f"| {p} | {c} | {n} | {f} | {l} | {u} | {a} |" for p, c, n, f, l, u, a in rep["publishers"][:50]]
    md += ["", f"documents without a date: {sum(n for _, n in rep['undated_by_publisher'])} (they stay in the corpus; the date only places a document on the query grid): "
           + ", ".join(f"{p} {n}" for p, n in rep["undated_by_publisher"][:8])]
    ec = rep["entity_coverage"]
    md += ["", f"entity coverage: Claim node has entities for {ec['claim_node_any']} documents, ClaimReview node for {ec['cr_node_any']}, both {ec['both']}, only Claim {ec['only_claim']}, only ClaimReview {ec['only_cr']}, neither {ec['neither']}",
           f"entities per claim (documents with any): median {rep['entities_per_claim']['median']}, mean {rep['entities_per_claim']['mean']}, max {rep['entities_per_claim']['max']}",
           "top entities on the Claim node: " + ", ".join(f"{e.rsplit('/', 1)[-1]} ({n})" for e, n in rep["top_entities"]["claim_node"]) +
           f" (share of the top one among all Claim-node links: {rep['claim_node_top_entity_share']}; if it is close to 1, the Claim-node entities are unusable)",
           "top entities on the ClaimReview node: " + ", ".join(f"{e.rsplit('/', 1)[-1]} ({n})" for e, n in rep["top_entities"]["cr_node"]),
           f"entities per document on the ClaimReview node (documents with any): {rep['entities_per_cr']}",
           f"duplicate claim texts: {rep['duplicate_claim_texts']}",
           f"documents sharing a Claim node (the same claim reviewed more than once): {rep['shared_claim_nodes']}",
           f"explanation text present: {rep['article_text_present']} of {rep['n_documents']}",
           f"language check (snapshot/language_rule.py, all zero on a store built with the rule): claims in a non-Latin script {rep['language_check']['claims_non_latin']}, "
           f"explanations {rep['language_check']['articles_non_latin']}, mojibake {rep['language_check']['mojibake']}"]
    if "language_rule" in rep:
        lr = rep["language_rule"]
        md += [f"language rule {lr['rule_version']} applied when the store was built: {lr['not_english']} of {lr['documents']} fact-checks removed as not English; "
               f"by reason {lr['by_reason']}; by language {lr['by_language']}; publisher languages {lr['publisher_languages']}; "
               "by publisher " + ", ".join(f"{p} {n}" for p, n in list(lr["by_publisher"].items())[:10])]
    if "B" in rep:
        md += ["", f"B (graph set size): {rep['B']}"]
    (out / "archive_stats.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
