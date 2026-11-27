"""
extend_pool_before.py -- the pool for the tables on the documents published before a month (in the paper the month
entity linking of the explanations stopped, 2025-04).

Why. eval/metrics.py --before removes from every run the documents published from that month on, before scoring. The
top 20 of what remains then reaches below rank 20 of the whole list, where the pool (pool/build_pool.py) judged
nothing, and an unjudged document counts as not relevant: the text systems would look worse in exactly those tables.
This script applies the pool's own rule to the lists cut that way, so the before-month tables are judged as deeply as
the main ones:
  - ranks 1 to --depth (20) of every ranked system (--ranked) and of every AI-only run (--ai-only), in the cut list;
  - for the ranked systems, down to the size B of the larger writer's strict set cut the same way, when B is larger
    than --depth, as build_pool.py does on the whole lists (a cut list is shorter than 100, so it may end earlier);
  - the strict sets need nothing: cutting a set only removes members, and every member is in the pool already.
Only pairs that are not in the pool yet are added. They are labelled by the AI judges only: they are never in the human
check sample, and check.json, shared.json, assignments.json and pool_summary.md are not touched.

The date rule is the one of eval/metrics.py --before: a document is kept if it is undated or published before the
month (its date read by systems/graph_boolean.sortable_date).

Writes WORK/pool/pool.jsonl again: the main pool's rows first, byte for byte as they were, then the added rows. An
added row has the fields of build_pool.py (the ranks in the whole lists in "ranks" and "min_rank", "check" false, the
same texts) plus "added_for" ("before YYYY-MM") and "ranks_before" (its ranks in the cut lists). The main pool's rows
are the rows without "added_for", so running the script again, or after build_pool.py has rebuilt the main pool, gives
the same file. Before writing, every main row's texts are rebuilt from the corpus and the country map and compared with
the pool's: a different corpus, country map or --context-chars stops the script. Also writes
WORK/pool/extension_before.json (the settings, the counts, the added pairs) and extension_before.md.

Then: judge/pack_for_judge.py packs a new bundle with the whole pool; with the label files of the first run copied
into its judge/ folder, judge/run_judges.py labels only the added pairs (it skips every pair already labelled).

Usage:
  python pool/extend_pool_before.py --work WORK --before 2025-04 --ranked bm25,pipeline \
      --sets graph_annotator_a_cr,graph_annotator_b_cr \
      --ai-only relaxed_annotator_a,relaxed_annotator_b,bm25_claimart,pipeline_claimart --country-map MAP.csv
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import kg  # noqa: E402
from common.jsonl import read_jsonl, write_json  # noqa: E402
from common.trec import read_run  # noqa: E402
from pool.build_pool import load_country_map, names  # noqa: E402
from systems.graph_boolean import sortable_date  # noqa: E402

TEXT_FIELDS = ("claim_text", "article_start", "date_published", "author_name", "country", "rating")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--before", required=True, help="YYYY-MM, the month given to eval/metrics.py --before (2025-04 in the paper)")
    ap.add_argument("--ranked", default="bm25,pipeline", help="as in build_pool.py: judged to --depth and down to a large set's size")
    ap.add_argument("--sets", default="", help="as in build_pool.py: the writers' strict sets (their size B, cut, sets the depth)")
    ap.add_argument("--ai-only", default="", help="as in build_pool.py: judged to --depth")
    ap.add_argument("--depth", type=int, default=20)
    ap.add_argument("--context-chars", type=int, default=500, help="as in build_pool.py")
    ap.add_argument("--country-map", default=None, help="as in build_pool.py (the same file, or the texts check stops the script)")
    args = ap.parse_args()
    if not re.fullmatch(r"\d{4}-\d{2}", args.before):
        raise SystemExit("--before must be a month, YYYY-MM (for example 2025-04)")
    work = Path(args.work)
    pool_path = work / "pool" / "pool.jsonl"
    ranked, sets_tags, ai_only = names(args.ranked), names(args.sets), names(args.ai_only)

    # the main pool: its rows and their exact lines
    main_rows, main_lines, old_added = [], [], 0
    with open(pool_path, encoding="utf-8", newline="") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("added_for"):
                old_added += 1
                continue
            main_rows.append(r)
            main_lines.append(line if line.endswith("\n") else line + "\n")
    in_pool = {(r["query_id"], r["doc_id"]) for r in main_rows}
    qids = list(dict.fromkeys(r["query_id"] for r in main_rows))

    corpus = {d["cr_iri"]: d for d in kg.load_corpus(work / "corpus.jsonl")}
    country = load_country_map(Path(args.country_map) if args.country_map else work / "stats" / "publishers_map.csv",
                               required=bool(args.country_map))

    def texts(d: str) -> dict:  # the texts of a pool row, built as build_pool.py builds them
        doc = corpus.get(d, {})
        return {"claim_text": doc.get("claim_text", ""), "article_start": (doc.get("article_text") or "")[:args.context_chars],
                "date_published": doc.get("date_published"), "author_name": doc.get("author_name"),
                "country": country.get(doc.get("author_name") or "", ""), "rating": doc.get("rating")}

    differ = [(r["query_id"], r["doc_id"]) for r in main_rows if {k: r.get(k) for k in TEXT_FIELDS} != texts(r["doc_id"])]
    if differ:
        raise SystemExit(f"{len(differ)} rows of the main pool have other texts than this corpus, country map and "
                         f"--context-chars give (first: {differ[0]}): use the files build_pool.py used")

    dates = {k: sortable_date(d.get("date_published")) for k, d in corpus.items()}
    keep = lambda d: not dates.get(d) or dates[d][:7] < args.before  # noqa: E731  (eval/metrics.py --before)
    runs = {s: read_run(work / "runs" / f"{s}.trec") for s in ranked + ai_only}
    sets = {}
    for s in sets_tags:
        p = work / "runs" / f"{s}_sets.jsonl"
        if not p.exists():
            raise SystemExit(f"missing: {p}")
        sets[s] = {r["query_id"]: r["doc_ids"] for r in read_jsonl(p)}
        outside = [(q, d) for q, ids in sets[s].items() for d in ids if (q, d) not in in_pool]
        if outside:
            raise SystemExit(f"{len(outside)} members of the strict set {s} are not in the pool: rebuild the pool first")

    added: dict[tuple[str, str], dict[str, dict[str, int]]] = {}
    by_system, by_query, beyond = Counter(), Counter(), 0
    for qid in qids:
        big_b = max([sum(1 for d in sets[s].get(qid, []) if keep(d)) for s in sets_tags] or [0])
        for s in ranked + ai_only:
            whole = [d for d, _ in runs[s].get(qid, [])]
            cut = [d for d in whole if keep(d)]
            down_to = max(args.depth, big_b) if s in ranked else args.depth
            for r, d in enumerate(cut[:down_to], 1):
                if (qid, d) in in_pool:
                    continue
                a = added.setdefault((qid, d), {"ranks": {}, "ranks_before": {}})
                a["ranks"][s] = whole.index(d) + 1
                a["ranks_before"][s] = r
    for (qid, d), a in added.items():
        by_query[qid] += 1
        for s in a["ranks_before"]:
            by_system[s] += 1
        if all(s in ranked and r > args.depth for s, r in a["ranks_before"].items()):
            beyond += 1

    tag = f"before {args.before}"
    order = {q: i for i, q in enumerate(qids)}
    rows = []
    for (qid, d) in sorted(added, key=lambda k: (order[k[0]], k[1])):
        a = added[(qid, d)]
        rows.append({"query_id": qid, "doc_id": d, "ranks": a["ranks"], "min_rank": min(a["ranks"].values()), "check": False,
                     **texts(d), "added_for": tag, "ranks_before": a["ranks_before"]})
    new = "".join(main_lines) + "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    old = pool_path.read_bytes().decode("utf-8")  # bytes, so that a line end is compared as it is
    if new != old:
        tmp = pool_path.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            f.write(new)
        tmp.replace(pool_path)
    write_json(work / "pool" / "extension_before.json", {
        "before": args.before, "depth": args.depth, "ranked": ranked, "ai_only": ai_only, "sets": sets_tags,
        "context_chars": args.context_chars, "main_pairs": len(main_rows), "added_pairs": len(rows),
        "added_only_for_B": beyond, "added_by_system": dict(by_system), "added_by_query": dict(by_query),
        "pairs": [[r["query_id"], r["doc_id"]] for r in rows]})
    lines = ["# Pool extension for the tables before " + args.before, "",
             f"the lists cut to the documents published before {args.before} or undated (eval/metrics.py --before), "
             f"judged as the main pool is: to depth {args.depth}, and the ranked systems ({', '.join(ranked)}) down to "
             f"the size of a cut strict set larger than {args.depth}; AI-only lists: {', '.join(ai_only) or 'none'}",
             f"main pool: {len(main_rows)} pairs; added: {len(rows)} pairs over {len(by_query)} queries "
             f"({beyond} of them only because a cut set is larger than {args.depth}); pool now: {len(main_rows) + len(rows)}",
             "", "| system | added pairs it brings |", "|---|---:|"]
    lines += [f"| {s} | {by_system.get(s, 0)} |" for s in ranked + ai_only]
    (work / "pool" / "extension_before.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[2:4]))
    print(("pool.jsonl unchanged" if new == old else f"wrote {pool_path}") +
          (f" (it held {old_added} added rows before; they were recomputed)" if old_added else "") +
          ", extension_before.json, extension_before.md")


if __name__ == "__main__":
    main()
