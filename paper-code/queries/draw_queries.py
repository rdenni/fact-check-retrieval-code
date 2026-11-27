"""
draw_queries.py -- the seeded random draw of the queries from the kept
candidates, cell by cell, and the rewrite step after it.

  draw            for every (country, year) cell of the allocation, draw its
                  number of queries uniformly at random, without replacement,
                  from the kept clusters of that cell (random.Random(seed) per
                  cell, the seed written into every query); for the countries
                  in --per-country the cells are pooled and the country's
                  total is drawn; a cell
                  with fewer kept candidates than queries is reported as a
                  shortfall, never topped up by hand; with --pool-short
                  every cell first draws its own quota, then
                  the queries a short cell cannot fill are drawn at random
                  (seed:<country>:top-up) from the candidates the country's
                  other years left, and the report lists the top-ups; with --pilot N it then
                  draws the pilot set: N further queries from the kept
                  candidates that the main draw did NOT take, so the pilot
                  and the main set never share a query and the main draw is
                  the same with or without a pilot; each pilot query picks a
                  cell at random with weight = the cell's number of main
                  queries (so the pilot has the main set's country and year
                  mix), then a candidate of that cell at random
                  (random.Random(f"{seed}:pilot")); written to --pilot-out
                  with query ids pilot-01, pilot-02, ...
                  After a review in quota mode (review_needs.py)
                  only the eligible kept rows are drawn (the K rows inside
                  each cell's reviewed prefix of the random order), and the
                  draw refuses to run while review_status.json says a cell
                  must continue (--allow-unfinished overrides, for tests only)
  rewrite-sheet   a workbook with the drawn queries: query_id, cell, period and
                  verdict words (from WORK/drafting/reviewed.jsonl, with --work),
                  claim, passage, the report's file, the passage's PDF page
                  (with --reports) and link, the model's hint and doubt (with
                  --proposals, from propose_rewrites.py; the model's
                  proposal itself is not shown), reworded (X
                  when the text put into free_text differs from the claim, and
                  any problem the import's checks find in it), free_text
                  (filled with the closed model's suggestion, or the model's proposal where
                  the closed model made none, for the reviewer to accept or correct), parts (empty: for rule W4,
                  the reviewer writes "first claim || second claim" when the
                  claim joins two or more claims that one fact-check would not check
                  together), context (empty: what the reviewer copies from the
                  report about the same claim to learn who, where and when),
                  background (empty: who or what a name of the query is,
                  written from the report, shown to the judges only, never
                  searched), note, names in the passage; with --keep OLD.xlsx
                  the background and the parts,
                  free_text, context and note of every query whose claim is
                  unchanged are carried over from OLD.xlsx; with --closed-proposals FILE
                  the closed model's proposals (one JSON line per query:
                  query_id, cluster_id, free_text, context, background, parts,
                  note) fill parts, free_text, context and background of every
                  row that holds nothing but the local model's proposal; a row
                  holding more (the closed model's proposal put in before, or the
                  reviewer's work) is never overwritten by a rebuild (--keep
                  wins); the closed model's reasons stay in its file (no closed_note
                  column); reworded is X when the text put
                  into free_text differs from the claim
  import-rewrite  reads the rewrites back into the query file as free_text,
                  keeps the draft as draft_text, the local model's proposal (from
                  the sheet's proposal column, or from --proposals) as
                  provenance.rewrite_proposal and the context as
                  provenance.rewrite_context, counts the changed proposals,
                  and lists every problem of the rewrite rules W1 to W4
                  (REVIEW-RULES.md): a capitalised word that is in neither the
                  passage, the claim, the draft nor the context (the query's
                  country, its adjective or abbreviation, a month and a
                  possessive 's are allowed), a year in neither of them nor the
                  period, a verdict word or an attribution the claim does not
                  have; with --reports, a context not copied word for word from
                  the report; a background whose names or years are in neither
                  the passage, the claim, the draft nor the context, or that has
                  a verdict word (rule W5, background_problems); for a row with parts (W4) it says which part the
                  query keeps, chosen at random with the draw's seed
                  (pick_part), records parts and kept part in the provenance,
                  and warns when the text looks closer to another part; a row
                  written for another claim (a replaced query) is refused;
                  with --closed-proposals FILE it keeps the closed model's proposal beside the final
                  text (provenance.rewrite_closed) and counts how many the
                  reviewer changed
  replace         rule Replacement: replaces --query-id, whose row
                  must already be D in the review log, by a kept row drawn at
                  random (seed:replace:<query id>:<n>) among those that neither
                  --queries nor the --other files use, from the query's cell, or
                  from its country when the query filled a short year or its
                  cell has none left; when the country has none left either,
                  the query is removed (a shortfall the paper reports); the
                  other queries do not change; the old file is kept beside it
                  and every replacement or removal is logged in
                  WORK/drafting/replacements.jsonl

The query file follows the paper's format (common/queries.py, version 6):
query_id, free_text, query_type "narrative", stratum {country, year}, source
{report, url_or_doi, date}, and the provenance fields of the draw
(candidate_id, cluster_id, passage, tier, publisher_type, sha256, draw_seed,
draw_rank, mentions, topic).

Usage:
  python queries/draw_queries.py draw --work WORK --allocation latex/figures/allocation.json --seed 20261010
         [--per-country "Hong Kong,Nigeria"] [--pilot 15 --pilot-out WORK/queries_pilot.json] --out WORK/queries_drawn.json
  python queries/draw_queries.py rewrite-sheet --queries WORK/queries_drawn.json --work WORK --proposals WORK/drafting/rewrite_proposals_queries_drawn.jsonl --reports reports --out rewrite.xlsx
  python queries/draw_queries.py import-rewrite --queries WORK/queries_drawn.json --sheet rewrite.xlsx --reports reports --out WORK/queries_v7.json
  python queries/draw_queries.py replace --queries WORK/queries_drawn.json --other WORK/queries_pilot.json --work WORK --query-id us-2016-03 --out WORK/queries_drawn.json
  (rebuild a sheet keeping the work: rewrite-sheet ... --keep rewrite.xlsx --out rewrite.xlsx)
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.jsonl import read_jsonl, write_json, read_json  # noqa: E402

FOLDER = {"United States": "us", "India": "india", "United Kingdom": "uk", "Australia": "australia", "Nigeria": "nigeria",
          "Ghana": "ghana", "South Africa": "south-africa", "Ireland": "ireland", "Hong Kong": "hong-kong",
          "France": "france", "Zimbabwe": "zimbabwe"}


def to_query(c: dict, k: int, seed: int, query_id: str | None = None, query_set: str = "main") -> dict:
    cell = c["cell"]
    return {"query_id": query_id or f"{cell}-{k:02d}", "set": query_set, "free_text": c["need"], "draft_text": c["need"], "query_type": "narrative",
            "stratum": {"country": c["country"], "year": c["year"]},
            "source": {"report": c["doc"], "url_or_doi": c["url"], "date": str(c["year"])},
            "filters": {}, "entity_groups": [], "boolean_formulas": [],
            "provenance": {"cluster_id": c["cluster_id"], "candidate_id": c["representative"], "claim": c["claim"], "passage": c["passage"],
                           "entities": c["entities"], "tier": c["tier"], "publisher_type": c["publisher_type"], "file": c["file"],
                           "sha256": c["sha256"], "mentions": c["mentions"], "docs": c["docs"], "topic": c["topic"],
                           "model": c["model"], "precision": c["precision"], "prompt_sha256": c["prompt_sha256"],
                           "draw_seed": seed, "draw_rank": k}}


ADJECTIVES = {"United States": ["American", "US", "U.S.", "USA", "America"], "India": ["Indian"],
              "United Kingdom": ["British", "UK", "U.K.", "Britain"], "Australia": ["Australian"], "Nigeria": ["Nigerian"],
              "Ghana": ["Ghanaian"], "South Africa": ["South African"], "Ireland": ["Irish"], "Hong Kong": ["Hong Kong"],
              "France": ["French"], "Zimbabwe": ["Zimbabwean"]}  # the words a rewrite may add for its country
MONTHS = {"January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"}
VERDICT = re.compile(r"(?i)\b(false|falsely|fake|debunk\w*|misleading|hoax\w*|allegedly|baseless|unfounded|myth|rumou?rs?|disinformation|misinformation|fabricated|doctored|manipulated|deepfake|decontextuali[sz]ed|AI-generated)\b")
ATTRIBUTION = re.compile(r"(?i)^\s*(claims?|allegations?|rumou?rs?|reports?|posts?) (that|of)\b|\b(said|claimed|alleged|asserted|suggests?|suggested) that\b")


def capitalised(s: str) -> set[str]:
    return {w for w in re.findall(r"\b[A-Z][\w'’.-]*\w", s)}


# English words that are capitalised only because they start a sentence and are never names (the rebuilt
# sheet flagged "An" in "An image ..." because its passage has no "an")
FUNCTION = {"an", "the", "this", "that", "these", "those", "there", "it", "its", "he", "she", "they", "we", "his", "her",
            "their", "some", "many", "most", "all", "every", "no", "not", "in", "on", "at", "of", "for", "from", "by", "with",
            "after", "before", "during", "since", "if", "when", "while", "and", "but", "or", "as", "is", "are", "was", "were"}


def bare(w: str) -> str:
    """A word without quotes, final dots and a possessive ending, lower-cased: "India's" -> "india", "U.S." -> "u.s"."""
    return re.sub(r"['’]s$", "", w.strip(".'’\"“”")).lower()


# a quotation mark (the query has none, since a person who searches does not type them): every quoted
# phrase opens with ‘ “ " or a straight ' at the start or after a space or a bracket; a closing ’ alone is not counted, because
# it is also the apostrophe of "Trump’s" and "farmers’"
QUOTE = re.compile(r"[‘“”\"]|(?:^|(?<=[\s(\[]))'")
# a hyphen or a dash (a person who searches does not type them either): inside a word ("COVID-19",
# "pro-Pakistan", "9–10") or standing between spaces; a hyphen becomes a space, which BM25's tokenizer reads the same way
DASH = re.compile(r"(?<=\w)[-‐‑‒–—](?=\w)|(?:^|\s)[-‐‑‒–—](?:\s|$)")


def background_problems(background: str, q: dict, context: str = "", period: str = "") -> list[str]:
    """Rule W5: the background may be reworded, but every name and year in it must be in the
    passage, the claim, the draft or the context (the report's words it was written from), and it carries no verdict
    word, since the judges are not told what the fact-checkers decided. No attribution check: a background may say who
    said what ("a witness who testified that ..."). No quotation-mark or hyphen check either: no system searches the background."""
    return [w for w in rewrite_problems(background, q, context, period) if not w.startswith(("written as an attribution", "quotation marks", "hyphens"))]


def rewrite_problems(text: str, q: dict, context: str = "", period: str = "") -> list[str]:
    """The mechanical checks of the rewrite rules (REVIEW-RULES.md, W1 to W4): a capitalised word of the
    rewrite must be in the passage, the claim, the draft or the context copied from the report (in any case, so a
    sentence's first word is fine), or be the query's country, its adjective or abbreviation, or a month (W2: nothing
    from elsewhere); a year must be in the same texts or in the period; no verdict word or attribution that the claim
    itself does not have (W1: nothing added to what is claimed). The limit of 30 words and the check against copying the
    passage were dropped: neither had a basis, a long claim is still one claim, and many passages are the
    claim itself."""
    p = q["provenance"]
    source = p["passage"] + " " + q.get("draft_text", "") + " " + p["claim"] + " " + context
    # a hyphenated word also counts by its pieces ("Anglican-Catholic rivalry" in the report gives "Anglican"
    # and "Catholic" to "The Anglican and Catholic churches ...")
    known = {bare(x) for w in re.findall(r"\w(?:[\w'’.-]*\w)?", source) for x in [w] + w.split("-")}
    country = q["stratum"]["country"]
    allowed = {bare(w) for name in [country] + ADJECTIVES.get(country, []) for w in name.split()} | {m.lower() for m in MONTHS}
    out = []
    added = sorted({w for w in capitalised(text) if bare(w) not in known | allowed | FUNCTION})
    if added:
        out.append(f"names not in the passage or the context: {', '.join(added)}")
    year = re.compile(r"\b(?:19|20)\d\d\b")
    years = sorted(set(year.findall(text)) - set(year.findall(source + " " + period)))
    if years:
        out.append(f"years not in the passage, the context or the period: {', '.join(years)}")
    verdict = sorted({m.group(0).lower() for m in VERDICT.finditer(text)} - {m.group(0).lower() for m in VERDICT.finditer(p["claim"])})
    if verdict:
        out.append(f"verdict words: {', '.join(verdict)}")
    if ATTRIBUTION.search(text) and not ATTRIBUTION.search(p["claim"]):
        out.append("written as an attribution (\"claims that\", \"X said that\"), not as the claim itself")
    if QUOTE.search(text):
        out.append("quotation marks: a query has none, write the words without them")
    if DASH.search(text):
        out.append("hyphens or dashes: a query has none, write a space or reword ('COVID 19', 'pro Pakistan')")
    return out


def same_statements(stated: list[tuple[str, str, bool]]) -> list[str]:
    """Rules W4 and D3 (rule A): a W4 pick can leave a query stating what another query states
    ("Muslims are violent" in uk-2019-01 and india-2024-07). stated holds, in the order of the query file, (query id, what the
    query states: the kept part of a split query, else its text, was it split). Two statements that are the same after
    lowercasing and keeping only letters and digits are reported, with the query to replace (rule Replacement, as a D3): the
    split one, or the later one in the file when both or neither were split. Only the same words are found here; the reviewer
    reads the picks for claims that say the same in other words."""
    norm = lambda s: " ".join(re.findall(r"[a-z0-9]+", s.lower()))  # noqa: E731
    first, out = {}, []
    for qid, text, split in stated:
        key = norm(text)
        if not key:
            continue
        if key in first:
            a_id, a_split = first[key]
            replace = a_id if a_split and not split else qid
            out.append(f"{a_id} and {qid} state the same claim ({text}): rules W4 and D3, replace {replace} (step 8f, rule D3)")
        else:
            first[key] = (qid, split)
    return out


WORD = re.compile(r"[a-z0-9]{4,}")


def split_parts(s: str) -> list[str]:
    """The claims the reviewer separated for rule W4, written in the sheet's parts column as "first || second"."""
    return [x.strip() for x in (s or "").split("||") if x.strip()]


def pick_part(parts: list[str], seed, query_id: str) -> int:
    """Rule W4: when the reviewed claim joins two or more claims and a fact-check of one would not check another,
    the query states one of them, chosen at random, not by the reviewer: random.Random("<draw seed>:part:<query id>").
    Returns the index of the kept part."""
    return random.Random(f"{seed}:part:{query_id}").randrange(len(parts))


def part_scores(text: str, parts: list[str]) -> list[float]:
    """For every part, the share of its words (4 letters or more) that the text also has: a reminder, never a decision,
    that the final text must state the kept part and not another one."""
    words = set(WORD.findall(text.lower()))
    return [len(words & set(WORD.findall(x.lower()))) / max(1, len(set(WORD.findall(x.lower())))) for x in parts]


def replacement_pool(q: dict, reviewed: list[dict], used: set[str]) -> tuple[list[dict], str]:
    """Rule Replacement: the kept rows the draw may use (decision K, eligible) that neither the main set nor
    the pilot uses, from the query's own cell; from its country when the query filled a short year (provenance
    drawn_from "top-up ...") or when its cell has none left. Sorted by cluster_id, so the pick depends only on the seed."""
    cell = q["provenance"]["cluster_id"].split(":")[0]
    free = [r for r in reviewed if r.get("decision") == "K" and r.get("eligible", True) and r["cluster_id"] not in used]
    own = sorted((r for r in free if r["cell"] == cell), key=lambda r: r["cluster_id"])
    if own and not str(q["provenance"].get("drawn_from", "")).startswith("top-up"):
        return own, f"its cell {cell}"
    country = sorted((r for r in free if r["country"] == q["stratum"]["country"]), key=lambda r: r["cluster_id"])
    return country, f"its country, {q['stratum']['country']}"


def pick_replacement(pool: list[dict], seed, query_id: str, before: int) -> dict:
    """The replacement, drawn at random from the pool with its own seed: "<draw seed>:replace:<query id>:<n>", n = the
    number of earlier replacements of the same query (so a second replacement is a new draw)."""
    return random.Random(f"{seed}:replace:{query_id}:{before}").choice(pool)


_pages: dict[Path, list[str]] = {}


def passage_page(pdf: Path, passage: str) -> str:
    """The PDF page(s) that hold the passage, for the reviewer who opens the report: pdftotext's pages
    (split at the form feed), letters and digits only, searched for runs of 12, then 8, then 5 words of the passage;
    "" when the file is not a PDF, is missing, or pdftotext is not installed."""
    if pdf.suffix.lower() != ".pdf" or not pdf.exists():
        return ""
    norm = lambda s: re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()  # noqa: E731
    if pdf not in _pages:
        import subprocess
        from fetch_pages import find_pdftotext
        exe = find_pdftotext()
        out = subprocess.run([exe, "-layout", str(pdf), "-"], capture_output=True).stdout.decode("utf-8", "replace") if exe else ""
        _pages[pdf] = [norm(t) for t in out.split("\f")]
    words = norm(passage).split()
    for n in (12, 8, 5):
        for i in range(0, max(1, len(words) - n + 1), n):
            hits = [k + 1 for k, t in enumerate(_pages[pdf]) if " ".join(words[i:i + n]) in t]
            if hits:
                return ", ".join(map(str, hits[:3]))
    return ""


def context_not_in_report(context: str, report: Path) -> str:
    """The pieces of a context (split at "…" or "...") that are not in the report's text (the .txt beside it), compared
    on letters and digits only with every space removed, so line breaks and letter-spaced headings ("El e cti o n W e e k")
    still match; "" when every piece is found or the report has no text."""
    txt = report.with_suffix(".txt")
    if not txt.exists():
        return ""
    squash = lambda s: re.sub(r"[^a-z0-9]+", "", s.lower())  # noqa: E731
    body = squash(txt.read_text(encoding="utf-8", errors="replace"))
    return "; ".join(piece.strip() for piece in re.split(r"…|\.\.\.", context) if squash(piece) and squash(piece) not in body)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["draw", "rewrite-sheet", "import-rewrite", "replace"])
    ap.add_argument("--keep", type=Path, default=None, help="rewrite-sheet: an earlier sheet whose free_text, parts, context and note are "
                                                              "copied for every query whose claim is unchanged")
    ap.add_argument("--other", type=Path, nargs="*", default=[], help="replace: the other query files (the pilot, or the main set), whose "
                                                                      "rows are already used")
    ap.add_argument("--query-id", default=None, help="replace: the query to replace")
    ap.add_argument("--work", type=Path, default=None)
    ap.add_argument("--allocation", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--per-country", default="", help="countries whose cells are pooled (rule 2)")
    ap.add_argument("--pool-short", action="store_true", help="top up every short cell from the candidates its country's other years left (option 2)")
    ap.add_argument("--pilot", type=int, default=0, help="size of the pilot set drawn from the candidates the main draw left (0 = none)")
    ap.add_argument("--pilot-out", type=Path, default=None, help="pilot query file (default: queries_pilot.json beside --out)")
    ap.add_argument("--allow-unfinished", action="store_true", help="draw although a cell of a quota-mode review must continue (tests only)")
    ap.add_argument("--queries", type=Path, default=None)
    ap.add_argument("--sheet", type=Path, default=None)
    ap.add_argument("--proposals", type=Path, default=None, help="rewrite-sheet: the model's proposals (propose_rewrites.py), copied into free_text "
                                                                   "where the closed model made none; import-rewrite: recorded as provenance.rewrite_proposal")
    ap.add_argument("--closed-proposals", type=Path, default=None, help="rewrite-sheet and import-rewrite: the closed model's proposals, one JSON line "
                                                                "per query (query_id, cluster_id, free_text, context, background, parts, note)")
    ap.add_argument("--reports", type=Path, default=None, help="the folder of the fetched reports: rewrite-sheet gives each passage's file and PDF "
                                                                 "page; import-rewrite checks that the context is copied from the report")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.command == "draw":
        alloc = json.load(open(args.allocation, encoding="utf-8"))["cells"]
        status_file = args.work / "drafting" / "review_status.json"
        if status_file.exists():  # a quota-mode review: every cell must have reached its target or have no row left
            todo = [c for c, v in json.load(open(status_file, encoding="utf-8")).items() if v["status"] == "continue"]
            if todo and not args.allow_unfinished:
                print(f"[draw] the review is not finished in {len(todo)} cells ({', '.join(todo)}); "
                      f"finish them and import again (review_needs.py import), then draw")
                return 1
        kept = [r for r in read_jsonl(args.work / "drafting" / "reviewed.jsonl") if r["decision"] == "K" and r.get("eligible", True)]
        by_cell: dict[str, list[dict]] = {}
        for r in kept:
            by_cell.setdefault(r["cell"], []).append(r)
        pooled = {s.strip() for s in args.per_country.split(",") if s.strip()}
        strata = []  # (name, main queries, candidates left after the main draw), for the pilot
        queries, report = [], ["# The draw", "", f"seed {args.seed}; per-country pooling: {', '.join(sorted(pooled)) or 'none'}; "
                                                  f"top-up of short cells from the same country: {'yes' if args.pool_short else 'no'}", "",
                               "| cell | queries | kept candidates | drawn | shortfall |", "|---|---|---|---|---|"]
        topups = []
        for country, cells in alloc.items():
            folder = FOLDER[country]
            if country in pooled:
                pool = [c for y in cells for c in by_cell.get(f"{folder}-{y}", [])]
                pool.sort(key=lambda c: c["cluster_id"])
                q = sum(cells.values())
                rng = random.Random(f"{args.seed}:{folder}")
                drawn = rng.sample(pool, min(q, len(pool)))
                for k, c in enumerate(drawn, start=1):
                    q_ = to_query(c, k, args.seed)
                    q_["provenance"]["drawn_from"] = f"{country} pooled over its cells"
                    queries.append(q_)
                report.append(f"| {country} (pooled) | {q} | {len(pool)} | {len(drawn)} | {max(0, q - len(pool))} |")
                taken = {c["cluster_id"] for c in drawn}
                strata.append((folder, q, [c for c in pool if c["cluster_id"] not in taken]))
                continue
            country_rest, missing, short = [], 0, []
            for y, q in cells.items():
                cell = f"{folder}-{y}"
                pool = sorted(by_cell.get(cell, []), key=lambda c: c["cluster_id"])
                rng = random.Random(f"{args.seed}:{cell}")
                drawn = rng.sample(pool, min(q, len(pool)))
                for k, c in enumerate(drawn, start=1):
                    queries.append(to_query(c, k, args.seed))
                report.append(f"| {cell} | {q} | {len(pool)} | {len(drawn)} | {max(0, q - len(pool))} |")
                taken = {c["cluster_id"] for c in drawn}
                rest = [c for c in pool if c["cluster_id"] not in taken]
                country_rest += rest
                if len(pool) < q:
                    missing += q - len(pool)
                    short.append(cell)
                strata.append((cell, q, rest))
            listed = {f"{folder}-{y}" for y in cells}  # kept rows of a year with no query of its own (a capped country,
            for cell in sorted(by_cell):               # Ghana 2024) can still fill the country's short years
                if cell.rsplit("-", 1)[0] == folder and cell not in listed:
                    country_rest += by_cell[cell]
            if args.pool_short and missing:
                # as a top-up: every year keeps its own draw; only the queries a short
                # year cannot fill are drawn, with their own seed, from the candidates the country's other years left
                rng = random.Random(f"{args.seed}:{folder}:top-up")
                extra = rng.sample(sorted(country_rest, key=lambda c: c["cluster_id"]), min(missing, len(country_rest)))
                for c in extra:
                    n_cell = sum(1 for q_ in queries if q_["provenance"]["cluster_id"].split(":")[0] == c["cell"])
                    q_ = to_query(c, n_cell + 1, args.seed)
                    q_["provenance"]["drawn_from"] = f"top-up of {', '.join(short)} from the other years of {country}"
                    queries.append(q_)
                    for name, _, rest in strata:
                        if name == c["cell"]:
                            rest.remove(c)
                topups.append(f"| {country} | {', '.join(short)} | {missing} | {len(extra)} | {missing - len(extra)} | "
                              f"{', '.join(f'{c_}: {n}' for c_, n in sorted(Counter(c['cell'] for c in extra).items()))} |")
        if topups:
            report += ["", "## Top-up of the short cells (--pool-short)", "",
                       "| country | short cells | missing queries | topped up | still missing | taken from |", "|---|---|---|---|---|---|"] + topups
        write_json(args.out, queries)
        if args.pilot:
            # The pilot is drawn after the main draw and only from what it left, so it cannot change the main set.
            rng = random.Random(f"{args.seed}:pilot")
            left = {name: list(rest) for name, _, rest in strata}
            weight = {name: q for name, q, _ in strata}
            pilot = []
            while len(pilot) < args.pilot:
                names = [n for n in sorted(left) if left[n] and weight[n] > 0]
                if not names:
                    break
                name = rng.choices(names, weights=[weight[n] for n in names])[0]
                c = left[name].pop(rng.randrange(len(left[name])))
                pilot.append(to_query(c, len(pilot) + 1, args.seed, query_id=f"pilot-{len(pilot) + 1:02d}", query_set="pilot"))
            pilot_out = args.pilot_out or args.out.parent / "queries_pilot.json"
            write_json(pilot_out, pilot)
            report += ["", f"## The pilot set: {len(pilot)} of {args.pilot} drawn (seed {args.seed}:pilot), from the kept candidates the main draw left",
                       "", "| query_id | cell | cluster_id |", "|---|---|---|"]
            report += [f"| {p['query_id']} | {p['stratum']['country']} {p['stratum']['year']} | {p['provenance']['cluster_id']} |" for p in pilot]
            if len(pilot) < args.pilot:
                report.append(f"\nSHORTFALL: only {len(pilot)} kept candidates were left for the pilot.")
            print(f"[draw] pilot: {len(pilot)} of {args.pilot} queries; {pilot_out}")
        (args.out.with_suffix(".draw.md")).write_text("\n".join(report) + "\n", encoding="utf-8")
        total = sum(sum(c.values()) for c in alloc.values())
        print(f"[draw] {len(queries)} of {total} queries drawn; report {args.out.with_suffix('.draw.md')}")
        return 0

    if args.command == "rewrite-sheet":
        import openpyxl
        from openpyxl.styles import Alignment, Font, PatternFill
        qs = read_json(args.queries)
        # the period and the verdict words come from the review log (the draw did not copy them into the queries); the
        # model's proposals, if any, fill "proposal" and are copied into free_text for the reviewer to accept or correct
        reviewed = {r["cluster_id"]: r for r in read_jsonl(args.work / "drafting" / "reviewed.jsonl")} if args.work else {}
        props = {r["query_id"]: r for r in read_jsonl(args.proposals)} if args.proposals else {}
        # The closed model's proposals: the closed model reads the report around each passage and proposes the sentence,
        # the copied words (context, background) and the W4 parts; the reviewer accepts or corrects every row. They are a
        # separate file so that the release can say which text came from the local model, which from the closed model, and what the
        # reviewer changed
        closed = {r["query_id"]: r for r in read_jsonl(args.closed_proposals)} if args.closed_proposals else {}
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "rewrite"
        # where the passage is in its report (file, PDF page, link) and the context column: the reviewer may
        # read what the report says about the same claim (the whole report) to learn who, where and when, and copies what was used into context
        # hint: the place and the date the passage gives (propose_rewrites.where_when), no longer appended in
        # brackets; the reviewer writes them into the sentence only when it cannot be understood without them
        # parts (rule W4): the reviewer writes "first claim || second claim" when the claim joins two or more claims
        # that one fact-check would not check together; import-rewrite then says which part the query keeps (at random)
        # no proposal column: free_text holds the closed model's suggestion (or the local model's proposal where
        # the closed model made none); the local model's proposals stay in their file and import-rewrite --proposals records them
        # no closed_note column and 'reworded' instead of 'checks': the closed model's reasons stay in its file
        # (and import-rewrite --closed-proposals records them); reworded is X when the text put into free_text differs from the claim
        cols = ["query_id", "cell", "period", "claim", "passage", "report_file", "pdf_page", "link", "verdict_words", "hint", "doubt",
                "reworded", "parts", "free_text", "context", "background", "note", "names_in_passage"]
        # background (after TREC topics and the TREC 2022 Health Misinformation track): who or what a
        # name of the query is, written from the report (reworded allowed, the report's words in context), for the judges
        # only; never searched, not part of the claim
        # --keep: the reviewer's work in an earlier sheet is carried over for every query whose claim is
        # unchanged, so the sheet can be rebuilt (after a replacement, for example) without losing anything
        kept_rows, carried, by_closed, not_put, not_carried = {}, 0, 0, [], []
        if args.keep and args.keep.exists():
            old = list(openpyxl.load_workbook(args.keep, read_only=True, data_only=True).worksheets[0].iter_rows(values_only=True))
            oh = [str(h) for h in old[0]]
            for row in old[1:]:
                if row and row[0]:
                    kept_rows[str(row[0])] = {k: ("" if row[i] is None else str(row[i])) for i, k in enumerate(oh) if i < len(row)}
        ws.append(cols)
        for c in ws[1]:
            c.font = Font(bold=True)
        # where to look hardest (on the text put into free_text): orange when it breaks a check
        # of the import; yellow when its wording differs from the claim (some rewordings describe the disinformation instead
        # of stating the claim, e.g. "A video showing burning ballots was posted")
        orange, yellow = PatternFill("solid", fgColor="F8CBAD"), PatternFill("solid", fgColor="FFF2CC")
        fi = cols.index("free_text") + 1
        for q in qs:
            p = q["provenance"]
            r = reviewed.get(p["cluster_id"], {})
            pr = props.get(q["query_id"], {})
            pr = pr if pr.get("cluster_id", p["cluster_id"]) == p["cluster_id"] else {}  # a proposal for a replaced query's old claim
            proposal = pr.get("proposal", "")
            pdf = args.reports.resolve() / p["file"] if args.reports else None
            mine = {"parts": "", "free_text": proposal, "context": "", "background": "", "note": ""}
            k = kept_rows.get(q["query_id"])
            same = bool(k) and k.get("claim", "").strip() == p["claim"].strip()
            if same:
                mine = {c: k.get(c, "") for c in mine} | {"free_text": k.get("free_text", "") or proposal}
                carried += 1
            elif k:  # named, so that work lost by an edited claim cell (us-2020-03) is never silent
                not_carried.append(q["query_id"])
            a = closed.get(q["query_id"])
            # a proposal for a replaced query's old claim is ignored; a context_only entry records what the closed model wrote into a row
            # the reviewer wrote and is never put into the sheet
            a = a if a and a.get("cluster_id") == p["cluster_id"] and not a.get("context_only") else None
            if a:
                # written already: the kept row holds more than the local model's proposal (the closed model's proposal, put in before,
                # or the reviewer's work), so a rebuild keeps it
                written = same and (any(k.get(c) for c in ("parts", "context", "background", "note"))
                                    or k.get("free_text", "") not in ("", proposal))
                if not written:
                    mine = {"parts": a.get("parts", ""), "free_text": a.get("free_text", "") or mine["free_text"],
                            "context": a.get("context", ""), "background": a.get("background", ""), "note": mine["note"]}
                    by_closed += 1
                elif any(k.get(c, "") != (a.get(c) or "") for c in ("parts", "free_text", "context", "background")):
                    not_put.append(q["query_id"])
            text = mine["free_text"]
            checks = rewrite_problems(text, q, mine["context"], r.get("period", "")) if text else []
            if mine["background"]:
                checks += ["background: " + w for w in background_problems(mine["background"], q, mine["context"], r.get("period", ""))]
            reworded = bool(text) and re.sub(r" \([^()]*\)\.?$", "", text).rstrip(". ").lower() != p["claim"].rstrip(". ").lower()
            ws.append([q["query_id"], f"{q['stratum']['country']} {q['stratum']['year']}", r.get("period", ""), p["claim"], p["passage"],
                       str(pdf) if pdf else p["file"], passage_page(pdf, p["passage"]) if pdf else "", q["source"]["url_or_doi"],
                       r.get("verdict_words", ""), pr.get("hint", ""), pr.get("doubt", ""),
                       ("X" if reworded else "") + (" (check: " + "; ".join(checks) + ")" if checks else ""), mine["parts"],
                       mine["free_text"], mine["context"], mine["background"], mine["note"], "; ".join(p["entities"])])
            if checks or reworded:
                ws.cell(row=ws.max_row, column=fi).fill = orange if checks else yellow
        for i, w in enumerate([14, 18, 10, 40, 60, 25, 8, 25, 30, 20, 25, 10, 35, 45, 35, 40, 25, 30], start=1):
            ws.column_dimensions[openpyxl.utils.get_column_letter(i)].width = w
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
        ws.freeze_panes = "A2"
        wb.save(args.out)
        print(f"[draw] rewrite sheet with {len(qs)} queries: {args.out}" + (f"; the work of {carried} rows carried over from {args.keep}" if args.keep else "")
              + (f"; the closed model's proposal put into {by_closed} rows" if args.closed_proposals else "")
              + (f"; the closed model's proposal NOT put into {len(not_put)} rows already written: {', '.join(not_put)}" if not_put else "")
              + (f"; NOT carried over, because the old sheet's claim cell differs from the query's claim (a replaced query, or an "
                 f"edited claim cell: copy the work back by hand): {', '.join(not_carried)}" if not_carried else ""))
        return 0

    if args.command == "import-rewrite":
        import openpyxl
        qs = read_json(args.queries)
        ws = openpyxl.load_workbook(args.sheet, read_only=True, data_only=True).worksheets[0]
        rows = list(ws.iter_rows(values_only=True))
        header = [str(h) for h in rows[0]]
        col = lambda name: header.index(name) if name in header else None  # noqa: E731
        get = lambda r, name: str(r[col(name)] or "").strip() if col(name) is not None and col(name) < len(r) else ""  # noqa: E731
        sheet = {get(r, "query_id"): r for r in rows[1:] if r and r[0]}
        warnings, missing, changed, proposed, with_context, picks = [], [], 0, 0, 0, []
        closed = {r["query_id"]: r for r in read_jsonl(args.closed_proposals)} if args.closed_proposals else {}
        by_closed, closed_text_changed, closed_any_changed = 0, 0, 0
        # the local model's proposals (no longer a column of the sheet): from --proposals, matched on the cluster
        local = {r["query_id"]: r for r in read_jsonl(args.proposals)} if args.proposals else {}
        # rule W4: the parts are fixed at the first import that shows the pick; <out>.w4.json remembers them
        # (key: query id and cluster id, so a replaced query starts afresh), and a later change is a warning, so nobody can
        # remove or reorder a split after seeing which part it keeps
        fixed_path = args.out.with_suffix(".w4.json")
        fixed = json.loads(fixed_path.read_text(encoding="utf-8")) if fixed_path.exists() else {}
        stated = []  # (query id, what it states, split?) for same_statements (rules W4 and D3)
        for q in qs:
            r = sheet.get(q["query_id"])
            if r and col("claim") is not None and get(r, "claim") != q["provenance"]["claim"].strip():
                # the row was written for another claim (the query was replaced): never import it silently
                warnings.append(f"{q['query_id']}: the sheet's row is for another claim (was the query replaced?): rebuild the sheet "
                                f"with rewrite-sheet --keep, which keeps the work of every other row")
                missing.append(q["query_id"])
                continue
            text = get(r, "free_text") if r else ""
            if not text:
                missing.append(q["query_id"])
                continue
            parts = split_parts(get(r, "parts"))
            key = f"{q['query_id']}|{q['provenance']['cluster_id']}"
            if key in fixed and parts != fixed[key]:
                warnings.append(f"{q['query_id']}: the parts changed after an import had shown the pick (they were: "
                                f"{' || '.join(fixed[key])}): rule W4 fixes them at the first import; put them back")
            elif len(parts) > 1:
                fixed.setdefault(key, parts)
            if len(parts) == 1:
                warnings.append(f"{q['query_id']}: parts holds one claim only: write the claims separated by ||, or leave it empty")
            elif parts:  # rule W4: the part is chosen at random with the draw's seed, never by the reviewer
                k = pick_part(parts, q["provenance"]["draw_seed"], q["query_id"])
                q["provenance"]["rewrite_parts"] = parts
                q["provenance"]["rewrite_part_kept"] = k + 1
                picks.append(f"- {q['query_id']}: keep part {k + 1} of {len(parts)}: {parts[k]}")
                stated.append((q["query_id"], parts[k], True))
                scores = part_scores(text, parts)
                if scores[k] < max(scores):
                    warnings.append(f"{q['query_id']}: W4 keeps part {k + 1} ({parts[k]}), but the text looks closer to another part: "
                                    f"write free_text for part {k + 1}")
            if len(parts) < 2:
                stated.append((q["query_id"], text, False))
            context = get(r, "context")
            background = get(r, "background")
            # W5 reworded: the background may be reworded, so it is no longer a receipt; the names and years of
            # the text and of the background must be in the passage, the claim, the draft or the context
            warnings += [f"{q['query_id']}: {w}" for w in rewrite_problems(text, q, context, get(r, "period"))]
            q.pop("background", None)
            if background:  # shown to the judges only (pool/judges_sheet.py, judge/prompts.py); no system searches it
                q["background"] = background
                warnings += [f"{q['query_id']}: background: {w}" for w in background_problems(background, q, context, get(r, "period"))]
            q["free_text"] = text
            if context:  # the words copied from the report around the passage, released so each added word can be checked
                q["provenance"]["rewrite_context"] = context
                with_context += 1
                lost = context_not_in_report(context, args.reports / q["provenance"]["file"]) if args.reports else ""
                if lost:
                    warnings.append(f"{q['query_id']}: context not found word for word in the report: {lost}")
            lp = local.get(q["query_id"], {})
            proposal = get(r, "proposal") or (lp.get("proposal", "") if lp.get("cluster_id") == q["provenance"]["cluster_id"] else "")
            if proposal:  # the model's proposal is kept beside the final text; the paper reports how many were changed
                proposed += 1
                changed += text != proposal
                q["provenance"]["rewrite_proposal"] = proposal
            note = get(r, "note")
            if note:
                q["provenance"]["rewrite_note"] = note
            a = closed.get(q["query_id"])
            if a and a.get("cluster_id") == q["provenance"]["cluster_id"] and a.get("context_only"):
                # The closed model wrote only the context of a row the reviewer wrote (pilot-01, pilot-02, pilot-05)
                q["provenance"]["rewrite_closed"] = {"context": a.get("context", ""), "note": a.get("note", ""), "model": a.get("model", ""),
                                                     "context_only": True}
            elif a and a.get("cluster_id") == q["provenance"]["cluster_id"]:
                # The closed model's proposal is kept beside the final text, like the local model's; the paper reports how
                # many sentences, and how many rows in any column, the reviewer changed
                mine = {"free_text": a.get("free_text", ""), "context": a.get("context", ""), "background": a.get("background", ""),
                        "parts": split_parts(a.get("parts", ""))}
                q["provenance"]["rewrite_closed"] = mine | {"note": a.get("note", ""), "model": a.get("model", "")}
                by_closed += 1
                closed_text_changed += text != mine["free_text"]
                closed_any_changed += (text, context, background, parts) != (mine["free_text"], mine["context"], mine["background"], mine["parts"])
        warnings += same_statements(stated)
        write_json(args.out, qs)
        fixed_path.write_text(json.dumps(fixed, ensure_ascii=False, indent=1), encoding="utf-8")
        rep = [f"# Rewrite import: {len(qs) - len(missing)} of {len(qs)} queries rewritten; {len(missing)} missing; {len(warnings)} warnings"
               + (f"; {changed} of {proposed} local-model proposals differ from the final text" if proposed else "")
               + (f"; {closed_text_changed} of {by_closed} of the closed model's sentences changed by the reviewer ({closed_any_changed} rows "
                  f"changed in any column)" if args.closed_proposals else "")
               + f"; {with_context} with context from the report", ""]
        rep += [f"- {w}" for w in warnings] + [f"- missing rewrite: {m}" for m in missing]
        if picks:
            rep += ["", "## Rule W4: the part each split query keeps (chosen at random with the draw's seed)", ""] + picks
        (args.out.with_suffix(".rewrite.md")).write_text("\n".join(rep) + "\n", encoding="utf-8")
        print(rep[0])
        for line in picks:
            print("[draw] W4" + line[1:])
        return 1 if missing else 0

    if args.command == "replace":
        # rule Replacement: only a query whose row breaks a review rule is replaced; its decision must already
        # be corrected in the review sheet and imported (review_needs.py import), so the log says D
        qs = read_json(args.queries)
        others = [x for f in args.other for x in read_json(f)]
        reviewed = list(read_jsonl(args.work / "drafting" / "reviewed.jsonl"))  # read_jsonl yields once: keep a list
        log = {r["cluster_id"]: r for r in reviewed}
        q = next((x for x in qs if x["query_id"] == args.query_id), None)
        if q is None:
            print(f"[draw] replace: no query {args.query_id} in {args.queries}")
            return 1
        old = q["provenance"]["cluster_id"]
        if log.get(old, {}).get("decision") != "D":
            print(f"[draw] replace: {old} ({args.query_id}) is not D in the review log: correct its decision in the review sheet "
                  f"(D, its rule, the reason in note), run review_needs.py import, then replace")
            return 1
        used = {x["provenance"]["cluster_id"] for x in qs + others} | {c for x in qs + others for c in x["provenance"].get("replaces", [])}
        pool, scope = replacement_pool(q, reviewed, used)
        before = q["provenance"].get("replaces", [])
        backup = args.queries.with_name(f"{args.queries.stem}.before_replace_{q['query_id']}_{len(before) + 1}.json")
        if not pool:
            # when the country has no unused kept row either, the query is removed and the paper
            # reports the shortfall, as the draw does; a row of another country would change the country proportions
            write_json(backup, qs)
            write_json(args.out, [x for x in qs if x is not q])
            with open(args.work / "drafting" / "replacements.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"query_id": q["query_id"], "replaced": old, "rule": log[old].get("rule", ""), "note": log[old].get("note", ""),
                                     "new": None, "pool": 0, "scope": f"none left in {q['stratum']['country']}: query removed"},
                                    ensure_ascii=False) + "\n")
            print(f"[draw] replace {q['query_id']}: {old} ({log[old].get('rule', '')}): no unused kept row left in its year or in "
                  f"{q['stratum']['country']}, so the query is REMOVED ({len(qs) - 1} queries left); the old file is kept as {backup.name}")
            return 0
        row = pick_replacement(pool, q["provenance"]["draw_seed"], q["query_id"], len(before))
        new = to_query(row, q["provenance"]["draw_rank"], q["provenance"]["draw_seed"], query_id=q["query_id"], query_set=q.get("set", "main"))
        new["provenance"]["replaces"] = before + [old]
        # the old drawn_from stays at the front, so a query that filled a short year ("top-up ...") still draws from its
        # country if it is replaced a second time
        first = str(q["provenance"].get("drawn_from") or "")
        new["provenance"]["drawn_from"] = ((first + "; " if first else "") + f"replacement of {old} ({log[old].get('rule', '')}), drawn at random "
                                           f"(seed {q['provenance']['draw_seed']}:replace:{q['query_id']}:{len(before)}) from the {len(pool)} unused kept rows of {scope}")
        write_json(backup, qs)
        write_json(args.out, [new if x is q else x for x in qs])
        with open(args.work / "drafting" / "replacements.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"query_id": q["query_id"], "replaced": old, "rule": log[old].get("rule", ""), "note": log[old].get("note", ""),
                                 "new": row["cluster_id"], "pool": len(pool), "scope": scope}, ensure_ascii=False) + "\n")
        print(f"[draw] replace {q['query_id']}: {old} ({log[old].get('rule', '')}) -> {row['cluster_id']}, drawn from the {len(pool)} "
              f"unused kept rows of {scope}; the old file is kept as {backup.name}")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
