"""
import_formulas.py -- the writers' Boolean formulas, written in a spreadsheet,
checked and merged into the paper's query file (version 6).

Why a spreadsheet. The writers are not programmers; an Excel file (or a CSV)
they fill is what they can use, and one row per (query, author) keeps every
formula with its entities, its not-expressible notes and its minutes, which is
what the query file needs (common/queries.py).

Usage:
  python queries/import_formulas.py template --queries Q.json --out sheet_annotator_a.xlsx --author annotator_a [--half 1of2]
  python queries/import_formulas.py import   --queries Q.json --sheet sheet_annotator_a.xlsx --out Q_with_annotator_a.json
         [--resolve-redirects [--redirect-cache FILE]] [--catalog WORK/stats/entity_catalog.csv]
         [--exclusions WORK/exclusions.json --exclude-as A] [--report report.md]

The sheet (.xlsx or .csv) has the columns the template writes, with query_id,
author_id and narrative filled in:
  query_id, author_id, narrative, formula, not_expressible, minutes, refinements, notes
  narrative    the query text, exactly as the query file has it; the writer reads it and
               must not edit it (an edited narrative is reported and ignored)
  formula      the Wikipedia titles in double quotes, combined with AND, OR, NOT and parentheses:
               ("Ukraine" OR "Armed Forces of Ukraine") AND "Black market" AND NOT "Video game"
               The quotes matter: a title can contain an operator word ("Trinidad and Tobago")
               or parentheses ("Mercury (planet)"). The import turns the titles into keys
               e1, e2, ... (common/boolean.quoted_to_keys) because the query file and every
               consumer use the key form; the text as written is kept in 'formula_written'.
               The old key form (formula 'e1 AND e2' plus an 'entities' column
               'e1 = Ukraine; e2 = NATO') is still accepted, so older sheets import.
  not_expressible   concepts of the narrative that have no page, separated by semicolons
  minutes, refinements   numbers; notes   free text

--half 1of2 / 2of2, --common N, --all-ordered (template) and the command assign-main split the queries between
the two writers. They are NOT used for the paper, where both writers write every query and each writer's formulas
are one run.

--resolve-redirects (import): asks the English Wikipedia API for today's title
of every page the writer wrote ("Bioweapon" -> "Biological agent"); every
answer is cached in a JSON file that is released with the collection, so the
resolution is reproducible offline. When today's page differs from the page as
written, BOTH are used: the term becomes (e3 OR e3w), e3 today's page and e3w
the page as written, marked alias_of e3 (the archive's
entities keep some names that Wikipedia has since changed, so it may use an old name, "Twitter" rather than today's
"X (social network)"). Without the flag the titles are taken as written.
The rule works in BOTH directions: the writer may also have typed today's name while the archive was linked
under an older one ("Autism" today, "Autism spectrum" in the archive, 174 documents;
"X (social network)" today, "Twitter" in the archive). So every name that
Wikipedia redirects TO today's page, today, is used too: keys e3o1, e3o2, ...,
marked alias_of e3 and alias_kind 'older name' (the page as typed is alias_kind
'as written'). These names are asked once per page (prop=redirects) and cached in
--incoming-cache (default: wikipedia_incoming_redirects.json next to the redirect
cache), released with the collection. With --catalog only the names the archive
actually links are written into the formula; a name the archive never links
matches nothing, so the result is the same and the formula stays short. Some
redirects are pages merged into a larger one rather than renames ("Passport
fraud" -> "Passport"); they are used too: the rule is mechanical, from Wikipedia
alone, the same for both writers. systems/graph_boolean.py --current-names-only
runs the formulas with today's pages only, to measure what the archive's old
names cost (eval/stale_names.py).

--catalog (import only; the writer never sees it):
adds to the report how many archive documents each page has, and writes the
page's address in the store's own spelling. The exact title is used whenever
the store has it; only when it does not, a page of the store that differs by
capital letters alone is used (the store
writes CRIME, Election_commission, Project_On_Government_Oversight), the one
with the most documents if there are several, and the report says so.
Wikipedia titles are case-sensitive after the first letter ('QI' the quiz
show, 'Qi' the concept), and the store holds 53 such pairs ('Islamic_State' 738
documents, 'Islamic_state' 6), so the case-blind match is never used when the
exact page exists.

What 'import' checks, per row: the formula parses and every title is quoted
(or, in the key form, every key has a title and every title is used); a
formula that would return "everything except" (common/boolean.is_wide_not)
is flagged; an edited narrative is flagged. Rows whose formula does not
parse or whose keys and titles do not match are NOT imported and the exit code
is 1; everything else is written. A row replaces an earlier formula of the same
author for the same query. --exclusions/--exclude-as records, for the pool,
that annotator A must not judge the queries this sheet gives formulas for.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.boolean import FormulaError, is_wide_not, parse, quoted_to_keys, terms  # noqa: E402
from common.jsonl import write_json  # noqa: E402
from common.queries import check_formula, load_queries  # noqa: E402
from queries.entity_catalog import load_catalog, title_from_uri, uri_from_title  # noqa: E402

SHEET_COLUMNS = ["query_id", "author_id", "narrative", "formula", "not_expressible", "minutes", "refinements", "notes"]
WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_PAUSE = 1.0   # seconds between two calls (Wikipedia's API etiquette: ask one thing at a time, not in bursts)
WIKI_TRIES = 8     # a 429 or 503 answer is retried after the wait Wikipedia asks for
ALIAS_SUFFIX = "w"  # e3w = the page as the writer typed it, kept next to e3 = today's page
OLDER_SUFFIX = "o"  # e3o1, e3o2 = older names that Wikipedia redirects to e3's page today (both directions)
SPLIT_SEED = 0
MAIN_AUTHOR = "main"


# ---------------------------------------------------------------------------
# Sheets: xlsx or csv
# ---------------------------------------------------------------------------
def read_sheet(path: Path) -> list[dict]:
    if path.suffix.lower() == ".xlsx":
        rows = _read_xlsx(path)
    else:
        rows = _read_csv(path)
    missing = [c for c in ("query_id", "author_id", "formula") if rows and c not in rows[0]]
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}; expected {SHEET_COLUMNS}")
    return rows


def _read_csv(path: Path) -> list[dict]:
    """Excel writes UTF-8 with a byte-order mark or the Windows code page; try both."""
    for enc in ("utf-8-sig", "cp1252"):
        try:
            with open(path, encoding=enc, newline="") as fh:
                return list(csv.DictReader(fh))
        except UnicodeDecodeError:
            continue
    raise SystemExit(f"cannot decode {path}")


def _read_xlsx(path: Path) -> list[dict]:
    import openpyxl
    ws = openpyxl.load_workbook(path, read_only=True, data_only=True).worksheets[0]
    it = ws.iter_rows(values_only=True)
    header = [str(c).strip() if c is not None else "" for c in next(it, [])]
    rows = []
    for vals in it:
        if vals is None or all(v is None or str(v).strip() == "" for v in vals):
            continue
        rows.append({h: ("" if v is None else str(v)) for h, v in zip(header, vals) if h})
    return rows


def write_sheet(path: Path, rows: list[dict]) -> None:
    if path.suffix.lower() == ".xlsx":
        import openpyxl
        from openpyxl.styles import Alignment, Font
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "formulas"
        ws.append(SHEET_COLUMNS)
        for c in ws[1]:
            c.font = Font(bold=True)
        for r in rows:
            ws.append([r.get(c, "") for c in SHEET_COLUMNS])
        widths = {"query_id": 14, "author_id": 12, "narrative": 70, "formula": 70,
                  "not_expressible": 30, "minutes": 9, "refinements": 11, "notes": 30}
        for i, c in enumerate(SHEET_COLUMNS, 1):
            ws.column_dimensions[openpyxl.utils.get_column_letter(i)].width = widths[c]
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
        ws.freeze_panes = "D2"
        wb.save(path)
    else:
        with open(path, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=SHEET_COLUMNS)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in SHEET_COLUMNS})


# ---------------------------------------------------------------------------
# The cells' small grammar
# ---------------------------------------------------------------------------
def parse_entities(text: str) -> dict[str, str]:
    """'e1 = Ukraine; e2 = Biological warfare' -> {'e1': 'Ukraine', 'e2': 'Biological warfare'}."""
    out: dict[str, str] = {}
    for part in (text or "").split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"entity entry without '=': '{part}'")
        k, v = part.split("=", 1)
        k, v = k.strip(), v.strip()
        if not k or not v:
            raise ValueError(f"empty key or title in '{part}'")
        if k in out:
            raise ValueError(f"key {k} given twice")
        out[k] = v
    return out


def split_list(text: str) -> list[str]:
    return [p.strip() for p in (text or "").split(";") if p.strip()]


MINUTE_WORDS = re.compile(r"\s*(min|mins|minute|minutes|minuto|minuti|')\.?$", re.IGNORECASE)


def to_number(text: str):
    """'3' -> 3, '4,5' -> 4.5, and a number followed by a minute word as the writers type it:
    '3 min', '3min', '3 minuti', "3'" -> 3. Anything else -> None (the report says so)."""
    text = MINUTE_WORDS.sub("", (text or "").strip()).strip().replace(",", ".")
    if not text:
        return None
    try:
        f = float(text)
    except ValueError:
        return None
    return int(f) if f.is_integer() else f


# ---------------------------------------------------------------------------
# Wikipedia redirects
# ---------------------------------------------------------------------------
class RedirectResolver:
    """title as written -> {'resolved': canonical title, 'exists': bool}, cached in a JSON file."""

    def __init__(self, cache_path: Path | None, online: bool):
        self.cache_path = cache_path
        self.online = online
        self.cache: dict[str, dict] = {}
        if cache_path and cache_path.exists():
            self.cache = json.loads(cache_path.read_text(encoding="utf-8"))

    def resolve(self, titles: list[str]) -> None:
        todo = sorted({t for t in titles if t not in self.cache})
        if todo and not self.online:
            print(f"WARNING: {len(todo)} titles are not in the redirect cache and are taken as written "
                  f"(add --resolve-redirects to ask Wikipedia): {', '.join(todo[:5])}", file=sys.stderr)
        if not todo or not self.online:
            return
        print(f"asking Wikipedia for {len(todo)} titles ({len(self.cache)} already in the cache)", file=sys.stderr)
        for i in range(0, len(todo), 50):  # the API takes at most 50 titles per call
            chunk = todo[i:i + 50]
            q = self._ask({"action": "query", "titles": "|".join(chunk), "redirects": 1, "format": "json", "formatversion": 2})
            normalized = {n["from"]: n["to"] for n in q.get("normalized", [])}
            redirects = {r["from"]: r["to"] for r in q.get("redirects", [])}
            missing = {p["title"] for p in q.get("pages", []) if p.get("missing") or p.get("invalid")}
            for t in chunk:
                t1 = normalized.get(t, t)
                t2 = redirects.get(t1, t1)
                self.cache[t] = {"resolved": t2, "exists": t2 not in missing}
            self._save()  # after every call, so a stopped run keeps what it already got
            time.sleep(WIKI_PAUSE)

    @staticmethod
    def _ask(params: dict) -> dict:
        """One API call. When Wikipedia answers 429 (too many requests) or 503 (busy) it says how long to
        wait (Retry-After); the call waits that long and tries again, up to WIKI_TRIES times."""
        req = urllib.request.Request(WIKI_API + "?" + urllib.parse.urlencode(params),
                                     headers={"User-Agent": "paper-code/import_formulas (research test collection)"})
        for attempt in range(1, WIKI_TRIES + 1):
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    return json.load(resp).get("query", {})
            except urllib.error.HTTPError as e:
                if e.code not in (429, 503) or attempt == WIKI_TRIES:
                    raise
                ra = (e.headers.get("Retry-After") or "").strip()
                wait = min(int(ra) if ra.isdigit() else 10 * attempt, 300)
                print(f"Wikipedia answered {e.code}: waiting {wait} s (try {attempt} of {WIKI_TRIES})", file=sys.stderr)
                time.sleep(wait)
        return {}

    def _save(self) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    def get(self, title: str) -> dict | None:
        return self.cache.get(title)


class IncomingRedirects:
    """today's page -> the names that Wikipedia redirects to it today (main namespace), cached in a JSON file.
    The other direction of RedirectResolver: the archive may hold the older name of a page the writer typed
    under today's name."""

    def __init__(self, cache_path: Path | None, online: bool):
        self.cache_path = cache_path
        self.online = online
        self.cache: dict[str, list[str]] = {}
        if cache_path and cache_path.exists():
            self.cache = json.loads(cache_path.read_text(encoding="utf-8"))

    def fetch(self, titles: list[str]) -> None:
        todo = sorted({t for t in titles if t not in self.cache})
        if todo and not self.online:
            print(f"WARNING: {len(todo)} pages are not in the cache of incoming redirects, so their older names are not used "
                  f"(add --resolve-redirects to ask Wikipedia): {', '.join(todo[:5])}", file=sys.stderr)
        if not todo or not self.online:
            return
        print(f"asking Wikipedia for the names that redirect to {len(todo)} pages ({len(self.cache)} already in the cache)", file=sys.stderr)
        for i in range(0, len(todo), 50):
            chunk = todo[i:i + 50]
            got: dict[str, list[str]] = {t: [] for t in chunk}
            params = {"action": "query", "titles": "|".join(chunk), "prop": "redirects", "rdlimit": "max",
                      "rdnamespace": 0, "format": "json", "formatversion": 2}
            while True:  # a page with many redirects comes in several answers ('continue')
                d = _ask_raw(params)
                q = d.get("query", {})
                back = {n["to"]: n["from"] for n in q.get("normalized", [])}
                for page in q.get("pages", []):
                    t = back.get(page["title"], page["title"])
                    if t in got:
                        got[t] += [r["title"] for r in page.get("redirects", [])]
                if "continue" not in d:
                    break
                params = {**params, **d["continue"]}
                time.sleep(WIKI_PAUSE)
            self.cache.update({t: sorted(set(v)) for t, v in got.items()})
            if self.cache_path:  # after every call, so a stopped run keeps what it already got
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                self.cache_path.write_text(json.dumps(self.cache, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
            time.sleep(WIKI_PAUSE)

    def get(self, title: str) -> list[str]:
        return self.cache.get(title) or []


def _ask_raw(params: dict) -> dict:
    """One API call returning the whole answer (with 'continue'); waits and retries on 429 and 503 like RedirectResolver._ask."""
    req = urllib.request.Request(WIKI_API + "?" + urllib.parse.urlencode(params),
                                 headers={"User-Agent": "paper-code/import_formulas (research test collection)"})
    for attempt in range(1, WIKI_TRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 503) or attempt == WIKI_TRIES:
                raise
            ra = (e.headers.get("Retry-After") or "").strip()
            wait = min(int(ra) if ra.isdigit() else 10 * attempt, 300)
            print(f"Wikipedia answered {e.code}: waiting {wait} s (try {attempt} of {WIKI_TRIES})", file=sys.stderr)
            time.sleep(wait)
    return {}


# ---------------------------------------------------------------------------
# One row -> one formula record
# ---------------------------------------------------------------------------
def formula_and_titles(row: dict) -> tuple[str, str, dict[str, str]]:
    """(formula in key form, formula as written, {key: title}) from a sheet row, in
    either form. Raises ValueError with the message the writer gets."""
    written = (row.get("formula") or "").strip()
    key_form, titles = quoted_to_keys(written)  # raises FormulaError if it does not parse
    column = parse_entities(row.get("entities") or "")
    if titles and column:
        raise ValueError("the formula has quoted titles and the 'entities' column is also filled: use one form only")
    if not titles:
        if not column:
            raise ValueError("no quoted title in the formula: write every Wikipedia title in double quotes, "
                             "for example (\"Ukraine\" OR \"Kyiv\") AND \"Biological agent\"")
        titles = column
    return key_form, written, titles


class CatalogLookup:
    """The archive's own spelling of a page (see --catalog above): the exact title first; only when the store has
    no page with exactly that title, a page that differs by capital letters alone, the one with the most documents."""

    def __init__(self, rows: list[dict]):
        self.exact = {title_from_uri(r["dbpedia_uri"]): r for r in rows}
        self.folded: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            self.folded[r["_key"]].append(r)

    def find(self, uri: str) -> tuple[dict | None, list[dict]]:
        """(the catalog row used or None, the other rows that differ only by capital letters when no exact page exists)"""
        title = title_from_uri(uri)
        if title in self.exact:
            return self.exact[title], []
        rows = sorted(self.folded.get(title.casefold(), []), key=lambda r: (-r["n_docs"], r["dbpedia_uri"]))
        return (rows[0], rows[1:]) if rows else (None, [])


def _entity(k: str, as_written: str, title: str, catalog: CatalogLookup | None, warnings: list[str]) -> dict:
    """The entity record of one key: the page title used and its DBpedia address (the store's own spelling when
    the catalog is given, which adds a [archive] line with the archive count to the report)."""
    uri = uri_from_title(title)
    if catalog is not None:
        hit, others = catalog.find(uri)
        if hit is None:
            warnings.append(f"[archive] {k} = '{title}' is not in the archive's entity links: that term matches nothing")
        else:
            case_note = ""
            if title_from_uri(hit["dbpedia_uri"]) != title_from_uri(uri):
                case_note = (f"; matched ignoring capital letters as '{title_from_uri(hit['dbpedia_uri'])}'"
                             + (f" (also in the store: {', '.join(title_from_uri(o['dbpedia_uri']) + ' ' + str(o['n_docs']) for o in others)})" if others else ""))
            uri = hit["dbpedia_uri"]  # the store's own spelling of the address
            warnings.append(f"[archive] {k} = '{title}': {hit['n_docs']} documents{case_note}")
    return {"label": as_written, "wikipedia_title": title, "wikidata_qid": None, "dbpedia_uri": uri}


def row_to_formula(row: dict, resolver: RedirectResolver | None, catalog_by_key: "CatalogLookup | None",
                   incoming: "IncomingRedirects | None" = None) -> tuple[dict | None, list[str], list[str]]:
    """Returns (formula record or None, errors, warnings). Errors block the row."""
    errors: list[str] = []
    warnings: list[str] = []
    try:
        formula_text, written, titles = formula_and_titles(row)
        node = parse(formula_text)
    except FormulaError as e:
        return None, [f"formula does not parse: {e}"], []
    except ValueError as e:
        return None, [str(e)], []
    used = terms(node)
    for k in used:
        if k not in titles:
            errors.append(f"key {k} is in the formula but has no title in 'entities'")
    for k in titles:
        if k not in used:
            errors.append(f"key {k} has a title but is not in the formula")
    if errors:
        return None, errors, []
    entities: dict[str, dict] = {}
    aliases: dict[str, str] = {}  # key -> key of the page as written, when it differs from today's page
    older: dict[str, list[str]] = {}  # key -> keys of the older names that redirect to today's page
    for k, as_written in titles.items():
        title = as_written
        rec = resolver.get(as_written) if resolver else None
        if rec:
            if rec["resolved"] != as_written:
                title = rec["resolved"]
                if uri_from_title(title) != uri_from_title(as_written):
                    aliases[k] = k + ALIAS_SUFFIX
                    warnings.append(f"{k}: '{as_written}' redirects to '{title}' on Wikipedia; both pages are used "
                                    f"(the formula reads ({k} OR {aliases[k]}))")
            if not rec["exists"]:
                warnings.append(f"{k}: '{as_written}' is not a page on English Wikipedia")
        entities[k] = _entity(k, as_written, title, catalog_by_key, warnings)
        if k in aliases:
            entities[aliases[k]] = dict(_entity(aliases[k], as_written, as_written, catalog_by_key, warnings),
                                        alias_of=k, alias_kind="as written")
        if incoming is not None:
            seen = {uri_from_title(title), uri_from_title(as_written)}
            names = []
            for n in incoming.get(title):
                u = uri_from_title(n)
                if u in seen:
                    continue
                seen.add(u)
                # with the catalog, only the names the archive links (the others match nothing anyway)
                if catalog_by_key is None or title_from_uri(u) in catalog_by_key.exact:
                    names.append(n)
            if names:
                older[k] = [f"{k}{OLDER_SUFFIX}{j}" for j in range(1, len(names) + 1)]
                group = [k] + ([aliases[k]] if k in aliases else []) + older[k]
                warnings.append(f"{k}: the archive also links older names that Wikipedia redirects to '{title}' today: "
                                + "; ".join(f"'{n}'" for n in names) + f"; all are used (the formula reads ({' OR '.join(group)}))")
                for ok, n in zip(older[k], names):
                    entities[ok] = dict(_entity(ok, n, n, catalog_by_key, warnings), alias_of=k, alias_kind="older name")
    for k in titles:
        group = ([aliases[k]] if k in aliases else []) + older.get(k, [])
        if group:
            # "X (social network)" becomes ("X (social network)" OR "Twitter"): a fact-check linked to any name of the page matches.
            formula_text = re.sub(rf"(?<![A-Za-z0-9_.\-]){re.escape(k)}(?![A-Za-z0-9_.\-])", f"({' OR '.join([k] + group)})", formula_text)
    if aliases or older:
        node = parse(formula_text)
    if is_wide_not(node):
        warnings.append("formula is a wide NOT ('everything except'); the graph system refuses it unless --allow-wide-not")
    out = {
        "author_id": (row.get("author_id") or "").strip(),
        "formula": formula_text,
        "formula_written": written,
        "entities": entities,
        "not_expressible": split_list(row.get("not_expressible") or ""),
        "minutes": to_number(row.get("minutes")),
        "refinements": to_number(row.get("refinements")),
    }
    if (row.get("notes") or "").strip():
        out["notes"] = row["notes"].strip()
    if (row.get("minutes") or "").strip() and out["minutes"] is None:
        warnings.append(f"minutes '{row['minutes']}' is not a number; stored as empty")
    return out, [], warnings


# ---------------------------------------------------------------------------
# Template, split in halves
# ---------------------------------------------------------------------------
def split_halves(queries: list[dict], seed: int = SPLIT_SEED) -> dict[str, int]:
    """query_id -> 1 or 2, alternating inside every (country, year) cell after a seeded shuffle."""
    cells: dict[tuple, list[str]] = defaultdict(list)
    for q in queries:
        st = q.get("stratum") or {}
        cells[(st.get("country"), st.get("year"))].append(q["query_id"])
    rng = random.Random(seed)
    out: dict[str, int] = {}
    start = 0
    for key in sorted(cells, key=str):
        ids = sorted(cells[key])
        rng.shuffle(ids)
        for i, qid in enumerate(ids):
            out[qid] = 1 + (i + start) % 2
        start = (start + len(ids)) % 2  # the next cell starts with the other half, so odd cells balance out
    return out


def common_subset(queries: list[dict], n: int, seed: int = SPLIT_SEED) -> set[str]:
    """n query ids spread over the (country, year) cells: a seeded shuffle inside every
    cell, then one query per cell in turn until n are taken (cells in a seeded order)."""
    cells: dict[tuple, list[str]] = defaultdict(list)
    for q in queries:
        st = q.get("stratum") or {}
        cells[(st.get("country"), st.get("year"))].append(q["query_id"])
    rng = random.Random(seed + 1)
    order = sorted(cells, key=str)
    rng.shuffle(order)
    for key in order:
        cells[key].sort()
        rng.shuffle(cells[key])
    out: set[str] = set()
    while len(out) < min(n, len(queries)):
        for key in order:
            if cells[key] and len(out) < n:
                out.add(cells[key].pop())
    return out


def template_rows(queries: list[dict], author: str, half: str | None, common: int = 0, all_ordered: bool = False) -> list[dict]:
    keep = None
    ordered = list(queries)
    if half:
        which = {"1of2": 1, "2of2": 2}.get(half)
        if which is None:
            raise SystemExit("--half must be 1of2 or 2of2")
        assignment = split_halves(queries)
        own = {qid for qid, h in assignment.items() if h == which}
        shared = common_subset(queries, common) if common else set()
        keep = own | shared
        if all_ordered:
            # option B: everything, but common first, then the own half, then the other half (a clean fallback to option C)
            block = {q["query_id"]: (0 if q["query_id"] in shared else 1 if q["query_id"] in own else 2) for q in queries}
            ordered = sorted(queries, key=lambda q: block[q["query_id"]])
            keep = None
    rows = []
    for q in ordered:
        if keep is not None and q["query_id"] not in keep:
            continue
        # The narrative alone: no country, year or source, so the writer, the text systems and the judge read the same text.
        rows.append({"query_id": q["query_id"], "author_id": author, "narrative": q["free_text"],
                     "formula": "", "not_expressible": "", "minutes": "", "refinements": "", "notes": ""})
    return rows


# ---------------------------------------------------------------------------
# Merge and report
# ---------------------------------------------------------------------------
def merge(queries: list[dict], sheet: list[dict], resolver: RedirectResolver | None,
          catalog_by_key: "CatalogLookup | None", incoming: "IncomingRedirects | None" = None) -> tuple[list[dict], list[dict]]:
    by_id = {q["query_id"]: q for q in queries}
    if resolver:
        titles = []
        for row in sheet:
            try:
                titles += list(formula_and_titles(row)[2].values())
            except (ValueError, FormulaError):
                pass
        resolver.resolve(titles)
        if incoming is not None:
            incoming.fetch([(resolver.get(t) or {}).get("resolved") or t for t in titles])
    report: list[dict] = []
    for i, row in enumerate(sheet, 2):  # line 2 is the first data row (line 1 is the header)
        qid = (row.get("query_id") or "").strip()
        author = (row.get("author_id") or "").strip()
        if not (row.get("formula") or "").strip() and not (row.get("entities") or "").strip():
            report.append({"line": i, "query_id": qid, "author_id": author, "status": "empty row, skipped", "details": []})
            continue
        if qid not in by_id:
            report.append({"line": i, "query_id": qid, "author_id": author, "status": "REJECTED", "details": ["unknown query_id"]})
            continue
        if not author:
            report.append({"line": i, "query_id": qid, "author_id": author, "status": "REJECTED", "details": ["empty author_id"]})
            continue
        rec, errors, warnings = row_to_formula(row, resolver, catalog_by_key, incoming)
        if rec is None:
            report.append({"line": i, "query_id": qid, "author_id": author, "status": "REJECTED", "details": errors})
            continue
        probs = check_formula(rec)  # the query file's own check, for consistency with every consumer
        q = by_id[qid]
        sheet_text = " ".join((row.get("narrative") or row.get("free_text") or "").split())
        if sheet_text and sheet_text != " ".join(q["free_text"].split()):
            warnings.append("the narrative in the sheet differs from the query file; the sheet's text is ignored")
        fs = [f for f in (q.get("boolean_formulas") or []) if f.get("author_id") != author]
        replaced = len(fs) != len(q.get("boolean_formulas") or [])
        fs.append(rec)
        q["boolean_formulas"] = fs
        report.append({"line": i, "query_id": qid, "author_id": author,
                       "status": "imported (replaced the earlier formula)" if replaced else "imported",
                       "details": warnings + [f"check: {p}" for p in probs]})
    return queries, report


def write_report(path: Path, report: list[dict], sheet: Path) -> None:
    n_ok = sum(1 for r in report if r["status"].startswith("imported"))
    n_rej = sum(1 for r in report if r["status"] == "REJECTED")
    lines = [f"# Import of {sheet.name}", "", f"rows: {len(report)}; imported: {n_ok}; rejected: {n_rej}", "",
             "Lines marked [archive] come from the archive catalog and are never shown to a formula writer.", ""]
    for r in report:
        lines.append(f"- line {r['line']}, {r['query_id']} / {r['author_id']}: **{r['status']}**")
        for d in r["details"]:
            lines.append(f"    - {d}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def assign_main(queries: list[dict], authors: list[str]) -> tuple[int, list[str]]:
    """authors = [writer of half 1, writer of half 2]. For every query, the formula of the
    assigned writer becomes the 'main' formula (a copy with author_id 'main' and
    copied_from). Returns (number assigned, ids of queries whose assigned writer has no
    formula yet)."""
    assignment = split_halves(queries)
    n, missing = 0, []
    for q in queries:
        writer = authors[assignment[q["query_id"]] - 1]
        fs = [f for f in (q.get("boolean_formulas") or []) if f.get("author_id") != MAIN_AUTHOR]
        src = next((f for f in fs if f.get("author_id") == writer), None)
        if src is None:
            missing.append(q["query_id"])
            q["boolean_formulas"] = fs
            continue
        main = dict(src)
        main["author_id"] = MAIN_AUTHOR
        main["copied_from"] = writer
        q["boolean_formulas"] = fs + [main]
        n += 1
    return n, missing


def update_exclusions(path: Path, annotator: str, query_ids: list[str]) -> None:
    """exclusions.json for pool/build_pool.py: annotator id -> query ids they must not judge."""
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[annotator] = sorted(set(data.get(annotator, [])) | set(query_ids))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("template")
    t.add_argument("--queries", required=True)
    t.add_argument("--out", required=True, help=".xlsx (Excel) or .csv")
    t.add_argument("--author", required=True, help="the author id written into every row, e.g. annotator_a")
    t.add_argument("--half", default=None, help="1of2 or 2of2: only one half of the queries, split inside every (country, year) cell")
    t.add_argument("--common", type=int, default=0, help="with --half: also the N queries of the common subset both writers formulate")
    t.add_argument("--all-ordered", action="store_true", help="with --half and --common: every query, ordered common, own half, other half (option B)")
    am = sub.add_parser("assign-main", help="copy the assigned writer's formula of every query as author 'main'")
    am.add_argument("--queries", required=True)
    am.add_argument("--out", required=True)
    am.add_argument("--authors", required=True, help="writer of half 1, writer of half 2, e.g. annotator_a,annotator_b")
    m = sub.add_parser("import")
    m.add_argument("--queries", required=True)
    m.add_argument("--sheet", required=True, help=".xlsx or .csv")
    m.add_argument("--out", required=True)
    m.add_argument("--resolve-redirects", action="store_true", help="ask the English Wikipedia API for canonical titles (cached)")
    m.add_argument("--redirect-cache", default=None, help="JSON cache of the answers (default: wikipedia_redirects.json next to --out)")
    m.add_argument("--incoming-cache", default=None, help="JSON cache of the names that redirect to each page (default: wikipedia_incoming_redirects.json next to the redirect cache)")
    m.add_argument("--catalog", default=None, help="WORK/stats/entity_catalog.csv: archive counts in the report, for the person who runs the import only")
    m.add_argument("--exclusions", default=None, help="exclusions.json for the pool: record that --exclude-as must not judge these queries")
    m.add_argument("--exclude-as", default=None, help="the annotator id of this sheet's author")
    m.add_argument("--report", default=None, help="markdown report (default: next to --out)")
    args = ap.parse_args()
    queries = load_queries(args.queries)
    if args.cmd == "template":
        if args.all_ordered and not (args.half and args.common):
            raise SystemExit("--all-ordered needs --half and --common")
        rows = template_rows(queries, args.author, args.half, args.common, args.all_ordered)
        write_sheet(Path(args.out), rows)
        print(f"template with {len(rows)} rows for {args.author}{' (' + args.half + ')' if args.half else ''}"
              f"{' + common ' + str(args.common) if args.common else ''}{' , all, ordered' if args.all_ordered else ''} -> {args.out}")
        return
    if args.cmd == "assign-main":
        authors = [a.strip() for a in args.authors.split(",") if a.strip()]
        if len(authors) != 2:
            raise SystemExit("--authors needs exactly two writers")
        n, missing = assign_main(queries, authors)
        write_json(Path(args.out), queries)
        print(f"main formula assigned for {n} queries -> {args.out}" + (f"; {len(missing)} queries without the assigned writer's formula yet: {', '.join(missing[:10])}" if missing else ""))
        sys.exit(1 if missing else 0)
    out = Path(args.out)
    resolver = None
    if args.resolve_redirects or args.redirect_cache:
        cache = Path(args.redirect_cache) if args.redirect_cache else out.parent / "wikipedia_redirects.json"
        resolver = RedirectResolver(cache, online=args.resolve_redirects)
        inc = Path(args.incoming_cache) if args.incoming_cache else cache.parent / "wikipedia_incoming_redirects.json"
        incoming = IncomingRedirects(inc, online=args.resolve_redirects)
    else:
        incoming = None
    catalog_by_key = CatalogLookup(load_catalog(Path(args.catalog))) if args.catalog else None
    sheet = Path(args.sheet)
    queries, report = merge(queries, read_sheet(sheet), resolver, catalog_by_key, incoming)
    write_json(out, queries)
    rep_path = Path(args.report) if args.report else out.parent / f"{sheet.stem}.import_report.md"  # one report per sheet
    write_report(rep_path, report, sheet)
    imported = [r for r in report if r["status"].startswith("imported")]
    n_rej = sum(1 for r in report if r["status"] == "REJECTED")
    if args.exclusions and args.exclude_as:
        update_exclusions(Path(args.exclusions), args.exclude_as, [r["query_id"] for r in imported])
    print(f"imported {len(imported)} formulas, rejected {n_rej} -> {out}; report {rep_path}")
    for r in report:
        if r["status"] == "REJECTED":
            print(f"  REJECTED line {r['line']} ({r['query_id']} / {r['author_id']}): " + "; ".join(r["details"]))
    sys.exit(1 if n_rej else 0)


if __name__ == "__main__":
    main()
