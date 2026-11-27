"""
paper_facts.py -- every fact of the paper that is computed from our own data and is
not already a macro of plot_archive.py, allocate_queries.py, prepare_cells.py or
model_selection/pareto_simple.py: the numbers and the name lists of Sections 3
and 4 (the snapshot, the language rule, the countries, the duplicates, the Mars
links, the run settings and timings), written as LaTeX macros, plus a report that
states every claim of the text with its evidence and PASS or FAIL.

Rule: a number or a claim about our data that no script
reproduces does not go in the paper. This script is where such facts are made.

Inputs (all produced by the pipeline; nothing is typed by hand):
  WORK/corpus.jsonl, WORK/entities_claim.jsonl, WORK/entities_cr.jsonl  (run_all.py --steps prepare)
  WORK/stats/archive_stats.json, WORK/stats/publishers_map.csv           (archive_stats.py, publisher_map.py)
  <store parent>/snapshot_manifest.json, <store parent>/language_decisions.jsonl   (build_store.py)
  --mirror  the N-Triples mirror (the texts of the removed documents, to name their scripts)
  --mirror-manifest  its manifest (the number of triples in the file)
  --runs    the manifest.json of the text run (settings, hardware, timings)
  --queries, --pilot  the query files (how many queries; the replacements of the draw; how many rewrite proposals the
                      reviewer changed)
  --allocation        allocation.json of queries/allocate_queries.py (optional): the queries that sit in another year
                      than the allocation planned, and the years topped up (Supplementary Section S6)
Outputs:
  --out/facts_numbers.tex   \\newcommand macros (default latex/figures)
  WORK/stats/paper_facts.md the claims with their evidence, each PASS or FAIL
  WORK/stats/paper_facts.json the same, machine-readable
Exit code 1 when a claim FAILS: the text then says something the data does not.

Usage:
  python stats/paper_facts.py --work WORK --store STORE --mirror MIRROR.nt --mirror-manifest MIRROR_MANIFEST.json
         --runs RUNS/manifest.json --queries Q.json --pilot QP.json [--allocation allocation.json] [--out DIR]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import kg  # noqa: E402
from common.jsonl import read_jsonl, write_json  # noqa: E402
from snapshot import language_rule  # noqa: E402
from stats.plot_archive import COUNTRY_MIN_DOCS, DISPLAY, MONTHS, tex_macro  # noqa: E402

# Names of the languages langdetect returns (names only, for the text).
LANG_NAMES = {"it": "Italian", "es": "Spanish", "fr": "French", "sk": "Slovak", "ro": "Romanian", "hu": "Hungarian",
              "sv": "Swedish", "cs": "Czech", "da": "Danish", "de": "German", "pt": "Portuguese", "nl": "Dutch",
              "et": "Estonian", "id": "Indonesian", "tl": "Tagalog", "ml": "Malayalam", "ta": "Tamil", "te": "Telugu",
              "kn": "Kannada", "hi": "Hindi", "mr": "Marathi", "bn": "Bengali", "pa": "Punjabi", "gu": "Gujarati"}
LATIN_LANGS = {"it", "es", "fr", "sk", "ro", "hu", "sv", "cs", "da", "de", "pt", "nl", "et", "id", "tl"}
# Unicode script names (first word of a character's Unicode name) as the text writes them.
SCRIPT_NAMES = {"DEVANAGARI": "Devanagari", "BENGALI": "Bengali", "MALAYALAM": "Malayalam", "GURMUKHI": "Gurmukhi",
                "TELUGU": "Telugu", "TAMIL": "Tamil", "GUJARATI": "Gujarati", "ARABIC": "Arabic", "CYRILLIC": "Cyrillic",
                "CJK": "Chinese", "KANNADA": "Kannada", "ORIYA": "Odia", "HANGUL": "Korean", "THAI": "Thai"}
MIN_SHARE_LISTED = 0.01   # a publisher or a script is named in a list when it holds at least 1 % of the group
LATIN_PAIR_MIN = 20       # a (publisher, language) pair of Latin-script removals is named when it has at least 20 documents
SHARED_SHORT_CHARS = 1000  # rule 1: a shared explanation text under this many characters is a short one (see the claim check)
SHARED_ORG_MIN_SHARE = 0.10  # an organisation is named among the holders of the long shared texts from this share of them
EXAMPLE_SHORT_CLAIM = "Israel Nukes Beirut"   # the example of a short English claim the detector misreads
NT_LINE = re.compile(r'^<([^>]*)> <([^>]*)> (.*) \.\s*$')
NT_ESC = re.compile(r'\\(u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8}|[tbnrf"\'\\])')
SIMPLE_ESC = {"t": "\t", "b": "\b", "n": "\n", "r": "\r", "f": "\f", '"': '"', "'": "'", "\\": "\\"}


def nt_literal(obj: str) -> str | None:
    """The text of an N-Triples literal ("..."@en or "..."^^<type>), unescaped; None for an IRI."""
    if not obj.startswith('"'):
        return None
    end = obj.rfind('"')
    body = obj[1:end]
    return NT_ESC.sub(lambda m: chr(int(m.group(1)[1:], 16)) if m.group(1)[0] in "uU" else SIMPLE_ESC[m.group(1)], body)


def dominant_script(text: str | None) -> str | None:
    """The Unicode script of most of the letters of a text that are not Latin (None if there are none)."""
    c = Counter()
    for ch in text or "":
        if ch.isalpha():
            try:
                first = unicodedata.name(ch).split()[0]
            except ValueError:
                continue
            if first != "LATIN":
                c[first] += 1
    return c.most_common(1)[0][0] if c else None


def name_list(items: list[str]) -> str:
    """'a', 'a and b', 'a, b, and c' (with the serial comma, as the paper writes its lists)."""
    items = list(items)
    if not items:
        return "none"
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + ", and " + items[-1]


def date_words(iso: str) -> str:
    """'2026-10-02' -> '2 October 2026'."""
    return f"{int(iso[8:10])} {MONTHS[int(iso[5:7]) - 1]} {iso[:4]}"


def latex_escape(text: str) -> str:
    """Escape the characters LaTeX treats as commands in a generated name or sentence."""
    return re.sub(r"([&%$#_])", r"\\\1", text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--store", required=True, help="the store; snapshot_manifest.json and language_decisions.jsonl are read from its parent")
    ap.add_argument("--mirror", required=True)
    ap.add_argument("--mirror-manifest", required=True)
    ap.add_argument("--runs", required=True, help="manifest.json of the text run")
    ap.add_argument("--queries", required=True)
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--allocation", default=None, help="allocation.json (queries/allocate_queries.py --out-json): the planned "
                    "number of queries of every (country, year) cell")
    ap.add_argument("--out", default=None, help="default: <repository>/latex/figures")
    ap.add_argument("--allow-fail", action="store_true", help="exit 0 even when a claim fails (the toy test: toy data is not the archive)")
    args = ap.parse_args()
    work = Path(args.work)
    snap = Path(args.store).resolve().parent
    out = Path(args.out) if args.out else Path(__file__).resolve().parent.parent.parent / "latex" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    macros, claims = {}, []

    def claim(text: str, ok: bool, evidence: str):
        claims.append({"claim": text, "status": "PASS" if ok else "FAIL", "evidence": evidence})

    manifest = json.loads((snap / "snapshot_manifest.json").read_text(encoding="utf-8"))
    mm = json.loads(Path(args.mirror_manifest).read_text(encoding="utf-8"))
    stats = json.loads((work / "stats" / "archive_stats.json").read_text(encoding="utf-8"))
    pmap = {}
    with open(work / "stats" / "publishers_map.csv", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            pmap[r["publisher"]] = r
    org = lambda p: (pmap.get(p, {}).get("ifcn_name") or p)  # noqa: E731
    disp = lambda p: DISPLAY.get(org(p), org(p))  # noqa: E731
    country = lambda p: pmap.get(p, {}).get("country") or "unknown"  # noqa: E731

    # ---- the snapshot ----
    macros["nTriplesMirror"] = int(mm["triples"])
    macros["nTriplesStore"] = int(manifest["triples_loaded"])
    macros["mirrorDate"] = mm["date"][:10]
    macros["mirrorDateWords"] = date_words(mm["date"][:10])
    scrape = re.search(r">= (\d+)", manifest["rules"]["scraping_failures"])
    macros["scrapeMin"] = int(scrape.group(1)) if scrape else "n/a"
    claim("The snapshot file and the store agree on the number of fact-checks loaded",
          int(mm["n_fact_checks_listed"]) == int(manifest["claimreviews_loaded"]),
          f"mirror manifest lists {mm['n_fact_checks_listed']} fact-checks; store manifest loaded {manifest['claimreviews_loaded']}")

    # ---- the language rule: constants, calibration publishers, publishers' languages ----
    lr = manifest["language_rule"]
    macros["langRuleVersion"] = lr["rule_version"]
    macros["langNonLatinPct"] = int(round(100 * language_rule.NON_LATIN_MAX))
    macros["langMojibakePct"] = int(round(100 * language_rule.MOJIBAKE_MAX))
    macros["langDetectP"] = f"{language_rule.DETECT_P:.1f}"
    macros["langAgreeP"] = f"{language_rule.AGREE_P:.1f}"
    macros["langPublisherMin"] = language_rule.PUBLISHER_LANGUAGE_MIN
    macros["langSeed"] = language_rule.SEED
    try:
        from importlib.metadata import version
        macros["langdetectVersion"] = version("langdetect")
    except Exception:  # noqa: BLE001
        macros["langdetectVersion"] = "n/a"
    (_, short_max), (mid_min, mid_end), (long_min, _) = language_rule.CALIBRATION_BUCKETS
    macros.update({"calibShortMax": short_max, "calibMidMin": mid_min, "calibMidMax": mid_end - 1, "calibLongMin": long_min})
    calib = [disp(p) for p in language_rule.CALIBRATION_PUBLISHERS]
    macros["calibPublishers"] = name_list(calib)
    macros["nCalibPublishers"] = len(calib)
    by_lang_pub = defaultdict(list)
    for pub, langs in sorted(lr["publisher_languages"].items()):
        by_lang_pub[tuple(sorted(langs, key=lambda c: LANG_NAMES.get(c, c)))].append(disp(pub))
    parts = []
    for langs, pubs in sorted(by_lang_pub.items(), key=lambda kv: (-len(kv[0]), kv[1])):
        parts.append(f"{name_list([LANG_NAMES.get(c, c) for c in langs])} for {name_list(sorted(pubs))}")
    macros["publisherLanguages"] = "; ".join(parts)

    # The example of a short English claim that the detector misreads.
    lang, p = language_rule.detect(EXAMPLE_SHORT_CLAIM)
    corpus = list(read_jsonl(work / "corpus.jsonl"))
    in_archive = any((d.get("claim_text") or "").strip().rstrip(".").lower() == EXAMPLE_SHORT_CLAIM.lower() for d in corpus)
    macros["exampleShortClaim"] = EXAMPLE_SHORT_CLAIM
    macros["exampleShortClaimLang"] = LANG_NAMES.get(lang, lang)
    claim(f"The detector calls the English claim \"{EXAMPLE_SHORT_CLAIM}\" a non-English language with p >= {language_rule.DETECT_P}",
          lang != language_rule.ENGLISH and p >= language_rule.DETECT_P,
          f"langdetect (seed {language_rule.SEED}) says {lang} with p {p:.3f}; the claim is in the collection: {in_archive}")

    # ---- the language decisions: who and what was removed ----
    dec = list(read_jsonl(snap / "language_decisions.jsonl"))
    removed = [r for r in dec if not r["english"]]
    n_dec = len(dec)
    by_country = Counter(country(r["author_name"]) for r in dec)
    rem_country = Counter(country(r["author_name"]) for r in removed)
    macros["pctIndiaBeforeRule"] = 100.0 * by_country["India"] / n_dec
    macros["pctUSBeforeRule"] = 100.0 * by_country["United States"] / n_dec
    macros["pctRemovedIndia"] = 100.0 * rem_country["India"] / max(len(removed), 1)
    claim("Before the language rule India was the largest country; the removed documents were almost all Indian",
          by_country.most_common(1)[0][0] == "India" and rem_country["India"] / max(len(removed), 1) >= 0.8,
          f"before: {dict(by_country.most_common(3))} of {n_dec}; removed: {dict(rem_country.most_common(3))} of {len(removed)}")
    nl = [r for r in removed if r["reason"] == "non_latin_claim"]
    pubs = Counter(disp(r["author_name"]) for r in nl)
    listed = [p for p, n in pubs.most_common() if n >= MIN_SHARE_LISTED * len(nl)]
    macros["nonLatinClaimPublishers"] = name_list(listed)
    macros["pctNonLatinClaimListed"] = 100.0 * sum(pubs[p] for p in listed) / max(len(nl), 1)
    # Latin-script languages other than English, by publisher (conditions 4 to 6).
    pairs = Counter((disp(r["author_name"]), r["language"]) for r in removed
                    if r["reason"] in ("article_detected", "publisher_language", "claim_article_agree") and r["language"] in LATIN_LANGS)
    by_pub = defaultdict(list)
    for (pub, lg), n in pairs.items():
        if n >= LATIN_PAIR_MIN:
            by_pub[pub].append((n, lg))
    parts = []
    for pub, items in sorted(by_pub.items(), key=lambda kv: -sum(n for n, _ in kv[1])):
        items.sort(reverse=True)
        parts.append(f"{pub} in {name_list([LANG_NAMES.get(lg, lg) for _, lg in items])} ({sum(n for n, _ in items):,})")
    macros["latinNonEnglish"] = "; ".join(parts)
    fac = [r for r in pmap.values() if org(r["publisher"]) == "facta"]
    macros["ifcnLanguageFacta"] = fac[0]["language"] if fac else "n/a"
    claim("The IFCN signatory list gives English as Facta's language although Facta writes in Italian",
          bool(fac) and fac[0]["language"] == "English" and "facta" in lr["publisher_languages"]
          and "it" in lr["publisher_languages"]["facta"],
          f"publishers_map.csv language for facta: {fac[0]['language'] if fac else 'missing'}; rule's publisher languages: {lr['publisher_languages'].get('facta')}")

    # Scripts of the removed texts: one pass over the mirror for the claim and explanation texts of removed documents.
    want_cr = {r["cr_iri"]: r["reason"] for r in removed if r["reason"] in ("non_latin_claim", "non_latin_article", "mojibake")}
    cr_claim, texts = {}, {}
    item_reviewed, text_p = kg.P_ITEM_REVIEWED, f"<{kg.P_TEXT}>"
    # Pass 1: the claim of every removed fact-check. Pass 2: the texts of those fact-checks and claims only
    # (the file is 1.5 GB; keeping every explanation in memory is not needed).
    import hashlib
    distinct, n_lines = set(), 0
    with open(args.mirror, encoding="utf-8") as fh:
        for line in fh:
            if line.strip() and not line.startswith("#"):
                n_lines += 1
                distinct.add(hashlib.md5(line.strip().encode("utf-8")).digest())
            if "itemReviewed" not in line:
                continue
            m = NT_LINE.match(line)
            if m and m.group(2) == item_reviewed and m.group(1) in want_cr:
                o = m.group(3)
                cr_claim[m.group(1)] = o[1:-1] if o.startswith("<") else o
    claim("The store holds the distinct triples of the snapshot file (the file repeats some lines)",
          len(distinct) == int(manifest["triples_loaded"]) and n_lines == int(mm["triples"]),
          f"file: {n_lines} triple lines, {len(distinct)} distinct; mirror manifest {mm['triples']}; store loaded {manifest['triples_loaded']}")
    del distinct
    wanted = set(want_cr) | set(cr_claim.values())
    with open(args.mirror, encoding="utf-8") as fh:
        for line in fh:
            if text_p not in line:
                continue
            m = NT_LINE.match(line)
            if m and m.group(1) in wanted:
                texts[m.group(1)] = nt_literal(m.group(3))
    script_claim, script_article, mojibake_scripts, moj_pubs = Counter(), Counter(), Counter(), Counter()
    for cr, reason in want_cr.items():
        if reason == "non_latin_claim":
            script_claim[dominant_script(texts.get(cr_claim.get(cr)))] += 1
        elif reason == "non_latin_article":
            script_article[dominant_script(texts.get(cr))] += 1
        else:
            raw = texts.get(cr) or texts.get(cr_claim.get(cr)) or ""
            try:
                fixed = raw.encode("latin-1", errors="ignore").decode("utf-8", errors="ignore")
            except Exception:  # noqa: BLE001
                fixed = ""
            mojibake_scripts[dominant_script(fixed)] += 1
            moj_pubs[disp(next(r["author_name"] for r in removed if r["cr_iri"] == cr))] += 1
    n_sc = sum(script_claim.values())
    macros["nonLatinClaimScripts"] = name_list([SCRIPT_NAMES.get(s, str(s).title()) for s, n in script_claim.most_common()
                                                if s and n >= MIN_SHARE_LISTED * n_sc])
    macros["nonLatinArticleScripts"] = name_list([SCRIPT_NAMES.get(s, str(s).title()) for s, n in script_article.most_common()
                                                  if s and n >= MIN_SHARE_LISTED * sum(script_article.values())])
    top_moj = mojibake_scripts.most_common(1)[0] if mojibake_scripts else (None, 0)
    macros["mojibakeScript"] = SCRIPT_NAMES.get(top_moj[0], str(top_moj[0]).title())
    macros["mojibakePublishers"] = name_list([p for p, _ in moj_pubs.most_common()])
    claim("The wrong-encoding texts are Telugu decoded with the wrong encoding",
          top_moj[0] == "TELUGU" and top_moj[1] >= 0.9 * max(sum(mojibake_scripts.values()), 1),
          f"re-decoded (Latin-1 bytes read as UTF-8), main script: {dict(mojibake_scripts)}; publishers {dict(moj_pubs)}")
    claim("Every removed claim and explanation was found in the mirror",
          all(cr in cr_claim for cr, rs in want_cr.items() if rs == "non_latin_claim"),
          f"{sum(1 for cr in want_cr if cr in cr_claim)} of {len(want_cr)} removed documents found with their claim")

    # ---- rule 1: the explanation texts shared by scrapeMin or more fact-checks, read from the mirror ----
    # The rule is meant for scraping failures (an error message stored in place of the explanation), but it removes every
    # fact-check whose text is shared that widely, so the paper says what else it removes: the shared texts, split at
    # SHARED_SHORT_CHARS characters, with the organisations that hold the long ones.
    # One pass over the mirror. Of every explanation text only its checksum, its length and (when it is short) the text
    # itself are kept: the file is 1.5 GB.
    import hashlib
    cr_key, key_len, key_text, cr_org, org_name = {}, {}, {}, {}, {}
    cr_claim_iri, claim_node_text = {}, {}
    p_text, p_author, p_name, p_item = f"<{kg.P_TEXT}>", f"<{kg.P_AUTHOR}>", f"<{kg.P_NAME}>", f"<{kg.P_ITEM_REVIEWED}>"
    with open(args.mirror, encoding="utf-8") as fh:
        for line in fh:
            if ("schema.org/text>" not in line and "schema.org/author>" not in line and "schema.org/itemReviewed>" not in line
                    and "/organization/" not in line):
                continue
            m = NT_LINE.match(line)
            if not m:
                continue
            s, pred, o = m.group(1), f"<{m.group(2)}>", m.group(3)
            if pred == p_text:
                tx = nt_literal(o)
                if tx is None:
                    continue
                if "/claim-review/" in s:
                    k = hashlib.md5(tx.encode("utf-8")).digest()
                    cr_key[s] = k
                    if k not in key_len:
                        key_len[k] = len(tx)
                        key_text[k] = tx if len(tx) < SHARED_SHORT_CHARS else ""
                else:
                    claim_node_text[s] = tx
            elif "/claim-review/" in s and pred == p_author:
                cr_org[s] = o.strip("<>")
            elif "/claim-review/" in s and pred == p_item:
                cr_claim_iri[s] = o.strip("<>")
            elif "/organization/" in s and pred == p_name:
                org_name[s] = nt_literal(o)
    by_key = defaultdict(list)
    for cr, k in cr_key.items():
        by_key[k].append(cr)
    shared = {k: crs for k, crs in by_key.items() if len(crs) >= macros["scrapeMin"]}
    short = {k: crs for k, crs in shared.items() if key_len[k] < SHARED_SHORT_CHARS}
    long_ = {k: crs for k, crs in shared.items() if key_len[k] >= SHARED_SHORT_CHARS}
    n_fc = lambda g: sum(len(v) for v in g.values())  # noqa: E731
    macros["nSharedTexts"] = len(shared)
    macros["nSharedTextsFc"] = n_fc(shared)
    macros["sharedShortChars"] = f"{SHARED_SHORT_CHARS:,}"
    macros["nSharedShortTexts"] = len(short)
    macros["nSharedShortFc"] = n_fc(short)
    macros["nSharedLongTexts"] = len(long_)
    macros["nSharedLongFc"] = n_fc(long_)
    lens = sorted(key_len[k] for k in long_)
    macros["sharedLongMedianChars"] = f"{lens[len(lens) // 2]:,}" if lens else "n/a"
    macros["sharedLongMinChars"] = f"{lens[0]:,}" if lens else "n/a"
    macros["sharedShortMaxChars"] = f"{max((key_len[k] for k in short), default=0):,}"
    org_fc = Counter(disp(org_name.get(cr_org.get(cr), "?")) for crs in long_.values() for cr in crs)
    holders = [(o, n) for o, n in org_fc.most_common() if n >= SHARED_ORG_MIN_SHARE * max(n_fc(long_), 1)]
    macros["sharedLongHolders"] = name_list([o for o, _ in holders])
    macros["pctSharedLongHolders"] = 100.0 * sum(n for _, n in holders) / max(n_fc(long_), 1)
    macros["pctSharedLongOfLoaded"] = 100.0 * n_fc(long_) / int(manifest["claimreviews_loaded"])
    claim("The fact-checks that rule 1 removes are those whose explanation text is shared by scrapeMin or more fact-checks, as the store's manifest counts them",
          n_fc(shared) == int(manifest["removed"]["scraping_failures"]),
          f"mirror: {len(shared)} texts, {n_fc(shared)} fact-checks; store manifest removed {manifest['removed']['scraping_failures']}")
    claim("The shared texts split cleanly at SHARED_SHORT_CHARS characters (every short one has at most 500 characters, every long one at least 2,000)",
          all(key_len[k] <= 500 for k in short) and all(key_len[k] >= 2000 for k in long_),
          f"short texts: {len(short)}, the longest {max((key_len[k] for k in short), default=0)} characters; long texts: {len(long_)}, the shortest {lens[0] if lens else 'n/a'}")
    markers = {"connection errors": "Connection aborted", "a notice about advertisements": "online advertisements",
               "a donation banner": "Stand up for the facts", "the word None": None}
    found = {k: any((v in key_text[x] if v else key_text[x] == "None") for x in short) for k, v in markers.items()}
    claim("Among the short shared texts are connection errors, a notice about advertisements, a donation banner and the word None",
          all(found.values()), f"found: {found}; the short texts begin: " + "; ".join(repr(" ".join(key_text[x].split())[:40]) for x in sorted(short, key=lambda x: key_len[x])))
    claim("Most of the long shared texts are held by two or three organisations",
          bool(holders) and sum(n for _, n in holders) >= 0.8 * n_fc(long_),
          f"holders of the {n_fc(long_)} fact-checks with a long shared text: {dict(org_fc.most_common(4))}")
    per_group = [len({claim_node_text.get(cr_claim_iri.get(cr)) for cr in crs}) for crs in long_.values()]
    distinct = sum(per_group)
    claim("The fact-checks that share a long text are about different claims: CimpleKG gave each claim of an article its own fact-check",
          bool(per_group) and min(per_group) > 1 and distinct >= 0.9 * n_fc(long_),
          f"{distinct} different claim texts among the {n_fc(long_)} fact-checks of the {len(long_)} long shared texts; "
          f"the fewest different claim texts in one group: {min(per_group) if per_group else 'n/a'}")
    del cr_key, by_key, claim_node_text, cr_claim_iri

    # ---- IFCN status, the stopped feeds (from plot_archive.py's figures.json), the organisations with two names ----
    status = {}
    for r in pmap.values():
        status[org(r["publisher"])] = r.get("ifcn_status", "")
    macros["nExpiredSignatories"] = sum(1 for s in status.values() if s == "expired")
    macros["nOrganisations"] = len(status)
    fj = out / "figures.json"
    behind = json.loads(fj.read_text(encoding="utf-8"))["behind_the_figures"] if fj.exists() else {}
    stopped = [o for o, _, _ in behind.get("stopped_feeds", [])]
    still = [o for o in stopped if status.get(o) == "verified"]
    macros["stoppedStillVerified"] = name_list([DISPLAY.get(o, o) for o in still])
    claim("Some organisations whose feed stopped are still verified IFCN signatories",
          bool(still), f"stopped feeds {stopped}; IFCN status in the signatory list: {[(o, status.get(o)) for o in stopped]}")
    # The fall in the number of documents per year after the peak year, split into the organisations no longer collected and
    # the others (the last year of the archive is not complete, so the comparison ends in the year before it).
    per_year, per_year_stopped = Counter(), Counter()
    for d in corpus:
        y = (d.get("date_published") or "")[:4]
        if y.isdigit():
            per_year[int(y)] += 1
            if org(d.get("author_name") or "") in stopped:
                per_year_stopped[int(y)] += 1
    if per_year:
        last_full = max(per_year) - 1
        peak = max((y for y in per_year if y <= last_full), key=lambda y: per_year[y], default=last_full)
        fall = per_year[peak] - per_year[last_full]
        fall_stopped = per_year_stopped[peak] - per_year_stopped[last_full]
        claim("The fall in the number of documents in the last years is partly due to the organisations no longer collected",
              fall > 0 and fall_stopped > 0,
              f"documents per year: {peak} (the peak) {per_year[peak]}, {last_full} (the last full year) {per_year[last_full]}, a fall of {fall}; "
              f"{fall_stopped} of the fall are documents of the organisations no longer collected, {fall - fall_stopped} of the others")
    names = dict((o, ns) for o, ns in behind.get("organisations_with_several_names", []))
    spans = {}
    with open(work / "stats" / "publishers.csv", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            spans[r["publisher"]] = r["last_month"]
    snopes = names.get("snopes.com", [])
    last_all = max(spans.values())
    claim("Snopes has two publisher names in the graph and only snopes.com continues to the archive's last month",
          sorted(snopes) == ["snopes.com", "snopescom"] and spans.get("snopes.com") == last_all and spans.get("snopescom", "") < last_all,
          f"names of the organisation snopes.com: {snopes}; last months: {[(n, spans.get(n)) for n in snopes]}; archive's last month {last_all}")

    # ---- countries, undated, early documents ----
    per_country = stats["per_country"]
    big = [c for c, v in per_country if v >= COUNTRY_MIN_DOCS]
    macros["countriesRest"] = name_list(big[3:])
    und = Counter()
    for p, v in stats["undated_by_publisher"]:
        und[disp(p)] += v
    n_und = sum(und.values())
    top2 = [p for p, _ in und.most_common(2)]
    macros["undatedTopPublishers"] = name_list(top2)
    macros["pctUndatedTop"] = 100.0 * sum(und[p] for p in top2) / max(n_und, 1)
    early = [d for d in corpus if (d.get("date_published") or "9999")[:4] < "2016"]
    early_pub = Counter(disp(d["author_name"]) for d in early)
    macros["nBeforeSixteen"] = len(early)
    macros["pctBeforeSixteen"] = 100.0 * len(early) / len(corpus) if corpus else 0.0
    macros["beforeSixteenTop"] = early_pub.most_common(1)[0][0] if early else "none"
    macros["pctBeforeSixteenTop"] = 100.0 * early_pub.most_common(1)[0][1] / len(early) if early else 0.0

    # ---- claim nodes shared by several fact-checks ----
    groups = defaultdict(list)
    for d in corpus:
        groups[d["claim_iri"]].append(d)
    shared = [g for g in groups.values() if len(g) > 1]
    macros["sharedClaimMin"] = min(len(g) for g in shared) if shared else 0
    macros["sharedClaimMax"] = max(len(g) for g in shared) if shared else 0
    same = sum(1 for g in shared if len({disp(d["author_name"]) for d in g}) == 1)
    macros["pctSharedSamePublisher"] = 100.0 * same / max(len(shared), 1)
    sp = Counter(disp(d["author_name"]) for g in shared for d in g)
    macros["sharedTopPublishers"] = name_list([p for p, _ in sp.most_common(3)])
    claim("The shared claim nodes are the same claim re-checked, mostly by the same publisher",
          same / max(len(shared), 1) >= 0.5,
          f"{len(shared)} shared claim nodes; {same} have a single publisher; top publishers {dict(sp.most_common(3))}")
    # The web addresses (schema:url of the ClaimReview) of the fact-checks of a group. Three kinds, in this order:
    # the same page under addresses that differ only in the scheme, in www or in a final slash; the same article (the last
    # part of the address) under another host or folder, as a copy on a test server or in an English clone folder;
    # all the other groups (a short address, an AMP page, or another page about the same claim).
    from urllib.parse import urlsplit
    from pyoxigraph import NamedNode, Store

    def page_key(u):
        s = urlsplit(u.strip())
        host = s.netloc.lower()
        return (host[4:] if host.startswith("www.") else host) + s.path.rstrip("/").lower() + ("?" + s.query if s.query else "")

    def article_slug(u):
        segs = [x for x in urlsplit(u.strip()).path.split("/") if x]
        return segs[-1].lower() if segs else ""
    need = {d["cr_iri"] for g in shared for d in g}
    addr = defaultdict(list)
    for q_ in Store.read_only(str(args.store)).quads_for_pattern(None, NamedNode(kg.P_URL), None):
        if q_.subject.value in need:
            addr[q_.subject.value].append(q_.object.value)
    kinds = Counter()
    for g in shared:
        us = [u for d in g for u in addr.get(d["cr_iri"], [])]
        if not us:
            continue
        slugs = {article_slug(u) for u in us}
        kinds["page" if len({page_key(u) for u in us}) == 1 else "slug" if len(slugs) == 1 and "" not in slugs else "other"] += 1
    n_addr = sum(kinds.values())
    macros["sharedPageSamePct"] = 100.0 * kinds["page"] / max(n_addr, 1)
    macros["sharedPageSlugPct"] = 100.0 * kinds["slug"] / max(n_addr, 1)
    macros["sharedPageOtherPct"] = 100.0 * kinds["other"] / max(n_addr, 1)
    claim("In most groups of fact-checks that share a claim node the fact-checks are the same page or the same article under other addresses",
          n_addr > 0 and (kinds["page"] + kinds["slug"]) / n_addr >= 0.5,
          f"{n_addr} groups with addresses: {kinds['page']} the same page (the addresses differ only in scheme, www or a final slash), "
          f"{kinds['slug']} the same article under another host or folder, {kinds['other']} other")

    # ---- the claim-level entities: one constant ----
    claim_ents = Counter()
    per_doc_max = 0
    for r in read_jsonl(work / "entities_claim.jsonl"):
        if r["entities"]:
            claim_ents.update(r["entities"])
            per_doc_max = max(per_doc_max, len(r["entities"]))
    top = claim_ents.most_common(1)[0][0] if claim_ents else ""
    macros["nDistinctClaimEntities"] = len(claim_ents)
    macros["claimEntityName"] = top.rsplit("/", 1)[-1].replace("_", " ")
    claim("Every document with a claim-level entity has exactly one, and it is the same for all",
          len(claim_ents) == 1 and per_doc_max == 1,
          f"distinct claim-level entities {len(claim_ents)} ({top}); most per document {per_doc_max}")

    # ---- the text run: settings, hardware, timings ----
    run = json.loads(Path(args.runs).read_text(encoding="utf-8"))
    prof = run["profile"]
    tim = run.get("timings_seconds") or {}
    macros.update({
        "runDate": run["created"][:10], "runDateWords": date_words(run["created"][:10]),
        "bmKOne": f"{run['bm25']['k1']}", "bmB": f"{run['bm25']['b']}", "bmImpl": run["bm25"]["implementation"],
        "channelDepth": run["channel_depth"], "outputDepth": run["output_depth"], "rrfK": run["rrf_k"],
        "booleanCap": int(run["boolean_cap"]), "maxTokens": prof.get("max_length", "n/a"), "modelDtype": prof.get("dtype", "n/a"),
        "embedderName": str(prof.get("embedder", "n/a")).split("/")[-1], "rerankerName": str(prof.get("reranker", "n/a")).split("/")[-1],
        "queryInstruction": prof.get("query_instruction", "n/a"),
        "gpuName": str(run.get("gpu") or "no GPU").replace("NVIDIA GeForce ", "NVIDIA "),
        "runPython": run.get("python", "n/a"), "runTorch": str(run.get("torch", "n/a")).split("+")[0],
        "runTransformers": run.get("transformers", "n/a"),
        "secondsBMTwentyFive": int(round(tim.get("bm25_build", 0) + tim.get("bm25_search", 0))),
        "minutesEmbed": int(round(tim.get("embed", 0) / 60)),
        "minutesRerank": int(round(tim.get("rerank", 0) / 60)),
    })
    plat = run.get("platform", "n/a")
    macros["runOS"] = "Windows 11" if plat.startswith("Windows-10-10.0.2") else plat  # Windows 11 reports itself as 10.0.22000 or later
    claim("The text run read the collection of this store", int(run["n_documents"]) == len(corpus),
          f"run manifest n_documents {run['n_documents']}; corpus {len(corpus)}")
    dense_meta = next((v for k, v in (run.get("step_meta") or {}).items() if k.startswith("dense_")), {})
    with_article = sum(1 for d in corpus if (d.get("article_text") or "").strip())
    macros["textMode"] = run.get("text_mode", "n/a")
    claim("The text run read the explanation alone (the text the graph's entities come from)", run.get("text_mode") == "article",
          f"run manifest text_mode {run.get('text_mode')}")
    claim("The text index holds exactly the documents with an explanation", dense_meta.get("n_docs") == with_article,
          f"embedded documents {dense_meta.get('n_docs')}; documents with an explanation {with_article}")
    claim("No filter was applied to any system", str(run.get("filters_applied", "")).startswith("none"),
          f"run manifest filters_applied: {run.get('filters_applied')}")

    # ---- the query files ----
    def qlist(path):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return d["queries"] if isinstance(d, dict) else d
    import hashlib
    q, qp = qlist(args.queries), qlist(args.pilot)
    macros["nQueriesFile"] = len(q)
    macros["nPilot"] = len(qp)
    ids, pids = {x["query_id"] for x in q}, {x["query_id"] for x in qp}
    claim("The pilot shares no query with the main set", not (ids & pids), f"{len(ids & pids)} shared query ids")
    plen = sorted(len((x.get("provenance") or {}).get("passage") or "") for x in q)
    macros["passageMedianChars"] = plen[len(plen) // 2] if plen else 0
    macros["passageMaxChars"] = plen[-1] if plen else 0
    # The text systems read only free_text. The run may have read another file with the same texts (the query file with
    # the writers' formulas added): then the file checksums differ but the queries the systems saw are the same.
    sha = hashlib.sha256(Path(args.queries).read_bytes()).hexdigest()
    run_sha, run_file = run.get("queries_sha256"), run.get("queries_file")
    if sha == run_sha:
        same, evidence = True, f"sha256 of --queries {sha[:12]} = run manifest"
    elif run_file and Path(run_file).exists() and hashlib.sha256(Path(run_file).read_bytes()).hexdigest() == run_sha:
        texts = {x["query_id"]: x.get("free_text") for x in q}
        run_texts = {x["query_id"]: x.get("free_text") for x in qlist(run_file)}
        differ = sorted(k for k in texts.keys() | run_texts.keys() if texts.get(k) != run_texts.get(k))
        same = not differ
        n_corr = len(run.get("corrections") or [])
        evidence = (f"the run read {Path(run_file).name} (sha256 {str(run_sha)[:12]}, as the run manifest records it"
                    + (f"; the manifest lists {n_corr} later corrections" if n_corr else "") + f"); its "
                    f"{len(run_texts)} query texts against the {len(texts)} of --queries: {len(differ)} differ"
                    + (f" ({', '.join(differ[:5])})" if differ else ""))
    else:
        same, evidence = False, (f"sha256 of --queries {sha[:12]}; run manifest {str(run_sha)[:12]}; the file the run "
                                 f"read ({run_file}) is missing or changed since the run")
    claim("The text run used these query texts", same, evidence)

    # ---- the draw: years topped up, replacements (Supplementary Section S6) ----
    # A query sits in the year of its stratum. A cell with fewer kept candidates than queries got its missing queries from
    # what the other years of its country had left (a top-up), and a replacement may come from the country instead of the
    # year; both move queries to another year, never to another country. Counted against the allocation, cell by cell:
    # the queries above a cell's planned number are the queries that sit in another year than planned.
    if args.allocation:
        alloc = json.loads(Path(args.allocation).read_text(encoding="utf-8"))
        planned = {(c, int(y)): n for c, ys in alloc["cells"].items() for y, n in ys.items()}
        drawn = Counter((x["stratum"]["country"], int(x["stratum"]["year"])) for x in q)
        cells = set(planned) | set(drawn)
        short = sorted(k for k in cells if drawn.get(k, 0) < planned.get(k, 0))
        over = {k: drawn.get(k, 0) - planned.get(k, 0) for k in cells if drawn.get(k, 0) > planned.get(k, 0)}
        macros["drawYearsMoved"] = sum(over.values())
        macros["drawCellsDiffer"] = len(short) + len(over)
        macros["drawYearsToppedUp"] = len(short)
        years_of = defaultdict(list)
        for c, y in short:
            years_of[c].append(str(y))
        groups = [f"{c} {name_list(ys)}" for c, ys in sorted(years_of.items())]
        macros["drawYearsToppedUpList"] = ("; ".join(groups[:-1]) + "; and " + groups[-1]) if len(groups) > 1 else (groups[0] if groups else "none")
        countries = {c for c, _ in cells}
        totals_equal = all(sum(n for (c, _), n in planned.items() if c == k) == sum(n for (c, _), n in drawn.items() if c == k) for k in countries)
        claim("The country totals of the drawn queries equal the allocation", totals_equal,
              f"{len(countries)} countries; {macros['drawCellsDiffer']} of {len(cells)} cells differ from the allocation")
        claim("No query was removed at the rewrite", len(q) == sum(planned.values()),
              f"{len(q)} queries in --queries; {sum(planned.values())} in the allocation")
    else:
        for k in ("drawYearsMoved", "drawCellsDiffer", "drawYearsToppedUp", "drawYearsToppedUpList"):
            macros[k] = "n/a"
    # A replaced query keeps its id and records the dropped candidates (provenance.replaces) and, in drawn_from, the rule
    # each one broke and where the new one was drawn from (its cell, or the whole country when the old query had filled a
    # short year or its cell had none left). A draw from the country may land in the year of the cell, so the count of
    # these draws (drawReplacedFromCountry) is not a count of queries in another year (drawYearsMoved is).
    items, n_draws, n_country = [], 0, 0
    for x in q:
        pv = x.get("provenance") or {}
        if not pv.get("replaces"):
            continue
        parts = [s for s in str(pv.get("drawn_from", "")).split("; ") if s.startswith("replacement of")]
        rules = [m.group(1) for s in parts for m in [re.search(r"\((D\d)\)", s)] if m]
        from_country = sum(1 for s in parts if "of its country" in s)
        n_draws += len(pv["replaces"])
        n_country += from_country
        items.append(f"{x['query_id']} ({', then '.join(rules) or '?'}{', drawn from the whole country' if from_country else ''})")
    macros["drawReplacedQueries"] = len(items)
    macros["drawReplacements"] = n_draws
    macros["drawReplacedFromCountry"] = n_country
    macros["drawReplacedList"] = name_list(items)
    # The rewrite: the local model's proposal (provenance.rewrite_proposal) and the closed model's
    # (provenance.rewrite_closed.free_text) stand beside the final text; a proposal is changed when the final text differs.
    local = [x for x in q if isinstance((x.get("provenance") or {}).get("rewrite_proposal"), str)]
    closed = [x for x in q if ((x.get("provenance") or {}).get("rewrite_closed") or {}).get("free_text")]
    macros["rewriteLocalProposals"] = len(local)
    macros["rewriteLocalChanged"] = sum(1 for x in local if x["free_text"] != x["provenance"]["rewrite_proposal"])
    macros["rewriteClosedProposals"] = len(closed)
    macros["rewriteClosedChanged"] = sum(1 for x in closed if x["free_text"] != x["provenance"]["rewrite_closed"]["free_text"])

    lines = ["% generated by paper-code/stats/paper_facts.py; do not edit by hand"] + [
        tex_macro(k, latex_escape(v) if isinstance(v, str) else v) for k, v in macros.items()]
    (out / "facts_numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(work / "stats" / "paper_facts.json", {"macros": macros, "claims": claims})
    md = ["# The paper's own facts and their evidence (stats/paper_facts.py)", "",
          "| status | claim | evidence |", "|---|---|---|"]
    md += [f"| {c['status']} | {c['claim']} | {c['evidence']} |" for c in claims]
    md += ["", "## Macros", ""] + [f"- \\{k} = {v}" for k, v in macros.items()]
    (work / "stats" / "paper_facts.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    bad = [c for c in claims if c["status"] == "FAIL"]
    print(f"facts_numbers.tex -> {out} ({len(macros)} macros); claims: {len(claims) - len(bad)} PASS, {len(bad)} FAIL")
    for c in bad:
        print("FAIL:", c["claim"], "|", c["evidence"])
    sys.exit(1 if bad and not args.allow_fail else 0)


if __name__ == "__main__":
    main()
