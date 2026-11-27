"""
language_rule.py -- decides, for every fact-check of the store, whether it is
English, from its claim text and its explanation text (rule version 2026-10-07); its constants are below, with
their reasons.

Why a rule at all: CimpleKG marks every fact-check with a language, and the
mirror takes the ones marked "English". On the October 2026 snapshot that label
is wrong for about a third of them: Hindi, Bengali, Malayalam, Punjabi, Telugu,
Tamil, Chinese... written in their own scripts (28 percent), Italian, Spanish,
French, Slovak, Romanian, Hungarian, Swedish, Danish written in Latin letters
(3 percent), English claims whose fact-check explanation is in an Indic script
(1 percent), and a few texts whose bytes were decoded with the wrong encoding.
A test collection of English narratives cannot contain documents that no
annotator can read, so they are removed from the store, and the paper says so
with these numbers.

The rule, applied in this order; the first condition that holds decides:
  1. non_latin_claim     more than NON_LATIN_MAX of the letters of the claim are not Latin letters;
  2. non_latin_article   the same for the explanation text (an English viral post fact-checked in Hindi);
  3. mojibake            more than MOJIBAKE_MAX of the non-space characters of the claim or of the
                         first DETECT_CHARS of the explanation are in U+0080..U+00FF: UTF-8 bytes read
                         as Latin-1 (72 Telugu explanations of one publisher), unreadable;
  4. article_detected    the language detector calls the explanation a non-English language with
                         probability >= DETECT_P (the explanation is what a judge reads and what the
                         text systems index in explanation mode, the paper's main runs);
  5. publisher_language  the detector calls the claim a non-English language L with probability
                         >= DETECT_P, and L is a language the same publisher demonstrably writes
                         in: at least PUBLISHER_LANGUAGE_MIN of its documents have claim AND
                         explanation both detected as L with probability >= DETECT_P (Lead Stories'
                         Spanish, Slovak, Romanian, Hungarian, Czech editions; facta's Italian);
  6. claim_article_agree the detector calls claim and explanation the same non-English language,
                         each with probability >= AGREE_P (two weaker signals that agree);
  otherwise the document is English.
Why the detector is not trusted alone on a claim: on seven publishers that are
surely English it calls a claim non-English with probability >= 0.9 in 0.02
percent of claims of 60 or more characters but in 10 percent of claims under
30 ("Israel Nukes Beirut" is called Estonian); conditions 5 and 6 are what
survives that test.

The detector is langdetect 1.0.9 (a port of Google's language-detection
library) with DetectorFactory.seed = SEED, which makes it deterministic: the
same text always gets the same answer (checked on 2,000 re-detected claims,
0 differences). Texts are cut to DETECT_CHARS characters.

Used by snapshot/build_store.py (which removes the non-English fact-checks and
writes language_decisions.jsonl next to the manifest) and runnable alone on a
corpus file to audit or reproduce the decisions:

  python snapshot/language_rule.py --corpus WORK/corpus.jsonl --out WORK/stats/language_decisions.jsonl

Output, one line per document: cr_iri, author_name, claim_non_latin,
article_non_latin, mojibake, claim_lang, claim_p, article_lang, article_p,
english (true/false), reason (one of the six names above or null), language
(the detected language behind reasons 4 to 6, or null).
"""
from __future__ import annotations

import argparse
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.jsonl import read_jsonl, write_json, write_jsonl  # noqa: E402

VERSION = "2026-10-07"
NON_LATIN_MAX = 0.5          # a claim or explanation is in another script when over half of its letters are not Latin
MOJIBAKE_MAX = 0.3           # over 30 percent of non-space characters in U+0080..U+00FF never happens in real text (accents stay under 10 percent)
DETECT_P = 0.9               # the detector's probability needed on its own
AGREE_P = 0.7                # the probability needed from each side when claim and explanation agree
PUBLISHER_LANGUAGE_MIN = 20  # documents with claim and explanation agreeing on L before L counts as a language of the publisher
DETECT_CHARS = 1000          # the detector reads at most this many characters of a text
SEED = 0                     # langdetect's random seed; fixed so that the decisions are reproducible
ENGLISH = "en"
UNKNOWN = "unk"              # the detector failed (no letters, too short)
NONE = "none"                # no explanation text

REASONS = ("non_latin_claim", "non_latin_article", "mojibake", "article_detected", "publisher_language", "claim_article_agree")

# Publishers that write only in English, used to measure the detector's false-alarm rate on claims
# (the "calibration" of the summary): how often it calls one of their claims non-English with p >= DETECT_P,
# by claim length. This is the evidence behind trusting the detector only with corroboration.
CALIBRATION_PUBLISHERS = ("snopescom", "politifact", "full-fact", "check-your-fact", "factcheckorg",
                          "the-washington-post-fact-checker", "australian-associated-press")
CALIBRATION_BUCKETS = ((0, 30), (30, 60), (60, 10**9))  # claim length in characters: under 30, 30 to 59, 60 and more


def non_latin_share(text: str | None) -> float | None:
    """Share of non-Latin letters among all letters of the text; None when the text has no letters.
    A letter is any alphabetic character; it is Latin when its Unicode name starts with LATIN (so accented
    letters and A-Z a-z are Latin; Devanagari, Bengali, Arabic, Cyrillic, Chinese characters... are not)."""
    latin = other = 0
    for ch in text or "":
        if "A" <= ch <= "Z" or "a" <= ch <= "z":
            latin += 1
        elif ch.isalpha():
            try:
                name = unicodedata.name(ch)
            except ValueError:
                name = ""
            if name.startswith("LATIN"):
                latin += 1
            else:
                other += 1
    total = latin + other
    return other / total if total else None


def mojibake_share(text: str | None) -> float:
    """Share of characters in U+0080..U+00FF among the non-space characters of the text."""
    t = text or ""
    n = sum(1 for ch in t if not ch.isspace())
    return (sum(1 for ch in t if 0x80 <= ord(ch) <= 0xFF) / n) if n else 0.0


_detector_ready = False


def _prepare_detector():
    global _detector_ready
    if not _detector_ready:
        from langdetect import DetectorFactory
        DetectorFactory.seed = SEED
        _detector_ready = True


def detect(text: str | None) -> tuple[str, float]:
    """(language code, probability) of the most likely language of the first DETECT_CHARS characters,
    or (UNKNOWN, 0.0) when the detector cannot decide."""
    if not text or not text.strip():
        return UNKNOWN, 0.0
    _prepare_detector()
    from langdetect import detect_langs
    try:
        best = detect_langs(text[:DETECT_CHARS])[0]
        return best.lang, round(best.prob, 3)
    except Exception:
        return UNKNOWN, 0.0


def features(rec: dict) -> dict:
    """The measurements of one document; the detector runs only when the script and encoding checks pass."""
    claim, explanation = rec.get("claim_text") or "", rec.get("article_text") or None
    row = {"cr_iri": rec["cr_iri"], "author_name": rec.get("author_name"), "claim_chars": len(claim),
           "claim_non_latin": non_latin_share(claim), "article_non_latin": non_latin_share(explanation),
           "mojibake": round(max(mojibake_share(claim), mojibake_share(explanation[:DETECT_CHARS] if explanation else None)), 3),
           "claim_lang": None, "claim_p": None, "article_lang": None, "article_p": None}
    if (row["claim_non_latin"] or 0) > NON_LATIN_MAX or (row["article_non_latin"] or 0) > NON_LATIN_MAX or row["mojibake"] > MOJIBAKE_MAX:
        return row
    row["claim_lang"], row["claim_p"] = detect(claim)
    row["article_lang"], row["article_p"] = detect(explanation) if explanation else (NONE, 0.0)
    return row


def publisher_languages(rows: Iterable[dict]) -> dict[str, set[str]]:
    """For every publisher, the non-English languages in which at least PUBLISHER_LANGUAGE_MIN of its
    documents have claim and explanation both detected with probability >= DETECT_P."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        cl, al = r.get("claim_lang"), r.get("article_lang")
        if cl and cl not in (ENGLISH, UNKNOWN) and (r["claim_p"] or 0) >= DETECT_P and al == cl and (r["article_p"] or 0) >= DETECT_P:
            counts[r["author_name"] or ""][cl] += 1
    return {pub: {lang for lang, n in c.items() if n >= PUBLISHER_LANGUAGE_MIN} for pub, c in counts.items()}


def decision(row: dict, pub_langs: dict[str, set[str]]) -> tuple[str | None, str | None]:
    """(reason, language) when the document is not English, (None, None) when it is."""
    if (row["claim_non_latin"] or 0) > NON_LATIN_MAX:
        return "non_latin_claim", None
    if (row["article_non_latin"] or 0) > NON_LATIN_MAX:
        return "non_latin_article", None
    if row["mojibake"] > MOJIBAKE_MAX:
        return "mojibake", None
    cl, cp, al, ap = row["claim_lang"], row["claim_p"] or 0, row["article_lang"], row["article_p"] or 0
    if al not in (ENGLISH, UNKNOWN, NONE, None) and ap >= DETECT_P:
        return "article_detected", al
    if cl not in (ENGLISH, UNKNOWN, None) and cp >= DETECT_P and cl in pub_langs.get(row["author_name"] or "", ()):
        return "publisher_language", cl
    if cl not in (ENGLISH, UNKNOWN, None) and cp >= AGREE_P and al == cl and ap >= AGREE_P:
        return "claim_article_agree", cl
    return None, None


def decide(records: Iterable[dict], progress: bool = False) -> list[dict]:
    """Two passes over the documents (dicts with cr_iri, claim_text, article_text, author_name): measure
    everything, learn the publishers' languages, decide. Returns one row per document, in input order."""
    rows = []
    for i, rec in enumerate(records, 1):
        rows.append(features(rec))
        if progress and i % 20000 == 0:
            print(f"  language rule: {i:,} documents measured", flush=True)
    pub_langs = publisher_languages(rows)
    for r in rows:
        reason, lang = decision(r, pub_langs)
        r["english"] = reason is None
        r["reason"] = reason
        r["language"] = lang
    return rows


def calibration(rows: list[dict]) -> dict:
    """The detector's false-alarm rate on the claims of the surely-English publishers, by claim length:
    {bucket: {"claims": n, "called_non_english": k, "rate_percent": 100 k / n}}."""
    out = {}
    for lo, hi in CALIBRATION_BUCKETS:
        n = k = 0
        for r in rows:
            if (r.get("author_name") or "") in CALIBRATION_PUBLISHERS and r.get("claim_lang") and lo <= r.get("claim_chars", 0) < hi:
                n += 1
                if r["claim_lang"] not in (ENGLISH, UNKNOWN) and (r["claim_p"] or 0) >= DETECT_P:
                    k += 1
        label = f"under {hi}" if lo == 0 else (f"{lo} and more" if hi >= 10**9 else f"{lo} to {hi - 1}")
        out[label] = {"claims": n, "called_non_english": k, "rate_percent": round(100.0 * k / n, 2) if n else None}
    return out


def summarise(rows: list[dict]) -> dict:
    pub_langs = publisher_languages(rows)
    dropped = [r for r in rows if not r["english"]]
    return {"rule_version": VERSION, "documents": len(rows), "english": len(rows) - len(dropped), "not_english": len(dropped),
            "calibration": {"publishers": list(CALIBRATION_PUBLISHERS), "false_alarms_by_claim_length": calibration(rows)},
            "by_reason": {k: v for k, v in Counter(r["reason"] for r in dropped).most_common()},
            "by_language": {k: v for k, v in Counter(r["language"] for r in dropped if r["language"]).most_common()},
            "by_publisher": {k: v for k, v in Counter(r["author_name"] or "" for r in dropped).most_common()},
            "publisher_languages": {p: sorted(ls) for p, ls in sorted(pub_langs.items()) if ls},
            "constants": {"NON_LATIN_MAX": NON_LATIN_MAX, "MOJIBAKE_MAX": MOJIBAKE_MAX, "DETECT_P": DETECT_P, "AGREE_P": AGREE_P,
                          "PUBLISHER_LANGUAGE_MIN": PUBLISHER_LANGUAGE_MIN, "DETECT_CHARS": DETECT_CHARS, "SEED": SEED}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, help="corpus.jsonl (run_all.py --steps prepare)")
    ap.add_argument("--out", required=True, help="decisions JSONL; a .summary.json is written next to it")
    args = ap.parse_args()
    rows = decide(read_jsonl(args.corpus), progress=True)
    write_jsonl(args.out, rows)
    s = summarise(rows)
    write_json(Path(args.out).with_suffix(".summary.json"), s)
    print(f"language rule {VERSION}: {s['documents']:,} documents, {s['english']:,} English, {s['not_english']:,} not English; "
          f"by reason {s['by_reason']}; publisher languages {s['publisher_languages']} -> {args.out}")


if __name__ == "__main__":
    main()
