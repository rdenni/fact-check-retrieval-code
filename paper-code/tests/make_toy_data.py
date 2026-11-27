"""
make_toy_data.py -- build a tiny fake CimpleKG store and a tiny query file so
the whole paper pipeline can be run end to end on any machine in a minute,
without the real data and without the real models.

Usage:
  python tests/make_toy_data.py --out tests/toy
Creates:
  tests/toy/store/           an Oxigraph store with N fake fact-checks
  tests/toy/queries_v6.json  3 queries with formulas by two writers and a consensus
  tests/toy/truth.json       which fake documents are 'relevant' (for the fake annotators)

Nothing here is real data. The entity IRIs are fake DBpedia-like IRIs.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import kg  # noqa: E402

TOPICS = {
    "vaccine": {
        "words": ["vaccine", "covid", "microchip", "5g", "magnet", "spike", "shedding"],
        "entities": ["Vaccine", "COVID-19", "5G", "Microchip", "Bill_Gates"],
    },
    "ukraine": {
        "words": ["ukraine", "nato", "biolab", "kyiv", "russia", "nazi", "weapon"],
        "entities": ["Ukraine", "NATO", "Russia", "Biological_weapon", "Kyiv"],
    },
    "climate": {
        "words": ["climate", "co2", "warming", "plant", "ice", "sun", "greening"],
        "entities": ["Climate_change", "Carbon_dioxide", "Sun", "Antarctica", "Plant"],
    },
    "other": {
        "words": ["election", "ballot", "tax", "migrant", "bridge", "school", "flood"],
        "entities": ["Election", "Tax", "Immigration", "Flood", "School"],
    },
}
DB = "http://dbpedia.org/resource/"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="tests/toy")
    ap.add_argument("--n-docs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "store").mkdir(parents=True)

    from pyoxigraph import Store, NamedNode, Literal, Quad
    store = Store(str(out / "store"))

    def add(s, p, o):
        store.add(Quad(NamedNode(s), NamedNode(p), o if isinstance(o, (NamedNode, Literal)) else NamedNode(o)))

    orgs = ["Snopes", "PolitiFact", "Boom", "AFP Fact Check", "Full Fact"]
    org_urls = ["https://www.snopes.com", "http://www.politifact.com", "http://www.boomlive.in", "https://factcheck.afp.com/", "https://fullfact.org"]
    for i, org in enumerate(orgs):
        add(f"http://data.cimple.eu/organization/{i}", kg.P_NAME, Literal(org))
        add(f"http://data.cimple.eu/organization/{i}", kg.P_URL, Literal(org_urls[i]))

    truth: dict[str, list[str]] = {"q1": [], "q2": [], "q3": []}
    docs = []
    for i in range(args.n_docs):
        topic = rng.choices(list(TOPICS), weights=[3, 3, 3, 4])[0]
        t = TOPICS[topic]
        words = rng.sample(t["words"], 3)
        claim_text = f"Claim {i}: {' '.join(words)} {rng.choice(['is true', 'was staged', 'causes harm', 'never happened'])}."
        cr = f"http://data.cimple.eu/claim-review/{i:04d}"
        claim = f"http://data.cimple.eu/claim/{i:04d}"
        add(cr, kg.P_TYPE, kg.C_CLAIMREVIEW)
        add(cr, kg.P_ITEM_REVIEWED, claim)
        add(claim, kg.P_TEXT, Literal(claim_text))
        if rng.random() < 0.9:  # an explanation text on most fact-checks, with more topic words
            add(cr, kg.P_TEXT, Literal(f"Fact check explanation {i}. " + " ".join(rng.sample(t["words"], 4)) + ". " + " ".join(rng.sample(TOPICS["other"]["words"], 2)) + "."))
        add(cr, kg.P_LANG, Literal("English"))
        add(cr, kg.P_AUTHOR, f"http://data.cimple.eu/organization/{rng.randrange(len(orgs))}")
        add(cr, kg.P_RATING, "http://data.cimple.eu/ontology#not_credible")
        if rng.random() < 0.9:  # some dates missing; many ties on purpose
            date = f"{rng.choice([2020, 2021, 2022, 2023, 2024])}-{rng.choice(['01', '06', '11'])}-{rng.choice(['05', '15'])}"
            add(cr, kg.P_DATE, Literal(date))
        # Claim-node entities: 1 to 3 from the topic; CR-node entities: those plus noise from other topics.
        claim_ents = rng.sample(t["entities"], rng.randint(1, 3)) if rng.random() < 0.85 else []
        for e in claim_ents:
            add(claim, kg.P_MENTIONS, DB + e)
        cr_ents = set(claim_ents) | set(rng.sample(TOPICS[rng.choice(list(TOPICS))]["entities"], 2))
        for e in cr_ents:
            add(cr, kg.P_MENTIONS, DB + e)
        docs.append((cr, topic, claim_text, claim_ents))
        # Fake ground truth: relevant if same topic and the claim shares 2 words with the query's words.
    qdefs = {
        "q1": ("vaccine", "COVID-19 vaccines contain microchips that are activated by 5G", ["vaccine", "microchip", "5g"],
               {"writer_a": ("e1 AND (e2 OR e3)", {"e1": "Vaccine", "e2": "Microchip", "e3": "5G"}),
                "writer_b": ("e1 AND e2", {"e1": "Vaccine", "e2": "COVID-19"}),
                "consensus": ("e1 AND (e2 OR e3 OR e4)", {"e1": "Vaccine", "e2": "Microchip", "e3": "5G", "e4": "COVID-19"})}),
        "q2": ("ukraine", "The United States runs secret biological weapons laboratories in Ukraine", ["ukraine", "biolab", "weapon"],
               {"writer_a": ("e1 AND e2", {"e1": "Ukraine", "e2": "Biological_weapon"}),
                "writer_b": ("(e1 OR e3) AND e2 AND NOT e4", {"e1": "Ukraine", "e2": "Biological_weapon", "e3": "Kyiv", "e4": "NATO"}),
                "consensus": ("(e1 OR e3) AND e2", {"e1": "Ukraine", "e2": "Biological_weapon", "e3": "Kyiv"})}),
        "q3": ("climate", "Carbon dioxide is plant food and more CO2 makes the planet greener", ["co2", "plant", "greening"],
               {"writer_a": ("e1 AND e2", {"e1": "Carbon_dioxide", "e2": "Plant"}),
                "writer_b": ("e1", {"e1": "Carbon_dioxide"}),
                "consensus": ("e1 AND (e2 OR e3)", {"e1": "Carbon_dioxide", "e2": "Plant", "e3": "Climate_change"})}),
    }
    for cr, topic, claim_text, _ in docs:
        for qid, (qt, _ft, qwords, _f) in qdefs.items():
            if topic == qt and sum(1 for w in qwords if w in claim_text) >= 2:
                truth[qid].append(cr)

    queries = []
    for qid, (topic, free_text, _w, formulas) in qdefs.items():
        bfs = []
        for author, (formula, ents) in formulas.items():
            bfs.append({
                "author_id": author,
                "formula": formula,
                "entities": {k: {"label": v.replace("_", " "), "wikipedia_title": v.replace("_", " "),
                                 "wikidata_qid": None, "dbpedia_uri": DB + v} for k, v in ents.items()},
                "not_expressible": ["activated by 5G (a relation, not an entity)"] if author == "writer_a" and qid == "q1" else [],
                "minutes": rng.randint(5, 20),
                "refinements": rng.randint(0, 3),
            })
        queries.append({
            "query_id": qid,
            "free_text": free_text,
            "query_type": "narrative",
            "stratum": {"country": "toy", "year": 2024},
            "source": {"report": "toy", "url_or_doi": None, "date": "2024-01-01"},
            "filters": {"language": "English", "date_range": ["2024-01-01", "2024-12-31"]},
            "entity_groups": [],
            "boolean_formulas": bfs,
        })
    (out / "queries_v6.json").write_text(json.dumps(queries, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    store.flush()
    print(f"toy store: {len(store)} triples, {len(docs)} fact-checks -> {out}")
    print("relevant per query (fake truth):", {q: len(v) for q, v in truth.items()})


if __name__ == "__main__":
    main()
