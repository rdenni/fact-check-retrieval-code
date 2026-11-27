# How well does entity search find fact-checks? The test collection

This is the test collection of the paper "How well does entity search find fact-checks? A test collection and an
evaluation on CimpleKG". It holds 200 queries written from reports of observers of disinformation, the identifiers of
the documents they search (the English fact-checks of CimpleKG, as copied on 2 October 2026), the runs of the systems
compared in the paper, the pool of judged documents, the relevance labels of three AI judges and of their majority (the
main judge), and the labels of two human judges on a check sample of 25 queries.

The code that made every file and computes every number of the paper is at
@CODE_URL@ (start from `paper-code/README.md`).

## Licence

Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0,
https://creativecommons.org/licenses/by-nc-sa/4.0/), the licence of CimpleKG, from which the document identifiers, the
dates, the publishers and the claim texts come. Attribution: CimpleKG (https://data.cimple.eu/), Burel et al., "CimpleKG:
A Continuously Updated Knowledge Graph on Misinformation, Factors and Fact-Checks", ISWC 2024. The labels of
Llama-3.3-70B-Instruct are released under the terms that `MODELS.md` quotes ("Built with Llama"). The code is under the
MIT licence (see the code repository).

If you hold rights in a text of this collection and want it removed, open an issue in the code repository.

## Identifiers

- `query_id`: a country, a year and a number, for example `us-2016-01` (United States, 2016, query 1). The year is the
  year of the claim, except for a query that replaced another: it keeps the identifier of the query it replaced, so
  `stratum` in `queries.json` gives the country and the year of every claim.
- `doc_id`: the IRI of a fact-check (a ClaimReview) in CimpleKG, for example
  `http://data.cimple.eu/claim-review/0001bf45...`.

## The files

| file | what it holds |
|---|---|
| `queries.json` | The 200 queries, one object each (described below). |
| `documents.jsonl` | One line per document of the collection: `doc_id`, `date_published`, `publisher` (CimpleKG's name for the organisation) and `claim_text`, the claim the fact-check checks, cut at 300 characters. The explanations of the fact-checks are not included. |
| `qrels/qrels_ai.txt` | The main judge's label (the majority of the three AI judges) for every pair of the pool, in TREC format: `query_id 0 doc_id relevance`, with relevance 1 (relevant) or 0. The main tables of the paper use it. |
| `qrels/qrels_human.txt` | The human label of every document of the check sample: the label of the judge who owns the query. |
| `qrels/qrels_ai_check.txt` | The main judge's label on exactly the documents of `qrels_human.txt`. |
| `qrels/pool_depth.jsonl` | One line per pair of the pool: the rank each system gave the document, whether it is in the check sample, the AI label and the human label. |
| `qrels/qrels_summary.md` | The counts of the three qrels files. |
| `judge/labels_llm_<judge>.jsonl` | Every label of each AI judge (`gptoss120b`: gpt-oss-120b; `llama33_70b`: Llama-3.3-70B-Instruct; `deepseekv32`: DeepSeek-V3.2; `majority`: the main judge), with the model, the precision, the provider that answered through OpenRouter, the checksum of the prompt, the token counts, the cost and the raw answer. |
| `judge/validation.md` | How each candidate judge agrees with the humans, and the rule that chose the main judge. |
| `labels/labels_H1.jsonl`, `labels/labels_H2.jsonl` | The labels of the two human judges, H1 and H2 (`y` or `n`; `kind` is `pool` or `repeat`, a document shown twice to measure consistency), with the checksum of the guideline they read. |
| `labels/labels_adjudicated.jsonl`, `labels/agreement.md` | The shared documents on which both human judges agree, and their agreement (Cohen's kappa). |
| `pool/pool.jsonl` | Every judged pair (`query_id`, `doc_id`), the rank each system gave it (`ranks`), and whether it is in the check sample; a pair added for the tables on the documents published before April 2025 also has `added_for` and `ranks_before`. |
| `pool/check.json`, `pool/shared.json`, `pool/pool_summary.md` | The check sample (its seed, its 25 queries in the order of the draw, the documents of each, the owner judge) and the documents both human judges labelled. |
| `runs/*.trec` | The runs, in TREC format (`query_id Q0 doc_id rank score tag`). `bm25` and `pipeline` are the two text systems of the paper; `dense` and `fused` are the first stages of the pipeline alone; `graph_annotator_a_cr` and `graph_annotator_b_cr` are the Boolean formulas of formula writers A and B (a set, ordered newest first; the score only keeps that order); `*_current` are the same formulas with today's names of the Wikipedia pages only; `relaxed_annotator_a` and `relaxed_annotator_b` are the generous version of the formulas; `*_claimart` are the text systems on the claim followed by the explanation, pooled and judged but not reported in the paper. |
| `runs/*_sets.jsonl`, `runs/*_report.md` | For each formula: its terms with their DBpedia entities, the size of the set of each term, the size of the whole set `B`, and every document of the set. |
| `runs/manifest.json`, `runs/manifest_claimart.json` | The settings of the text runs: models and their revisions, parameters, hardware, timings, the checksums of the run files, and the corrections made to the manifest since the run. |
| `writers/` | `formula_repairs.csv` (every formula repaired before the runs, as written and as repaired, with the rule), `title_changes.csv` (every formula changed by a writer's choice of page for a flagged title), and Wikipedia's answers used to resolve the titles (`wikipedia_redirects.json`, `wikipedia_incoming_redirects.json`, `wikipedia_disambiguation.json`), so that this step can be repeated offline. |
| `snapshot/` | `mirror_manifest.json` (the copy of CimpleKG: date, number of triples, checksum), `snapshot_manifest.json` (the cleaning rules and their counts), `language_decisions.jsonl` (every decision of the language rule with its measurements; no text), `ifcn_sources.json` (the list of IFCN signatories used for the countries). |
| `sources.csv` | The reports the queries come from: country, years, tier, label, address, size, sha256 of the saved file, date fetched. The reports themselves are not included. |
| `drafting_model.json` | The local model that drafted the candidate claims, as every reply recorded it: name, precision, Ollama version, the checksum (digest) of the model file, and the checksum of the prompt. |
| `eval/*.md` | The reports of the measures (main judge, each AI judge alone, the human check sample, the documents before April 2025, the reasons of the misses, the oracle, the older page names, the agreement of the formula writers). |
| `MODELS.md`, `LICENSE`, `reproduce.sh` | The models and their licences, the licence, and the script that reruns the systems with the code. |
| `CHECKSUMS.sha256` | The sha256 of every other file: `sha256sum -c CHECKSUMS.sha256` checks them all. |

### `queries.json`

A list of 200 objects:

- `query_id`, `set` (`main`), `query_type`;
- `free_text`: the query, the text the systems receive;
- `background` (35 queries): a sentence for the judges when a name in the query may be unknown to them; no system
  reads it;
- `draft_text`: the claim as the local model drafted it from the report;
- `stratum`: the country and the year of the claim; `source`: the report (its label in `sources.csv`, its address,
  its year);
- `boolean_formulas`: one object per formula writer (`annotator_a`, `annotator_b`): `formula` (with `e1`, `e2`, ... for
  the entities), `formula_written` (as the writer typed it), `entities` (for each term: the label, the Wikipedia title,
  the Wikidata identifier when known, the DBpedia IRI, and `alias_of` when the term is another name of a page the writer
  chose), `not_expressible` (the concepts the writer could not express as a page), `minutes`, `refinements`, `notes`;
- `provenance`: the candidate and its merged group (`candidate_id`, `cluster_id`), the claim kept by the review
  (`claim`), the saved report file and its sha256 (`file`, `sha256`), the drafting model, its precision and the checksum
  of the prompt, what the drafting recorded about the claim (`entities`, `mentions`, `docs`, `topic`) and about its
  source (`tier`, `publisher_type`), the seed and the rank of the random draw (`draw_seed`, `draw_rank`), `drawn_from`
  and `replaces` for a query drawn from another year or replacing one that broke a review rule, and the two proposals
  for the rewritten
  query: `rewrite_proposal` (the local model's) and `rewrite_closed` (a closed model's, with the model's name in its
  field `model`); for a claim that held several claims (rule W4), `rewrite_parts` and the part kept, `rewrite_part_kept`.
  The reviewer accepted or corrected every proposal; `free_text` is the final text.

The passages of the reports, and the words of the reports that the rewrite used, are not included.
`filters` and `entity_groups` are empty and no system uses them.

## Scoring a run

The qrels and the runs are standard TREC files, so any evaluation tool reads them, for example with ir_measures:

```python
import ir_measures
from ir_measures import P, nDCG, R
qrels = list(ir_measures.read_trec_qrels("qrels/qrels_ai.txt"))
run = list(ir_measures.read_trec_run("runs/pipeline.trec"))
print(ir_measures.calc_aggregate([P@10, P@20, nDCG@20, R@20], qrels, run))
```

The paper leaves out the queries with no relevant document in the pool, scores a formula's result as a set, and
compares the text systems with a set at the same size; `eval/metrics.py` of the code does all of it (run it from the
folder `paper-code` of the code repository):

```
python eval/metrics.py --work WORK --qrels WORK/qrels/qrels_ai.txt --name ai --ranked bm25,pipeline \
    --generous relaxed_annotator_a,relaxed_annotator_b --extra bm25_claimart,pipeline_claimart \
    --sets graph_annotator_a_cr,graph_annotator_b_cr
```

where `WORK` is a folder with this collection's `qrels/` and `runs/`. It writes `WORK/eval/metrics_ai.md`, the same as
`eval/metrics_ai.md` here (`--extra` adds the rows of the runs on the claim and the explanation, which the paper does
not report). The tables on the documents published before April
2025 (`--before 2025-04`) also need the dates: write `WORK/corpus.jsonl` from `documents.jsonl`, one line per document
with `cr_iri` (the `doc_id`) and `date_published`.

## What is not here, and why

- The texts of the reports: most are copyrighted, and anyone can fetch them again from `sources.csv` and check them
  against the sha256.
- The explanations of the fact-checks, which the systems search: the code copies the English part of CimpleKG from its
  public SPARQL endpoint (`snapshot/mirror_english.py`, then `snapshot/build_store.py`); `snapshot/mirror_manifest.json`
  gives the date and the checksum of the copy used. CimpleKG changes every day, so a new copy can differ from it.
- The drafting files (the model's replies, the candidates, the review log), which quote the reports.
