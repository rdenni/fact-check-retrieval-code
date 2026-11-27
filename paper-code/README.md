# Finding fact-checks about a narrative or a claim: the code

This is the code of the paper that measures how well entity search in the CimpleKG knowledge graph finds the
fact-checks about a narrative or a claim, compared with text search. It builds the document collection from a dated
snapshot of CimpleKG, the queries from observer reports, the runs of the three systems, the pool, the judgments and
the measures.

No data is in this repository. Every script takes the Oxigraph store and a work folder as arguments (`--store`,
`--work`). The test collection (the queries with both writers' formulas, the runs, the pool, the labels of the AI and
human judges, the qrels, the snapshot's manifests) is released separately, as the paper's data availability statement
says; `release/export_release.py` builds it.

## The three systems

| | 1. Boolean entity search | 2. BM25 | 3. Text pipeline |
|---|---|---|---|
| query | a Boolean formula over Wikipedia pages, written by each of two writers | the query sentence | the query sentence |
| what it searches | the DBpedia entities CimpleKG links in the explanation | the explanation text | the explanation text |
| code | `systems/graph_boolean.py` | `systems/bm25.py` | `systems/pipeline_2026.py` |

All three search the fact-check explanations. On the English part of CimpleKG every claim is linked to the same entity,
"Mars", so the entity search can only use the entities of the explanations; the text systems read the same explanations, so
that the systems differ only in how they search. The pipeline is BM25 and Qwen3-Embedding-0.6B fused by reciprocal
rank fusion, then reranked by Qwen3-Reranker-0.6B. No model is fine-tuned and nothing is tuned on the queries. The
models are chosen by written rules (`model_selection/RULES.md`); their revisions are in `MODELS.md`.

The text pipeline runs on a laptop: `--profile laptop` is an NVIDIA RTX 4070 Laptop GPU with 8 GB of memory, under
Windows (`profiles.py`). `--profile cpu-test` runs a tiny random model on the CPU, only to test the code.

## Install

Python 3.11. `pip install -r requirements.txt` (the file says how to install PyTorch with CUDA on Windows);
`requirements-paper.txt` holds the exact version of every package of the environment of the paper's runs. The AI judges
run through an API, OpenRouter, from the same environment (`judge/RUN-WITH-API.md`); running them on one's own GPUs
needs a Linux machine with its own requirements (`judge/requirements-judge.txt`, `judge/RUN-ON-GPU.md`).

## Check the code without any data or model

`python tests/run_toy.py` builds a fake 120-document store and 3 queries, runs every script with a tiny random model,
fake judges and a fake language-model server, and stops if anything breaks (about one minute on a CPU). It must print
`TOY RUN OK`.

## The steps, in order

The snapshot (folder `snapshot/`):
```
python snapshot/mirror_english.py ...        # copy the English fact-checks and their entity links from CimpleKG's public SPARQL endpoint
python snapshot/build_store.py ...           # load the copy into an Oxigraph store, applying the cleaning rules (scraping failures, empty claims, language)
python snapshot/publisher_map.py ...         # each publisher's country and IFCN status
python stats/archive_stats.py --store STORE --work WORK [--country-map publishers.csv]   # the counts by year, month, country and publisher
python snapshot/language_rule.py --corpus WORK/corpus.jsonl --out WORK/stats/language_decisions.jsonl   # every language decision with its measurements
```

The queries (folder `queries/`; the rules are in `queries/REVIEW-RULES.md`; the sources of the report passages that the prompts and the rules quote are in `queries/SOURCES.md`):
```
python queries/allocate_queries.py --stats WORK/stats --total 200 --floor 3 --since 2016 --min-docs 50 --out-tex allocation.tex --out-json allocation.json
                                             # the number of queries per (country, year) cell
python queries/fetch_edmo_briefs.py --text   # the EDMO monthly briefs
python queries/fetch_pages.py                # every other report of queries/report_sources.csv, with its text and checksum
python queries/check_source_overlap.py --mirror MIRROR.nt --corpus WORK/corpus.jsonl   # a source made of the archive's documents is excluded
python queries/prepare_cells.py              # one file per cell with its documents
python queries/draft_needs.py --cells reports/cells --work WORK --base-url URL --model M --precision P --resume
                                             # a local model lists the false statements of each cell, each with a word-for-word passage
python queries/dedup_needs.py --work WORK    # merge of the same claim within a country
python queries/review_needs.py sheet|import --work WORK ...      # the keep-or-drop review, under the written rules
python queries/draw_queries.py draw ...      # the random draw of the queries and the pilot, with a published seed
python queries/propose_rewrites.py ...       # a local model proposes each query sentence
python queries/draw_queries.py rewrite-sheet|import-rewrite|replace ...   # the rewrite under rules W1 to W5
```
At the rewrite a second proposal comes from a closed model, which reads the report around each passage
(`draw_queries.py rewrite-sheet|import-rewrite --closed-proposals FILE`); the reviewer accepts or corrects every row, and
both proposals are kept beside the final text (`provenance.rewrite_proposal`, `provenance.rewrite_closed`).

The formulas (two writers, each writes every query from the query and English Wikipedia alone; their instruction, in
English, is `queries/FORMULA-INSTRUCTION.md`):
```
python queries/import_formulas.py template --queries Q.json --out sheet_a.xlsx --author annotator_a
python queries/import_formulas.py import --queries Q.json --sheet sheet_a.xlsx --resolve-redirects --catalog WORK/stats/entity_catalog.csv --out Q.json
python queries/title_check.py --sheet SHEET.xlsx --writer-name NAME --out FEEDBACK.txt ...   # disambiguation pages and non-pages, sent back to the writer
python queries/entity_catalog.py build --work WORK --node cr                                 # the archive's entity pages with counts; never shown to a writer
```

The runs:
```
python run_all.py --queries Q.json --store STORE --work WORK --profile laptop --node cr --text-mode article   # BM25 and the pipeline on the explanations
python systems/graph_boolean.py --queries Q.json --store STORE --work WORK --author annotator_a --node cr --backend index   # each writer's graph run (then annotator_b)
python systems/graph_boolean.py ... --current-names-only     # the same formulas with today's page names only (eval/stale_names.py)
python systems/ranked_entities.py --queries Q.json --work WORK --node cr --author annotator_a --tag relaxed_annotator_a   # the generous version of each writer's formula
```
`reproduce.ps1` (Windows) and `reproduce.sh` (Linux) chain these runs; they first run the text systems on the claim
followed by the explanation and rename those runs `*_claimart`; they are pooled and judged, and the paper does not report them.

The pool and the judgments:
```
python pool/build_pool.py --work WORK --queries Q.json --ranked bm25,pipeline --sets graph_annotator_a_cr,graph_annotator_b_cr --ai-only relaxed_annotator_a,relaxed_annotator_b,bm25_claimart,pipeline_claimart --judges R,M --country-map WORK/stats/publishers_map.csv
python pool/extend_pool_before.py --work WORK --before 2025-04 --ranked bm25,pipeline --sets graph_annotator_a_cr,graph_annotator_b_cr --ai-only relaxed_annotator_a,relaxed_annotator_b,bm25_claimart,pipeline_claimart --country-map WORK/stats/publishers_map.csv   # the documents that the lists cut before April 2025 move up into the top 20, judged too
python judge/pack_for_judge.py --work WORK --queries Q.json --out judge_bundle --zip   # what the AI judges need (judge/RUN-WITH-API.md)
python judge/run_judges.py run --config judge/judges_api.json --bundle judge_bundle   # the three AI judges through OpenRouter (first with --max-items 200)
python pool/judges_sheet.py make --work WORK --queries Q.json --out SHEETS             # one Excel sheet per human judge, with the guideline (pool/GUIDELINE.md)
python pool/judges_sheet.py make ... --only R --round 2                               # a fresh blank sheet for one judge, rows in another order (a second round)
python pool/judges_sheet.py read --work WORK --sheet SHEETS/sheet_M.xlsx --annotator M # a returned sheet -> WORK/labels/labels_M.jsonl
python judge/adjudicate.py --work WORK --annotators R,M   # the judges' kappa on the shared documents (no discussion: the owner's label is final)
python judge/validate_judge.py --work WORK --annotators R,M                            # each AI judge against the humans; the pass rule; the main judge
python qrels/build_qrels.py --work WORK --annotators R,M --llm-labels WORK/judge/labels_llm_<main judge>.jsonl
```

The measures:
```
python eval/metrics.py --work WORK --qrels WORK/qrels/qrels_ai.txt --name ai --ranked bm25,pipeline --generous relaxed_annotator_a,relaxed_annotator_b --extra bm25_claimart,pipeline_claimart --sets graph_annotator_a_cr,graph_annotator_b_cr
python eval/failure_protocol.py --work WORK --queries Q.json --qrels WORK/qrels/qrels_ai.txt --graph-tag graph_annotator_a_cr --node cr --name ai_a
python eval/oracle_formula.py --work WORK --qrels WORK/qrels/qrels_ai.txt --graph-tag graph_annotator_a_cr --node cr --name ai_a
python eval/stale_names.py --work WORK --qrels WORK/qrels/qrels_ai.txt --name ai
python eval/writer_agreement.py --work WORK --queries Q.json --authors annotator_a,annotator_b
```
`eval/metrics.py` gives P@10, P@20, nDCG@20 and recall against the pool for the ranked systems, set precision, recall,
F1 and size for each writer's result set, the ranked systems cut at each set's size, and Wilcoxon tests with the
Benjamini-Hochberg correction; `--before 2025-04` repeats every table on the documents published before CimpleKG
stopped linking entities.

The Results section of the paper, from the saved result files (after all the measures above, with the single-judge
robustness runs `--qrels WORK/qrels_<model>/qrels_ai.txt --name ai_<model>`, `--qrels WORK/qrels/qrels_human.txt --name human`,
`--qrels WORK/qrels/qrels_ai_check.txt --name ai_check` and `--before 2025-04 --name ai_before`):
```
python eval/paper_results.py --work WORK --queries Q.json    # latex/figures/results_numbers.tex, results_*.tex, results_*.pdf, results.json
```

The paper's numbers: `stats/plot_archive.py`, `stats/paper_facts.py`, `stats/live_checks.py` and
`model_selection/pareto_simple.py` write every number, table and figure of the paper as LaTeX files;
`reproduce_paper.py --snap SNAP --paper-figures DIR` rebuilds them all from the snapshot folder (the Oxigraph store, the
N-Triples mirror and its manifest, the IFCN file, the query files and the work folder of the runs; the counts of the report sources also need the fetched reports, --reports DIR, or --skip-sources) and compares each
with the paper's file of the same name in DIR; without `--paper-figures` it only rebuilds them. `release/export_release.py
--humans ID1,ID2` assembles the released collection (the queries, the formulas, the runs, the pool, the labels; no
quotation of the reports and no explanation texts; the two human judges are H1 and H2, as in the paper).

