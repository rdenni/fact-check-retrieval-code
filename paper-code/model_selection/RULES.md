# The rules that chose the models

Every model of the paper was chosen by a rule written before it was used: the text pipeline's models before any query
existed, the AI judges before any relevance label. No model was chosen by trying it on the queries.

## 1. The embedding model of the text pipeline (rule S, written 2026-09-17)

**Data.** One file: `mteb/english/MTEB_eng_v2_summary(1).csv`, downloaded from
https://huggingface.co/spaces/mteb/leaderboard, benchmark MTEB(eng, v2), on 2026-09-17. One column: `Retrieval`, the
model's average nDCG@10 over the 10 retrieval datasets of that benchmark (ArguAna, ClimateFEVERHardNegatives,
CQADupstackGamingRetrieval, CQADupstackUnixRetrieval, FEVERHardNegatives, FiQA2018, HotpotQAHardNegatives, SCIDOCS,
TRECCOVID, Touche2020Retrieval.v3). Retrieval is the only thing the model does in the pipeline.

**Conditions**, all read from columns of that file:
- C1 `Open Weights` = true: the model must be downloaded and run.
- C2 `Openness: Open license` = true: the leaderboard's own flag that the licence permits use.
- C3 `Openness: Paper` = true: a technical report or paper exists, so the model can be cited and its training data read.
- C4 `Zero-shot` >= 90: the share of the benchmark's 41 datasets the model did NOT declare as training data; 90 means
  at most 4 declared datasets. A score on one's own training data is not a fair score. NA (never declared) fails.
- C5 memory: the model plus 2 GB of working room must fit the graphics card.

**Choice.** Sort the models that pass C1 to C4 by the Retrieval column, best first. Pareto front: keep a model only if
it needs less memory than every model kept above it. Take the top of the front that fits the card. Script:
`pareto_simple.py`; report: `report-2026-09-17-v5-simple.md`, which also shows the front under every combination of
C3 and C4.

**Result.** Front: Qwen3-Embedding-8B (Retrieval 69.44, 14.1 GB) -> Qwen3-Embedding-4B (68.46, 7.5 GB) ->
Qwen3-Embedding-0.6B (61.83, 1.1 GB). On the 8 GB laptop GPU of the paper's runs the rule picks Qwen3-Embedding-0.6B.
Without C3 and C4 it would pick Yuan-embedding-2.0-en (70.69): a Qwen3-Embedding-0.6B fine-tune with no report. With
C3 but not C4 the 8 GB pick would be Jasper-Token-Compression-600M (66.19), which declares 21 of the 41 datasets as
training data.

## 2. The reranker

The Qwen3-Reranker of the same size as the chosen embedder (0.6B): released with the embedder, same licence, same
instruction format. There is no reranker leaderboard comparable to MTEB, and the vendors' own tables disagree; taking
the reranker released with the embedder is also what Upravitelev et al. (2026) do.

## 3. BM25 (written 2026-09-18)

The setting of the BEIR benchmark's lexical baseline: "Anserini with the default Lucene parameters (k=0.9 and b=0.4)"
(Thakur et al. 2021, Section 3). Implementation: bm25s (Lu 2024), method "lucene", k1 = 0.9, b = 0.4, lower-casing,
English stop words, Snowball English stemmer, exact retrieval. No parameter is tuned on the queries. Code:
`common/textprep.py`, `systems/bm25.py`. The same index is the first channel of the pipeline.

## 4. The AI judges (version 7, written 2026-11-20, before any label)

**Why a version 7.** Version 6 (2026-11-16) ran the judges on GPUs: per family the
largest eligible model that vLLM starts. No GPU machine was available; the only compute offered was an account on
OpenRouter, an API that routes each request to a company that serves the model. Two of version 6's models are no
longer served by any API we checked on 2026-11-20: Llama-3-70B-Instruct and the March and April 2024 Command-R and
Command-R+ (Cohere deprecated them in September 2025). And a paper read on 2026-11-20 tests many open models alone
against human labels (Keller et al., SIGIR 2026). The conditions of version 6 stay; one condition is added and the way a
model is picked changes. No label existed when this version was written.

**Eligible models** (all conditions):
- open weights (Faggioli et al. 2023, Section 7.2: a closed model can be changed by its owner without notice);
- not of the Qwen family, the family of the pipeline's models (Balog et al. 2025, Section 5.1: model judges favour
  their own family);
- evaluated as a relevance judge against human relevance labels in the literature;
- a licence that allows publishing the labels;
- (new) served on OpenRouter on 2026-11-20 by at least one provider that declares the model's released precision
  (for a model released in bf16 and offered by no provider in bf16, a 16-bit format) and accepts the seed parameter.
  The provider listings of that day are in `openrouter_2026-11-20/`.

**The evidence.** Keller et al. 2026, Table 3: Cohen's kappa with the TREC Robust04 relevance labels of open models
that relabel the pool with the prompt of Thomas et al.: Llama-3.3-70B 0.80, Llama-3.1-70B 0.76, gpt-oss-120b 0.72,
DeepSeek-V3.2 0.71, Nemotron-3-30B 0.71, gpt-oss-20b 0.64, Mistral-3-14B 0.63, Llama-3.1-8B 0.24 (the Qwen models
left out; the closed GPT-4.1 has 0.64). Farzi and Dietz 2025 also use Llama-3.3-70B as a judge and compare the system
rankings with those of human labels; the TREC 2025 RAG ensemble includes gpt-oss-120b (overview, p. 6). The models
tested by Alaofi et al. 2026 (Llama-3-70B and 8B, Command-R and Command-R+ of 2024) are no longer served.

**Three families, one model each.** Three, as in version 6, so that the majority of the three is a candidate.
- Within a family, the largest eligible model; at equal size, the one with the higher kappa in Keller et al.:
  OpenAI, gpt-oss-120b; Meta, Llama-3.3-70B-Instruct (0.80 against Llama-3.1-70B's 0.76).
- The Cohere family has no eligible model: its tested models are no longer served and no later Cohere model has been
  tested as a judge. Its place goes to the eligible family whose model has the highest kappa in Keller et al.;
  DeepSeek-V3.2 and Nemotron-3-30B tie at 0.71, and the tie is broken by the larger model first, the order this rule
  already uses within a family: DeepSeek-V3.2 (685 billion parameters on its model card) before Nemotron-3-30B (32
  billion).

**Precision and providers.** gpt-oss-120b in its released MXFP4 (OpenRouter's "fp4"); DeepSeek-V3.2 in its released
fp8; Llama-3.3-70B-Instruct, released in bf16, in fp16, the only 16-bit offer. Of the providers that serve that
precision and accept and apply every parameter sent (the temperature, the seed, the reply's length and, where it is
sent, the reasoning setting), as listed on 2026-11-20, the cheapest is used and the next cheapest is the only fallback:
gpt-oss-120b, Near AI then CoreWeave; Llama-3.3-70B-Instruct, CoreWeave (the only one); DeepSeek-V3.2, AtlasCloud then
Baidu. GMICloud, cheaper, lists the reasoning setting but does not apply it: on a neutral question, not a pair of the
pool, it thought whatever was sent, while AtlasCloud and Baidu answered without thinking
(`openrouter_2026-11-20/deepseek_thinking_test.txt`). No other provider is allowed. Every label records the provider that answered and the model
version the API reports. The list is `judge/judges_api.json`.

**Settings.** One prompt (`judge/prompts.py`), the human judges' guideline in the shape of the UMBRELA prompt, fixed
before any label and never tuned. Temperature 0 and a fixed seed. gpt-oss-120b reasons with effort medium, as in the
TREC 2025 RAG ensemble and in Keller et al., who use each model's default; DeepSeek-V3.2 answers without its thinking
mode, which is off by default; Llama-3.3-70B-Instruct has no reasoning step. Every model labels every pooled pair.

**The main judge.** Candidates: each model alone and the majority of the three (a pair is relevant if at least two
say YES). A candidate passes if (a) its Cohen's kappa with each human judge reaches the lower end of the 95 percent
bootstrap interval of the two human judges' kappa with each other, on the independent labels of the documents both
judged, and (b) on the check queries the systems come out in the same order with the human labels and with its labels.
The main judge is the passing candidate with the highest mean kappa with the two judges; the others are reported as a
robustness table. If no candidate passes, the candidate with the highest mean kappa is used and the paper says that it
did not pass. Script: `judge/validate_judge.py`.
