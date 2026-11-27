# Embedder choice by rule S (RULES.md, section 1), leaderboard file of 17 September 2026

File: `mteb/english/MTEB_eng_v2_summary(1).csv`. Column: `Retrieval` = average nDCG@10 over the 10 retrieval datasets of MTEB(eng, v2).
Models with a retrieval score, a known size, Open Weights = true and Openness: Open license = true: 160 (conditions C1, C2).

## Legend
- **Rank**: position by the Retrieval column among these models (1 = highest average nDCG@10).
- **Retrieval**: the column value, 0 to 100. A difference of 1 point is one nDCG@10 point averaged over 10 datasets.
- **Memory GB**: the board's estimate of the weights in 16-bit, from its Memory (MB) column, divided by 1024.
- **GPU needed**: the smallest usual card that holds the weights plus 2 GB of working room; **$/h**: rough rental price range in September 2026 (price trackers; not a quote).
- **Paper**: the board's own flag Openness: Paper (a technical report or paper exists). Checked by hand for Yuan (no report on the model card) and Jasper (arXiv 2511.14405): the flag is right.
- **Zero-shot**: the board's number, the share of its 41 datasets the model did NOT declare as training data (95 = trained on 2 of 41; 48 = trained on 21 of 41; NA = never declared). Condition C4 requires at least 90.

## The ranking, top 25

| Rank | Model | Retrieval | Memory GB | GPU needed | $/h | Paper | Zero-shot | Licence | Released |
|---|---|---|---|---|---|---|---|---|---|
| 1 | IEITYuan/Yuan-embedding-2.0-en | 70.69 | 2.2 | your laptop, 8 GB | 0 | no | 95 | apache-2.0 | 2025-11-27 |
| 2 | Qwen/Qwen3-Embedding-8B | 69.44 | 14.1 | 24 GB card (RTX 3090/4090, L4) | 0.3-0.7 | yes | 95 | apache-2.0 | 2025-06-05 |
| 3 | Qwen/Qwen3-Embedding-4B | 68.46 | 7.5 | 16 GB card (T4, A10G) | 0.2-0.5 | yes | 95 | apache-2.0 | 2025-06-05 |
| 4 | Kingsoft-LLM/QZhou-Embedding | 67.12 | 14.1 | 24 GB card (RTX 3090/4090, L4) | 0.3-0.7 | yes | 48 | apache-2.0 | 2025-08-24 |
| 5 | infgrad/Jasper-Token-Compression-600M | 66.19 | 2.2 | your laptop, 8 GB | 0 | yes | 48 | mit | 2025-11-14 |
| 6 | annamodels/LGAI-Embedding-Preview | 66.18 | 26.5 | 48 GB card (L40S, A6000) | 0.7-1.5 | yes | 51 | apache-2.0 | 2025-06-11 |
| 7 | infly/inf-retriever-v1 | 64.07 | 13.2 | 16 GB card (T4, A10G) | 0.2-0.5 | yes | 80 | apache-2.0 | 2024-12-24 |
| 8 | Qwen/Qwen3-Embedding-0.6B | 61.83 | 1.1 | your laptop, 8 GB | 0 | yes | 95 | apache-2.0 | 2025-06-05 |
| 9 | GeoGPT-Research-Project/GeoEmbedding | 60.91 | 27.0 | 48 GB card (L40S, A6000) | 0.7-1.5 | no | 58 | apache-2.0 | 2025-04-22 |
| 10 | infly/inf-retriever-v1-1.5b | 60.83 | 2.9 | your laptop, 8 GB | 0 | yes | 80 | apache-2.0 | 2025-02-08 |
| 11 | codefuse-ai/F2LLM-v2-14B | 60.63 | 52.1 | 80 GB card (A100, H100) | 0.8-3 | yes | 43 | apache-2.0 | 2026-03-09 |
| 12 | ai-sage/Giga-Embeddings-instruct-10B-A1.8B-0826 | 60.03 | 19.5 | 24 GB card (RTX 3090/4090, L4) | 0.3-0.7 | no | NA | mit | 2026-08-19 |
| 13 | codefuse-ai/F2LLM-v2-8B | 59.82 | 28.2 | 48 GB card (L40S, A6000) | 0.7-1.5 | yes | 43 | apache-2.0 | 2026-03-09 |
| 14 | ai-sage/Giga-Embeddings-instruct | 59.68 | 12.6 | 16 GB card (T4, A10G) | 0.2-0.5 | no | NA | mit | 2025-09-23 |
| 15 | codefuse-ai/F2LLM-4B | 59.63 | 7.5 | 16 GB card (T4, A10G) | 0.2-0.5 | yes | 43 | apache-2.0 | 2025-09-18 |
| 16 | ai-sage/Giga-Embeddings-instruct-3B-0826 | 59.45 | 5.9 | your laptop, 8 GB | 0 | no | NA | mit | 2026-08-19 |
| 17 | ICT-TIME-and-Querit/ICT-TIME-and-Querit-embedding-v1 | 59.37 | 7.5 | 16 GB card (T4, A10G) | 0.2-0.5 | yes | 80 | apache-2.0 | 2026-05-02 |
| 18 | codefuse-ai/F2LLM-v2-4B | 59.13 | 15.0 | 24 GB card (RTX 3090/4090, L4) | 0.3-0.7 | yes | 43 | apache-2.0 | 2026-03-09 |
| 19 | Snowflake/snowflake-arctic-embed-l | 59.04 | 1.2 | your laptop, 8 GB | 0 | yes | 95 | apache-2.0 | 2024-04-12 |
| 20 | Snowflake/snowflake-arctic-embed-l-v2.0 | 58.56 | 2.1 | your laptop, 8 GB | 0 | yes | 95 | apache-2.0 | 2024-12-04 |
| 21 | KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5 | 58.45 | 1.8 | your laptop, 8 GB | 0 | yes | 58 | apache-2.0 | 2025-09-30 |
| 22 | Snowflake/snowflake-arctic-embed-m-v2.0 | 58.41 | 1.1 | your laptop, 8 GB | 0 | yes | 95 | apache-2.0 | 2024-12-04 |
| 23 | Alibaba-NLP/gte-Qwen2-7B-instruct | 58.09 | 28.4 | 48 GB card (L40S, A6000) | 0.7-1.5 | yes | NA | apache-2.0 | 2024-06-15 |
| 24 | Snowflake/snowflake-arctic-embed-m-v1.5 | 58.05 | 0.4 | your laptop, 8 GB | 0 | yes | 95 | apache-2.0 | 2024-07-08 |
| 25 | codefuse-ai/F2LLM-1.7B | 57.83 | 3.2 | your laptop, 8 GB | 0 | yes | 43 | apache-2.0 | 2025-09-18 |

## The Pareto front under each combination of conditions (C1 and C2 always on)

Reading rule: start at rank 1; keep a model only if it needs less memory than every model kept above it; stop at the first kept model that fits the laptop. A model that is bigger and lower-ranked than a kept model never appears (it is dominated).

### Front with no further condition

| Rank | Model | Retrieval | Memory GB | GPU needed | $/h | Paper | Zero-shot | Licence | Released |
|---|---|---|---|---|---|---|---|---|---|
| 1 | IEITYuan/Yuan-embedding-2.0-en | 70.69 | 2.2 | your laptop, 8 GB | 0 | no | 95 | apache-2.0 | 2025-11-27 |

### Front with C3 paper

| Rank | Model | Retrieval | Memory GB | GPU needed | $/h | Paper | Zero-shot | Licence | Released |
|---|---|---|---|---|---|---|---|---|---|
| 2 | Qwen/Qwen3-Embedding-8B | 69.44 | 14.1 | 24 GB card (RTX 3090/4090, L4) | 0.3-0.7 | yes | 95 | apache-2.0 | 2025-06-05 |
| 3 | Qwen/Qwen3-Embedding-4B | 68.46 | 7.5 | 16 GB card (T4, A10G) | 0.2-0.5 | yes | 95 | apache-2.0 | 2025-06-05 |
| 5 | infgrad/Jasper-Token-Compression-600M | 66.19 | 2.2 | your laptop, 8 GB | 0 | yes | 48 | mit | 2025-11-14 |

### Front with C4 zero-shot >= 90

| Rank | Model | Retrieval | Memory GB | GPU needed | $/h | Paper | Zero-shot | Licence | Released |
|---|---|---|---|---|---|---|---|---|---|
| 1 | IEITYuan/Yuan-embedding-2.0-en | 70.69 | 2.2 | your laptop, 8 GB | 0 | no | 95 | apache-2.0 | 2025-11-27 |

### Front with C3 paper + C4 zero-shot >= 90 **<- rule S**

| Rank | Model | Retrieval | Memory GB | GPU needed | $/h | Paper | Zero-shot | Licence | Released |
|---|---|---|---|---|---|---|---|---|---|
| 2 | Qwen/Qwen3-Embedding-8B | 69.44 | 14.1 | 24 GB card (RTX 3090/4090, L4) | 0.3-0.7 | yes | 95 | apache-2.0 | 2025-06-05 |
| 3 | Qwen/Qwen3-Embedding-4B | 68.46 | 7.5 | 16 GB card (T4, A10G) | 0.2-0.5 | yes | 95 | apache-2.0 | 2025-06-05 |
| 8 | Qwen/Qwen3-Embedding-0.6B | 61.83 | 1.1 | your laptop, 8 GB | 0 | yes | 95 | apache-2.0 | 2025-06-05 |

## What rule S picks

- 24 GB card (RTX 3090/4090, L4): **Qwen/Qwen3-Embedding-8B** (rank 2, Retrieval 69.44, 14.1 GB).
- 16 GB card (T4, A10G): **Qwen/Qwen3-Embedding-4B** (rank 3, Retrieval 68.46, 7.5 GB).
- your laptop, 8 GB: **Qwen/Qwen3-Embedding-0.6B** (rank 8, Retrieval 61.83, 1.1 GB).
- Reranker: the Qwen3-Reranker of the same size as the chosen embedder (released with it, same licence, same instruction format).

## What the other combinations would pick, so nothing is hidden

- Without C3 and C4: IEITYuan/Yuan-embedding-2.0-en (rank 1, 70.69, 2.2 GB). It is a Qwen3-Embedding-0.6B fine-tune with no report, nothing to cite and no statement of what it was trained on.
- With C3 only: Qwen/Qwen3-Embedding-8B (69.44, 14.1 GB, zero-shot 95) -> Qwen/Qwen3-Embedding-4B (68.46, 7.5 GB, zero-shot 95) -> infgrad/Jasper-Token-Compression-600M (66.19, 2.2 GB, zero-shot 48). The laptop model here declares 21 of the board's 41 datasets as training data, so its Retrieval number is partly a score on its own training data.
