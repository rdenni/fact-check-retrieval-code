# Running the AI judge on a GPU machine

The paper's labels were made through an API (`RUN-WITH-API.md`). This page is for running open-weight models on one's
own GPUs with vLLM instead, for example to label the pool again with the same models.

What it does: open-weight language models label every pooled (query, fact-check) pair of the bundle as relevant or
not, one model after the other, with one fixed prompt (`judge/prompts.py`). Everything is one script,
`judge/run_judges.py`. For each model it downloads the model, starts a local vLLM server, labels every pair, stops the
server, and records what ran; a last command writes the results into one zip. If it is interrupted, run the same
command again: it continues where it stopped.

You need this repository and the data folder `judge_bundle.zip` (made by `judge/pack_for_judge.py`; data never goes
into the repository). The models to run are listed in `judge/judges.json`, written for the machine in the shape of
`judge/judges_api.json` but with Hugging Face model ids (for example `openai/gpt-oss-120b`,
`meta-llama/Llama-3.3-70B-Instruct`, `deepseek-ai/DeepSeek-V3.2`), no `base_url`, and `serve_args` for vLLM where needed.

## Before you start (once)

1. A Linux machine with NVIDIA cards large enough for the models of `judge/judges.json`, Python 3.10 to 3.12, and
   about 400 GB of free disk for the model files (`check` prints the free space).
2. A Hugging Face account and a read token (huggingface.co, Settings, Access Tokens). Gated models (Llama, Command-R)
   need their licence accepted on their Hugging Face page while logged in; Llama is approved by hand, which can take a
   day.

## The commands

```bash
git clone <this repository>
cd <the repository folder>/paper-code
python3 -m venv .venv-judge && source .venv-judge/bin/activate
pip install -r judge/requirements-judge.txt
export HF_TOKEN=hf_xxx            # your read token

# 1. check the machine (a few seconds): GPUs, versions, token, disk
python judge/run_judges.py check --out ~/judge_out

# 2. the run (a few hours; if it stops, run the same command again)
unzip judge_bundle.zip -d ~/judge_bundle
python judge/run_judges.py run --config judge/judges.json --bundle ~/judge_bundle --gpus 1

# 3. pack the results
python judge/run_judges.py pack --bundle ~/judge_bundle --out ~/judge_results.zip --add ~/judge_out/env.json
```

`--gpus N` gives one model N cards (needed only for a model that does not fit one card). `--only NAME` runs one model
of the list. On a cloud machine, run step 2 inside `tmux` (or with `nohup ... &`) so that closing the terminal
does not stop it.

## What the zip holds

`judge_results.zip`: the labels (`labels_llm_<model>.jsonl`), one `run_<model>.json` per model (model, model revision,
vLLM version, GPU, start and end time), the vLLM logs, and the machine's description (`env.json`).
