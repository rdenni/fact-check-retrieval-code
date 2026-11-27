# Running the AI judges through an API (the paper's run)

What it does: three open-weight language models label every pooled (query, fact-check) pair of the bundle as
relevant or not, one model after the other, with one fixed prompt (`judge/prompts.py`). The models are not run on our
own machine: OpenRouter (openrouter.ai), an API that routes each request to a company that serves the model, answers
them. The models, their precision and the providers allowed are fixed in `judge/judges_api.json`, chosen by section 4
of `model_selection/RULES.md` (version 7); the provider listings the rule was applied to are in
`model_selection/openrouter_2026-11-20/`. No GPU and no vLLM are needed: the main `requirements.txt` is enough.

Every label records the model, the precision asked for, the provider that answered, the model version the provider
reports, the generation id, the token counts and the cost. The API key is read from an environment variable and is
never written anywhere.

You need this repository, the data folder `judge_bundle` (made by `judge/pack_for_judge.py`; data never goes into the
repository) and an OpenRouter API key with credit.

## 1. The key (once)

Put the key in an environment variable, in a terminal of its own, so that it never appears in a file or a log:

```powershell
setx OPENROUTER_API_KEY "the key"
```

then close every terminal and open a new one (`setx` only reaches terminals opened afterwards). On Linux or macOS:
`export OPENROUTER_API_KEY="the key"` in the shell that runs the judge.

## 2. A measuring run (a few cents)

Label a small first part of the pool with each model and read the cost the API reports:

```bash
python judge/run_judges.py run --config judge/judges_api.json --bundle BUNDLE --max-items 200
```

For each model the script prints the labels, the mean tokens and "cost reported by the server", with the cost of the
pairs still to label. These labels are kept: the full run skips them.

## 3. The full run

```bash
python judge/run_judges.py run --config judge/judges_api.json --bundle BUNDLE
```

`--only NAME` runs one model of the list. If the run is interrupted, or a provider is down for a while, run the same
command again: it continues where it stopped. If the API refuses the key, the account has no credit left, or no
allowed provider serves the model (HTTP 401, 402, 403 or 404), the run stops at once with the API's message and
keeps every label written so far; fix the cause and run the same command again.

## 4. The results

The labels are in `BUNDLE/judge/labels_llm_<name>.jsonl` and one `run_<name>.json` per model. Copy the label files
into `WORK/judge/` and go on with `judge/README.md`, step 3.
