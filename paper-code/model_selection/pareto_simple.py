"""Rule S (model_selection/RULES.md, section 1): the embedder choice.

One file, one column, four yes/no conditions, one Pareto front.

  file      : mteb/english/MTEB_eng_v2_summary(1).csv (downloaded from the MTEB leaderboard, benchmark MTEB(eng, v2), 2026-09-17)
  column    : "Retrieval" = the model's average nDCG@10 over the 10 retrieval datasets of MTEB(eng, v2)
  conditions: all read from columns of the same file (nothing is our own judgement)
     C1  Open Weights = true              (we can download and run it)
     C2  Openness: Open license = true    (the board's own flag that the licence permits use)
     C3  Openness: Paper = true           (a technical report or paper exists; we can cite it and read what it was trained on)
     C4  Zero-shot >= 90                  (the model declares at most 4 of the board's 41 datasets as training data;
                                           a score on your own training data is not a fair score; NA = not declared = fails)
  front     : sort the models that pass by the column, best first; keep a model only if it needs less memory than every
              model above it; stop after the first kept model that fits the laptop (memory + 2 GB headroom <= 8 GB).

The report shows the front under every combination of C3 and C4 (C1 and C2 always on), so nothing is hidden.
Output: paper-code/model_selection/report-2026-09-17-v5-simple.md
"""
import csv, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE  # the leaderboard files are in model_selection/mteb/
FILE = os.path.join(ROOT, "mteb", "english", "MTEB_eng_v2_summary(1).csv")
FILE_TASKS = os.path.join(ROOT, "mteb", "english", "MTEB_eng_v2_per_task(1).csv")  # one column per dataset: counts the datasets
DOWNLOAD_DATE = "2026-09-17"  # the day both files were downloaded from the leaderboard
OUT = os.path.join(HERE, "report-2026-09-17-v5-simple.md")
OUT_TEX = os.path.join(HERE, "..", "..", "latex", "figures", "model_numbers.tex")  # the paper's latex/ folder, next to paper-code/
COLUMN = "Retrieval"
ZS_MIN = 90
HEADROOM_GB = 2.0
LAPTOP_GB = 8
CARDS = [(8, "your laptop, 8 GB", "0"), (16, "16 GB card (T4, A10G)", "0.2-0.5"), (24, "24 GB card (RTX 3090/4090, L4)", "0.3-0.7"),
         (48, "48 GB card (L40S, A6000)", "0.7-1.5"), (80, "80 GB card (A100, H100)", "0.8-3")]
TOP = 25


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def card_for(mem_gb):
    need = mem_gb + HEADROOM_GB
    for size, name, price in CARDS:
        if need <= size:
            return name, price
    return "more than 80 GB", "3+"


def fits_laptop(mem_gb):
    return mem_gb + HEADROOM_GB <= LAPTOP_GB


def front(models, use_paper, use_zs):
    """models: sorted best first. Returns the Pareto front down to the first laptop-fitting model."""
    kept, best_mem = [], None
    for m in models:
        if use_paper and not m["paper"]:
            continue
        if use_zs and (m["zs"] is None or m["zs"] < ZS_MIN):
            continue
        if best_mem is None or m["mem"] < best_mem:
            kept.append(m)
            best_mem = m["mem"]
            if fits_laptop(m["mem"]):
                break
    return kept


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-report", default=OUT, help="the markdown report (default: the dated report next to this script)")
    ap.add_argument("--out-tex", default=OUT_TEX, help="the LaTeX macros for the paper (default: latex/figures/model_numbers.tex)")
    args = ap.parse_args()
    with open(FILE, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    models = []
    for r in rows:
        score, mem = num(r[COLUMN]), num(r["Memory (MB)"])
        if score is None or mem is None:
            continue  # no retrieval score (e.g. cross-encoders) or no known size (closed API models)
        if r["Open Weights"] != "true" or r["Openness: Open license"] != "true":
            continue  # C1, C2
        models.append(dict(name=r["Model"], score=score, mem=mem / 1024, paper=r["Openness: Paper"] == "true",
                           zs=num(r["Zero-shot"]), lic=r["License"] or "?", released=r["Release Date"][:10], maxlen=r["Max Tokens"]))
    models.sort(key=lambda m: -m["score"])
    for i, m in enumerate(models, 1):
        m["rank"] = i

    def row(m):
        card, price = card_for(m["mem"])
        zs = "NA" if m["zs"] is None else f"{m['zs']:.0f}"
        return f"| {m['rank']} | {m['name']} | {m['score']:.2f} | {m['mem']:.1f} | {card} | {price} | {'yes' if m['paper'] else 'no'} | {zs} | {m['lic']} | {m['released']} |"

    head = ("| Rank | Model | Retrieval | Memory GB | GPU needed | $/h | Paper | Zero-shot | Licence | Released |\n"
            "|---|---|---|---|---|---|---|---|---|---|")
    L = []
    say = L.append
    say("# Embedder choice by rule S (RULES.md, section 1), leaderboard file of 17 September 2026")
    say("")
    say(f"File: `mteb/english/MTEB_eng_v2_summary(1).csv`. Column: `{COLUMN}` = average nDCG@10 over the 10 retrieval datasets of MTEB(eng, v2).")
    say(f"Models with a retrieval score, a known size, Open Weights = true and Openness: Open license = true: {len(models)} (conditions C1, C2).")
    say("")
    say("## Legend")
    say("- **Rank**: position by the Retrieval column among these models (1 = highest average nDCG@10).")
    say("- **Retrieval**: the column value, 0 to 100. A difference of 1 point is one nDCG@10 point averaged over 10 datasets.")
    say("- **Memory GB**: the board's estimate of the weights in 16-bit, from its Memory (MB) column, divided by 1024.")
    say(f"- **GPU needed**: the smallest usual card that holds the weights plus {HEADROOM_GB:.0f} GB of working room; **$/h**: rough rental price range in September 2026 (price trackers; not a quote).")
    say("- **Paper**: the board's own flag Openness: Paper (a technical report or paper exists). Checked by hand for Yuan (no report on the model card) and Jasper (arXiv 2511.14405): the flag is right.")
    say(f"- **Zero-shot**: the board's number, the share of its 41 datasets the model did NOT declare as training data (95 = trained on 2 of 41; 48 = trained on 21 of 41; NA = never declared). Condition C4 requires at least {ZS_MIN}.")
    say("")
    say(f"## The ranking, top {TOP}")
    say("")
    say(head)
    for m in models[:TOP]:
        say(row(m))
    say("")
    say("## The Pareto front under each combination of conditions (C1 and C2 always on)")
    say("")
    say("Reading rule: start at rank 1; keep a model only if it needs less memory than every model kept above it; stop at the first kept model that fits the laptop. A model that is bigger and lower-ranked than a kept model never appears (it is dominated).")
    for use_paper, use_zs in [(False, False), (True, False), (False, True), (True, True)]:
        label = " + ".join(x for x, on in [("C3 paper", use_paper), (f"C4 zero-shot >= {ZS_MIN}", use_zs)] if on) or "no further condition"
        tag = " **<- rule S**" if (use_paper and use_zs) else ""
        say("")
        say(f"### Front with {label}{tag}")
        say("")
        say(head)
        for m in front(models, use_paper, use_zs):
            say(row(m))
    fs = front(models, True, True)
    say("")
    say("## What rule S picks")
    say("")
    for m in fs:
        card, _ = card_for(m["mem"])
        say(f"- {card}: **{m['name']}** (rank {m['rank']}, Retrieval {m['score']:.2f}, {m['mem']:.1f} GB).")
    say("- Reranker: the Qwen3-Reranker of the same size as the chosen embedder (released with it, same licence, same instruction format).")
    say("")
    say("## What the other combinations would pick, so nothing is hidden")
    say("")
    f0 = front(models, False, False)[0]
    say(f"- Without C3 and C4: {f0['name']} (rank {f0['rank']}, {f0['score']:.2f}, {f0['mem']:.1f} GB). It is a Qwen3-Embedding-0.6B fine-tune with no report, nothing to cite and no statement of what it was trained on.")
    f1 = front(models, True, False)
    say("- With C3 only: " + " -> ".join(f"{m['name']} ({m['score']:.2f}, {m['mem']:.1f} GB, zero-shot {'NA' if m['zs'] is None else int(m['zs'])})" for m in f1) + ". The laptop model here declares 21 of the board's 41 datasets as training data, so its Retrieval number is partly a score on its own training data.")
    with open(args.out_report, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    write_macros(models, fs, args.out_tex)


def write_macros(models, fs, path):
    """The paper's sentences about the rule, as LaTeX macros (latex/figures/model_numbers.tex): the download date,
    the conditions, what the rule picks per card, and the higher-scoring models the conditions excluded, with why."""
    with open(FILE_TASKS, encoding="utf-8") as fh:
        n_datasets = len(next(csv.reader(fh))) - 2  # columns after Rank and Model: one per dataset of the benchmark
    picks = []
    for m in fs:
        size = next(s for s, _, _ in CARDS if m["mem"] + HEADROOM_GB <= s)
        picks.append((size, m))
    picks.sort(key=lambda x: -x[0])
    excluded = []
    for size, pick in picks:
        for m in models:
            if m["rank"] >= pick["rank"] or m["mem"] + HEADROOM_GB > size or m in excluded:
                continue
            if not m["paper"] or m["zs"] is None or m["zs"] < ZS_MIN:
                excluded.append(m)
    excluded.sort(key=lambda m: m["rank"])

    def why(m):
        if not m["paper"]:
            return f"{m['name'].split('/')[-1]} has no technical report"
        if m["zs"] is None:
            return f"{m['name'].split('/')[-1]} does not declare its training data"
        return f"{m['name'].split('/')[-1]} declares {round(n_datasets * (100 - m['zs']) / 100)} of the {n_datasets} benchmark datasets as training data"
    d = DOWNLOAD_DATE
    month = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"][int(d[5:7]) - 1]
    macros = {
        "mtebDownloadDate": f"{int(d[8:10])} {month} {d[:4]}",
        "nMtebDatasets": str(n_datasets),
        "mtebMaxTrained": str(int(n_datasets * (100 - ZS_MIN) / 100)),
        "nModelsOpen": str(len(models)),
        "rulePicks": ", ".join(f"{m['name'].split('/')[-1]} for {size}~GB" for size, m in picks[:-1]) + (", and " if len(picks) > 2 else " and ") + f"{picks[-1][1]['name'].split('/')[-1]} for {picks[-1][0]}~GB",
        "nExcludedHigher": str(len(excluded)),
        "excludedHigher": "; ".join(why(m) for m in excluded),
    }
    lines = ["% generated by paper-code/model_selection/pareto_simple.py; do not edit by hand"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"macros -> {path}")


if __name__ == "__main__":
    main()
