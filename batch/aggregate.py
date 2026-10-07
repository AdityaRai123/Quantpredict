# #####################################################################################
# QUANTPREDICT - CROSS-STOCK SUMMARY
#
#   py batch/aggregate.py
#
# Reads every outputs/<SYMBOL>/data/summary.json and writes outputs/_summary/:
#   all_stocks.csv     one row per stock: best model / ensemble, baseline, key accuracies
#   all_results.csv    long table: every stock x every model/ensemble x every metric
#   overview.json      the same, compact, for the website (plus per-method averages)
# Safe to run at any time; it only reads finished stocks.
# #####################################################################################
import os, glob, json

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS = os.path.join(ROOT, "outputs")
SUMMARY_DIR = os.path.join(OUTPUTS, "_summary")


def main():
    summaries = []
    for path in sorted(glob.glob(os.path.join(OUTPUTS, "*", "data", "summary.json"))):
        with open(path, encoding="utf-8") as f:
            summaries.append(json.load(f))
    if not summaries:
        print("[aggregate] no finished stocks yet")
        return
    os.makedirs(SUMMARY_DIR, exist_ok=True)

    long_rows, stock_rows = [], []
    for s in summaries:
        for row in s["results"]:
            long_rows.append({"symbol": s["symbol"], **row})
        acc = {row["name"]: row["accuracy"] for row in s["results"]}
        b = s["best"]
        stock_rows.append({
            "symbol": s["symbol"],
            "company": s.get("company", s["symbol"]),
            "industry": s.get("industry", ""),
            "start": s["date_range"]["start"], "end": s["date_range"]["end"],
            "test_start": s["test"]["start"], "test_days": s["test"]["n"],
            "majority_baseline": s["majority_baseline"],
            "noise_margin": s["noise_margin"],
            "best_individual": b["individual"]["name"],
            "best_individual_acc": b["individual"]["accuracy"],
            "best_ensemble": b["ensemble"]["name"],
            "best_ensemble_acc": b["ensemble"]["accuracy"],
            "ensemble_minus_individual": round(b["ensemble"]["accuracy"] - b["individual"]["accuracy"], 6),
            "majority_vote_7_acc": acc.get("Majority Voting (7 = 6 classical + LSTM)"),
            "weighted_prob_7_acc": acc.get("Weighted Voting - probabilities (7 models)"),
            "adaptive_prob_7_acc": acc.get("Adaptive Weighted Voting - probabilities (7 models)"),
            "stacking_7_acc": acc.get("Stacking (7 = 6 classical + LSTM)"),
            "corporate_actions": len(s["corporate_actions"]),
            "checks_passed": sum(c["pass"] for c in s["checks"]),
            "checks_total": len(s["checks"]),
            "latest_date": s["latest"]["date"],
            "latest_majority_7": "UP" if s["latest"]["ensembles"]["Majority Voting (7)"]["pred"] else "DOWN",
            "runtime_min": round(s["runtime_sec"] / 60, 1),
        })

    long_df = pd.DataFrame(long_rows)
    stock_df = pd.DataFrame(stock_rows).sort_values("symbol")
    long_df.to_csv(os.path.join(SUMMARY_DIR, "all_results.csv"), index=False)
    stock_df.to_csv(os.path.join(SUMMARY_DIR, "all_stocks.csv"), index=False)

    # Per-method averages across stocks (how each model/ensemble does "in general").
    methods = (long_df.groupby(["name", "kind"])
               .agg(mean_accuracy=("accuracy", "mean"), std_accuracy=("accuracy", "std"),
                    mean_f1=("f1", "mean"), mean_roc_auc=("roc_auc", "mean"),
                    stocks=("symbol", "count"))
               .reset_index().sort_values("mean_accuracy", ascending=False))
    # How often each method beats that stock's majority-class baseline.
    base = dict(zip(stock_df["symbol"], stock_df["majority_baseline"]))
    long_df["beats_baseline"] = long_df["accuracy"] > long_df["symbol"].map(base)
    methods = methods.merge(long_df.groupby("name")["beats_baseline"].mean().rename("beats_baseline_rate"),
                            left_on="name", right_index=True)
    methods.to_csv(os.path.join(SUMMARY_DIR, "method_averages.csv"), index=False)

    def clean(v):
        if isinstance(v, (float, np.floating)):
            return None if not np.isfinite(v) else round(float(v), 6)
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
        return v

    overview = {
        "n_stocks": len(stock_df),
        "stocks": [{k: clean(v) for k, v in rec.items()} for rec in stock_df.to_dict("records")],
        "methods": [{k: clean(v) for k, v in rec.items()} for rec in methods.to_dict("records")],
        "accuracy_matrix": {
            sym: {r["name"]: clean(r["accuracy"]) for r in grp.to_dict("records")}
            for sym, grp in long_df.groupby("symbol")
        },
    }
    with open(os.path.join(SUMMARY_DIR, "overview.json"), "w", encoding="utf-8") as f:
        json.dump(overview, f, indent=1)
    print(f"[aggregate] {len(stock_df)} stocks -> {SUMMARY_DIR}")


if __name__ == "__main__":
    main()
