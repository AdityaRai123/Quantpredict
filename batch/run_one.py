# #####################################################################################
# QUANTPREDICT - RUN THE FULL PIPELINE FOR ONE STOCK AND EXPORT EVERYTHING
#
#   py batch/run_one.py RELIANCE
#
# Runs Quantpredict.py UNCHANGED (via runpy) with the stock picked through the
# QP_STOCK_SYMBOL / QP_OUTPUT_DIR environment variables, then reads the finished
# pipeline's variables and writes a tidy per-stock folder:
#
#   outputs/<SYMBOL>/
#     plots/    the 10 matplotlib figures the pipeline already saves
#     tables/   results.csv, voting_comparison.csv, validation_scores.csv,
#               test_predictions.csv, price_history_adjusted.csv
#     data/     summary.json  (metrics, weights, checks, latest prediction ...)
#               series.json   (price history + test-period day-by-day series, for charts)
#
# Must be run from the project root (Quantpredict.py reads archive/ relatively).
# #####################################################################################
import os, sys, json, time, glob, shutil, runpy

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def r(v, nd=6):
    """Round a float for JSON; NaN/inf -> None."""
    v = float(v)
    return None if not np.isfinite(v) else round(v, nd)


def rl(arr, nd=6):
    return [r(v, nd) for v in np.asarray(arr, dtype=float).ravel()]


def day(d):
    return pd.Timestamp(d).strftime("%Y-%m-%d")


def metrics(m):
    return {k: r(v) for k, v in m.items()}


def export(g, symbol, out_dir, runtime_sec):
    data_dir = os.path.join(out_dir, "data")
    tables_dir = os.path.join(out_dir, "tables")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    from sklearn.metrics import confusion_matrix

    y_test = np.asarray(g["y_test_common"]).astype(int)
    all_8 = list(g["all_8_names"])
    all_7 = list(g["all_7_names"])
    split_idx = int(g["split_idx"])
    dates_model = pd.to_datetime(g["dates_model"]).reset_index(drop=True)
    test_dates = dates_model.iloc[split_idx:].reset_index(drop=True)
    df = g["df"]
    df_model = g["df_model"]

    # ---- company metadata --------------------------------------------------------
    meta = {}
    meta_path = os.path.join(ROOT, "archive", "stock_metadata.csv")
    if os.path.exists(meta_path):
        md = pd.read_csv(meta_path)
        row = md[md["Symbol"] == symbol]
        if len(row):
            meta = {"company": str(row["Company Name"].iloc[0]),
                    "industry": str(row["Industry"].iloc[0]).title(),
                    "isin": str(row["ISIN Code"].iloc[0])}

    # ---- result tables -----------------------------------------------------------
    final_table = g["final_table"]
    results = [{"name": name, "kind": row["kind"],
                **{k: r(row[k]) for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]}}
               for name, row in final_table.iterrows()]

    def best(kind):
        sub = final_table[final_table["kind"] == kind]
        return {"name": sub.index[0],
                **{k: r(sub.iloc[0][k]) for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]}}

    voting_table = g["voting_table"]
    voting = [{k: (r(v) if isinstance(v, (float, np.floating)) else
                   int(v) if isinstance(v, (np.integer,)) else v)
               for k, v in rec.items()} for rec in voting_table.to_dict("records")]

    # ---- confusion matrices for every model and ensemble -------------------------
    all_preds = {**{n: g["model_predictions"][n] for n in all_8}, **g["ensemble_predictions"]}
    confusion = {n: confusion_matrix(y_test, np.asarray(p).astype(int), labels=[0, 1]).tolist()
                 for n, p in all_preds.items()}

    # ---- adaptive voting ----------------------------------------------------------
    adaptive = {}
    for key, a in g["adaptive"].items():
        W = np.asarray(a["test_weights"])
        adaptive[key] = {
            "names": list(a["names"]), "window": int(a["window"]), "beta": r(a["beta"]),
            "grid": [{k: r(v) for k, v in rec.items()} for rec in a["grid"].to_dict("records")],
            "next_weights": rl(a["next_weights"]),
            "mean_weights": rl(W.mean(axis=0)),
        }

    # ---- latest (next-day) prediction --------------------------------------------
    latest_row = g["latest_row"]
    aw7 = g["adaptive"]["7"]["next_weights"]
    latest = {
        "date": day(latest_row["Date"]),
        "close": r(latest_row["Close"], 4),
        "models": {n: {"pred": int(g["latest_preds"][n]), "prob": r(g["latest_probs"][n])} for n in all_8},
        "ensembles": {
            "Majority Voting (7)": {"pred": int(g["maj7"]), "detail": f"{int(g['votes_up_7'])}/7 models voted UP"},
            "Weighted Voting - labels (7)": {"pred": int(g["w_label_7"] > 0.5), "score": r(g["w_label_7"])},
            "Weighted Voting - probabilities (7)": {"pred": int(g["w_prob_7"] >= g["THRESHOLD"]), "score": r(g["w_prob_7"])},
            "Adaptive Weighted - labels (7)": {"pred": int(g["a_label_7"] > 0.5), "score": r(g["a_label_7"])},
            "Adaptive Weighted - probabilities (7)": {"pred": int(g["a_prob_7"] >= g["THRESHOLD"]), "score": r(g["a_prob_7"])},
            "Majority Voting (8) [experimental]": {"pred": int(g["maj8"]), "detail": f"{int(g['votes_up_8'])}/8 models voted UP"},
            "Weighted Voting - probabilities (8) [experimental]": {"pred": int(g["w_prob_8"] >= g["THRESHOLD"]), "score": r(g["w_prob_8"])},
        },
        "adaptive_weights_7": {n: r(w) for n, w in zip(all_7, aw7)},
    }

    fs_val = g["fs_val_df"]
    summary = {
        "symbol": symbol,
        **meta,
        "dataset": g["DATASET_NAME"],
        "dl_backend": g["DL_BACKEND"],
        "runtime_sec": round(runtime_sec, 1),
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "rows": {"clean": int(len(df)), "model": int(len(df_model))},
        "date_range": {"start": day(df["Date"].iloc[0]), "end": day(df["Date"].iloc[-1])},
        "train": {"start": day(g["TRAIN_START"]), "end": day(g["TRAIN_END"]), "n": split_idx},
        "test": {"start": day(g["TEST_START"]), "end": day(g["TEST_END"]), "n": int(len(y_test))},
        "class_balance": {"down": r(g["class_balance"].get(0, 0)), "up": r(g["class_balance"].get(1, 0))},
        "test_up_rate": r(y_test.mean()),
        "majority_baseline": r(g["majority_class_rate"]),
        "noise_margin": r(g["margin"]),
        "corporate_actions": [{"date": str(d), "gap": r(raw, 4), "factor": r(f, 4)}
                              for d, raw, f in g["corporate_actions"]],
        "features": list(g["feature_columns"]),
        "feature_selection": {
            "top_k": int(g["TOP_K_FEATURES"]),
            "mi_scores": rl(g["mi_scores"]),
            "tree_importances": rl(g["tree_importances"]),
            "mi_selected": [g["feature_columns"][i] for i in g["mi_indices"]],
            "tree_selected": [g["feature_columns"][i] for i in g["tree_indices"]],
            "best_set": g["BEST_FEATURE_SET"],
            "eval_set": g["FS_EVAL_SET"],
            "walk_forward_f1": {str(c): {str(i): r(v) for i, v in fs_val[c].items()} for c in fs_val.columns},
        },
        "results": results,
        "best": {"individual": best("Individual"), "ensemble": best("Ensemble"),
                 "feature_selected": best("Feature-selected")},
        "voting": voting,
        "validation": {n: metrics(m) for n, m in g["validation_scores"].items()},
        "weights": {"static_7": {n: r(w) for n, w in zip(all_7, g["weights_7"])},
                    "static_8": {n: r(w) for n, w in zip(all_8, g["weights_8"])}},
        "adaptive": adaptive,
        "confusion": confusion,
        "training_curves": {
            "lstm": {k: rl(v) for k, v in g["lstm_history"].items() if k in ("loss", "val_loss")},
            "vit": {k: rl(v) for k, v in g["vit_history"].items() if k in ("loss", "val_loss")},
        },
        "latest": latest,
        "checks": [{"label": label, "pass": bool(ok)} for label, ok in g["checks"]],
    }

    # ---- day-by-day series for interactive charts --------------------------------
    test_close = df_model["Close"].iloc[split_idx:].to_numpy()
    series = {
        "price": {"date": [day(d) for d in df["Date"]],
                  "close": rl(df["Close"], 2), "volume": [int(v) for v in df["Volume"]]},
        "test": {
            "date": [day(d) for d in test_dates],
            "close": rl(test_close, 2),
            "actual": y_test.tolist(),
            "model_prob": {n: rl(g["model_probabilities"][n], 4) for n in all_8},
            "model_pred": {n: np.asarray(g["model_predictions"][n]).astype(int).tolist() for n in all_8},
            "ensemble_pred": {n: np.asarray(p).astype(int).tolist()
                              for n, p in g["ensemble_predictions"].items()},
            "adaptive_weights_7": {n: rl(g["adaptive"]["7"]["test_weights"][:, j], 4)
                                   for j, n in enumerate(all_7)},
        },
    }

    with open(os.path.join(data_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(data_dir, "series.json"), "w", encoding="utf-8") as f:
        json.dump(series, f, separators=(",", ":"))

    # ---- CSV tables for download -------------------------------------------------
    preds = pd.DataFrame({"Date": series["test"]["date"], "Close": test_close, "Actual_Up": y_test})
    for n in all_8:
        preds[f"{n} P(UP)"] = np.round(np.asarray(g["model_probabilities"][n], dtype=float), 6)
        preds[f"{n} pred"] = np.asarray(g["model_predictions"][n]).astype(int)
    for n, p in g["ensemble_predictions"].items():
        preds[f"{n} pred"] = np.asarray(p).astype(int)
    preds.to_csv(os.path.join(tables_dir, "test_predictions.csv"), index=False)

    pd.DataFrame(g["validation_scores"]).T.to_csv(os.path.join(tables_dir, "validation_scores.csv"))
    df[["Date", "Open", "High", "Low", "Close", "Volume"]].to_csv(
        os.path.join(tables_dir, "price_history_adjusted.csv"), index=False)


def main():
    symbol = sys.argv[1].upper()
    out_dir = os.path.join(ROOT, "outputs", symbol)
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    os.environ["QP_STOCK_SYMBOL"] = symbol
    os.environ["QP_OUTPUT_DIR"] = plots_dir      # the pipeline saves PNGs + 2 CSVs here
    os.environ.setdefault("MPLBACKEND", "Agg")   # save figures without opening windows
    os.chdir(ROOT)

    t0 = time.time()
    g = runpy.run_path(os.path.join(ROOT, "Quantpredict.py"), run_name="__main__")
    runtime = time.time() - t0

    # The pipeline must have loaded the requested stock, not its fallback file.
    loaded = os.path.splitext(os.path.basename(g["filename"]))[0].upper()
    if loaded != symbol:
        raise RuntimeError(f"Pipeline loaded {g['filename']} instead of {symbol}.csv")

    # Move the pipeline's CSVs from plots/ to tables/ and drop the "<SYMBOL>_" prefix.
    tables_dir = os.path.join(out_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    for path in glob.glob(os.path.join(plots_dir, f"{symbol}_*")):
        name = os.path.basename(path)[len(symbol) + 1:]
        dest = os.path.join(tables_dir if name.endswith(".csv") else plots_dir, name)
        shutil.move(path, dest)

    export(g, symbol, out_dir, runtime)
    print(f"\n[run_one] {symbol} exported to {out_dir} in {runtime / 60:.1f} min")


if __name__ == "__main__":
    main()
