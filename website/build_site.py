# #####################################################################################
# QUANTPREDICT - BUILD THE STATIC WEBSITE FROM outputs/
#
#   py website/build_site.py
#
# Reads every finished stock in outputs/ and writes a self-contained static site:
#
#   docs/                         (GitHub Pages can serve this folder as-is)
#     index.html                  full HTML document (any static host / local server)
#     data/overview.json          cross-stock tables for the "Results" section
#     data/stocks/<SYMBOL>.json   one stock's summary + day-by-day series
#     downloads/stocks/<SYMBOL>.zip         plots + tables + data + log for one stock
#     downloads/quantpredict_all_results.zip every stock's tables + the cross-stock summary
#   website/artifact.html         the same page as a fragment, for publishing as an Artifact
#
# The page itself is assembled from website/src/ (template.html, styles.css, app.js);
# CSS and JS are inlined so the page is a single file next to its data.
# #####################################################################################
import os, re, json, glob, shutil, zipfile, subprocess, sys, time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS = os.path.join(ROOT, "outputs")
SRC = os.path.join(ROOT, "website", "src")
SITE = os.path.join(ROOT, "docs")

MODELS = ["Logistic Regression", "Random Forest", "Gradient Boosting", "XGBoost",
          "SVM", "KNN", "LSTM", "Vision Transformer"]

# Every ensemble row the pipeline registers, with a short label and its family.
ENSEMBLES = [
    ("Majority Voting (6 classical)", "Majority", 6, "labels"),
    ("Majority Voting (7 = 6 classical + LSTM)", "Majority", 7, "labels"),
    ("Majority Voting (8 = 7 + ViT) [experimental]", "Majority", 8, "labels"),
    ("Soft Voting (7 = 6 classical + LSTM)", "Soft", 7, "probabilities"),
    ("Soft Voting (8 = 7 + ViT) [experimental]", "Soft", 8, "probabilities"),
    ("Weighted Voting - labels (7 models)", "Weighted", 7, "labels"),
    ("Weighted Voting - probabilities (7 models)", "Weighted", 7, "probabilities"),
    ("Weighted Voting - labels (8 models) [experimental]", "Weighted", 8, "labels"),
    ("Weighted Voting - probabilities (8 models) [experimental]", "Weighted", 8, "probabilities"),
    ("Adaptive Weighted Voting - labels (7 models)", "Adaptive", 7, "labels"),
    ("Adaptive Weighted Voting - probabilities (7 models)", "Adaptive", 7, "probabilities"),
    ("Adaptive Weighted Voting - labels (8 models) [experimental]", "Adaptive", 8, "labels"),
    ("Adaptive Weighted Voting - probabilities (8 models) [experimental]", "Adaptive", 8, "probabilities"),
    ("Stacking (7 = 6 classical + LSTM)", "Stacking", 7, "probabilities"),
    ("Stacking (8 = 7 + ViT) [experimental]", "Stacking", 8, "probabilities"),
]
ENSEMBLE_INFO = {n: {"family": f, "size": k, "rule": r} for n, f, k, r in ENSEMBLES}
MAJ7 = "Majority Voting (7 = 6 classical + LSTM)"


def r6(v):
    if v is None:
        return None
    v = float(v)
    return None if not np.isfinite(v) else round(v, 6)


def load_stocks():
    stocks = []
    for path in sorted(glob.glob(os.path.join(OUTPUTS, "*", "data", "summary.json"))):
        d = os.path.dirname(path)
        with open(path, encoding="utf-8") as f:
            summary = json.load(f)
        with open(os.path.join(d, "series.json"), encoding="utf-8") as f:
            series = json.load(f)
        stocks.append((summary, series))
    return stocks


def build_overview(stocks):
    rows = []
    for s, _ in stocks:
        for r in s["results"]:
            rows.append({"symbol": s["symbol"], "baseline": s["majority_baseline"], **r})
    long = pd.DataFrame(rows)

    # ---- one record per stock ------------------------------------------------------
    stock_recs = []
    for s, _ in stocks:
        acc = {r["name"]: r["accuracy"] for r in s["results"]}
        ind = {m: acc[m] for m in MODELS}
        stock_recs.append({
            "symbol": s["symbol"], "company": s.get("company", s["symbol"]),
            "industry": s.get("industry", ""),
            "start": s["date_range"]["start"], "end": s["date_range"]["end"],
            "test_start": s["test"]["start"], "test_end": s["test"]["end"], "test_n": s["test"]["n"],
            "baseline": s["majority_baseline"], "margin": s["noise_margin"],
            "best_individual": s["best"]["individual"]["name"],
            "best_individual_acc": s["best"]["individual"]["accuracy"],
            "best_ensemble": s["best"]["ensemble"]["name"],
            "best_ensemble_acc": s["best"]["ensemble"]["accuracy"],
            "mean_individual_acc": r6(np.mean(list(ind.values()))),
            "maj7_acc": acc[MAJ7],
            "latest_date": s["latest"]["date"],
            "latest_votes_up_7": sum(s["latest"]["models"][m]["pred"] for m in MODELS[:7]),
            "corporate_actions": len(s["corporate_actions"]),
            "checks_passed": sum(c["pass"] for c in s["checks"]),
            "checks_total": len(s["checks"]),
        })

    # ---- per-method averages across stocks --------------------------------------------
    methods = []
    for name, g in long.groupby("name"):
        kind = g["kind"].iloc[0]
        rec = {"name": name, "kind": kind, "n": int(len(g)),
               "mean_acc": r6(g["accuracy"].mean()), "median_acc": r6(g["accuracy"].median()),
               "std_acc": r6(g["accuracy"].std()), "min_acc": r6(g["accuracy"].min()),
               "max_acc": r6(g["accuracy"].max()),
               "mean_f1": r6(g["f1"].mean()), "mean_auc": r6(g["roc_auc"].mean()),
               "beats_baseline": int((g["accuracy"] > g["baseline"]).sum()),
               "beats_50": int((g["accuracy"] > 0.5).sum())}
        if name in ENSEMBLE_INFO:
            rec.update(ENSEMBLE_INFO[name])
        if kind == "Feature-selected":
            m = re.match(r"(.+) \[(MI|Tree) Top-\d+\]$", name)
            rec.update({"model": m.group(1), "set": m.group(2)})
        methods.append(rec)
    methods.sort(key=lambda m: -m["mean_acc"])

    acc_matrix = {sym: {r["name"]: r["accuracy"] for r in g.to_dict("records")}
                  for sym, g in long.groupby("symbol")}

    # ---- RQ1: which single model wins most often ---------------------------------------
    best_counts = {m: 0 for m in MODELS}
    for rec in stock_recs:
        best_counts[rec["best_individual"]] += 1

    # ---- RQ2: 7-model vs 8-model (with ViT) ensembles, paired by stock -------------------
    vit_pairs = []
    for fam, rule in [("Majority", "labels"), ("Soft", "probabilities"), ("Weighted", "labels"),
                      ("Weighted", "probabilities"), ("Adaptive", "labels"),
                      ("Adaptive", "probabilities"), ("Stacking", "probabilities")]:
        n7 = next(n for n, f, k, r in ENSEMBLES if f == fam and k == 7 and r == rule)
        n8 = next(n for n, f, k, r in ENSEMBLES if f == fam and k == 8 and r == rule)
        d = np.array([acc_matrix[s][n8] - acc_matrix[s][n7] for s in acc_matrix])
        vit_pairs.append({"family": fam, "rule": rule, "name7": n7, "name8": n8,
                          "mean7": r6(np.mean([acc_matrix[s][n7] for s in acc_matrix])),
                          "mean8": r6(np.mean([acc_matrix[s][n8] for s in acc_matrix])),
                          "mean_delta": r6(d.mean()), "wins8": int((d > 0).sum()),
                          "ties": int((d == 0).sum()), "n": int(len(d))})

    # ---- RQ3: each ensemble vs majority voting and vs the stock's best single model -------
    ens_vs = []
    for name, fam, k, rule in ENSEMBLES:
        a = np.array([acc_matrix[s][name] for s in acc_matrix])
        maj = np.array([acc_matrix[s][MAJ7] for s in acc_matrix])
        best = np.array([max(acc_matrix[s][m] for m in MODELS) for s in acc_matrix])
        mean_ind = np.array([np.mean([acc_matrix[s][m] for m in MODELS]) for s in acc_matrix])
        ens_vs.append({"name": name, "family": fam, "size": k, "rule": rule,
                       "mean_acc": r6(a.mean()),
                       "delta_vs_maj7": r6((a - maj).mean()), "wins_vs_maj7": int((a > maj).sum()),
                       "delta_vs_best_single": r6((a - best).mean()),
                       "wins_vs_best_single": int((a > best).sum()),
                       "delta_vs_mean_single": r6((a - mean_ind).mean()),
                       "wins_vs_mean_single": int((a > mean_ind).sum())})

    # ---- RQ4: feature selection, per model (only stocks where that variant exists) --------
    fs = []
    for m in MODELS:
        rec = {"model": m, "all": r6(np.mean([acc_matrix[s][m] for s in acc_matrix]))}
        for tag in ["MI", "Tree"]:
            key = f"{m} [{tag} Top-20]"
            pairs = [(acc_matrix[s][key], acc_matrix[s][m]) for s in acc_matrix if key in acc_matrix[s]]
            rec[tag.lower()] = r6(np.mean([p[0] for p in pairs])) if pairs else None
            rec[f"{tag.lower()}_all"] = r6(np.mean([p[1] for p in pairs])) if pairs else None
            rec[f"{tag.lower()}_n"] = len(pairs)
            rec[f"{tag.lower()}_wins"] = sum(p[0] > p[1] for p in pairs)
        fs.append(rec)
    fs_set_counts = {}
    for s, _ in stocks:
        k = s["feature_selection"]["best_set"]
        fs_set_counts[k] = fs_set_counts.get(k, 0) + 1

    # ---- the test period pooled over every stock -------------------------------------
    test_days = int(sum(s["test"]["n"] for s, _ in stocks))
    up_days = int(round(sum(s["test_up_rate"] * s["test"]["n"] for s, _ in stocks)))
    margins = [s["noise_margin"] for s, _ in stocks]

    return {
        "generated": time.strftime("%Y-%m-%d"),
        "n_stocks": len(stocks),
        "models": MODELS,
        "ensembles": [{"name": n, "family": f, "size": k, "rule": r} for n, f, k, r in ENSEMBLES],
        "first_date": min(s["date_range"]["start"] for s, _ in stocks),
        "last_date": max(s["date_range"]["end"] for s, _ in stocks),
        "test_days_total": test_days, "test_up_rate": r6(up_days / test_days),
        "margin_range": [r6(min(margins)), r6(max(margins))],
        "n_features": len(stocks[0][0]["features"]),
        "n_checks": len(stocks[0][0]["checks"]),
        "checks_all_passed": int(sum(rec["checks_passed"] == rec["checks_total"] for rec in stock_recs)),
        "dl_backend": stocks[0][0]["dl_backend"],
        "stocks": stock_recs,
        "methods": methods,
        "acc": acc_matrix,
        "best_counts": best_counts,
        "vit_pairs": vit_pairs,
        "ensemble_vs": ens_vs,
        "feature_selection": fs,
        "fs_best_set_counts": fs_set_counts,
    }


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, separators=(",", ":"))


def build_zips(stocks):
    dl = os.path.join(SITE, "downloads")
    stock_dir = os.path.join(dl, "stocks")
    os.makedirs(stock_dir, exist_ok=True)
    sizes = {}
    for s, _ in stocks:
        sym = s["symbol"]
        src = os.path.join(OUTPUTS, sym)
        dest = os.path.join(stock_dir, f"{sym}.zip")
        with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
            for sub in ["plots", "tables", "data"]:
                for p in sorted(glob.glob(os.path.join(src, sub, "*"))):
                    z.write(p, f"{sym}/{sub}/{os.path.basename(p)}")
            z.write(os.path.join(src, "log.txt"), f"{sym}/pipeline_log.txt")
            z.writestr(f"{sym}/README.txt", STOCK_README.format(sym=sym))
        sizes[sym] = os.path.getsize(dest)

    all_zip = os.path.join(dl, "quantpredict_all_results.zip")
    with zipfile.ZipFile(all_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(glob.glob(os.path.join(OUTPUTS, "_summary", "*.csv"))):
            z.write(p, f"quantpredict_all_results/summary/{os.path.basename(p)}")
        for s, _ in stocks:
            sym = s["symbol"]
            for p in all_results_tables(sym):
                z.write(p, f"quantpredict_all_results/stocks/{sym}/{os.path.basename(p)}")
            z.write(os.path.join(OUTPUTS, sym, "data", "summary.json"),
                    f"quantpredict_all_results/stocks/{sym}/summary.json")
        z.writestr("quantpredict_all_results/README.txt", ALL_README)
    sizes["_all"] = os.path.getsize(all_zip)
    return sizes


def all_results_tables(sym):
    """Result tables for the all-stocks download (the raw price history is left to the per-stock ZIP)."""
    return [p for p in sorted(glob.glob(os.path.join(OUTPUTS, sym, "tables", "*.csv")))
            if not p.endswith("price_history_adjusted.csv")]


def build_bundles(stocks):
    """Artifact hosting cannot serve .zip files, so the same contents go out as JSON bundles
    (text as-is, PNGs base64) and the page assembles the ZIP in the viewer's browser.
    data/summary.json + series.json are added by the page from data/stocks/<SYMBOL>.json."""
    import base64
    out = os.path.join(ROOT, "website", "artifact_files", "bundles")
    os.makedirs(out, exist_ok=True)

    def text(path):
        with open(path, encoding="utf-8") as f:
            return f.read()

    for s, _ in stocks:
        sym = s["symbol"]
        src = os.path.join(OUTPUTS, sym)
        files = []
        for p in sorted(glob.glob(os.path.join(src, "plots", "*.png"))):
            with open(p, "rb") as f:
                files.append({"path": f"{sym}/plots/{os.path.basename(p)}", "b64": base64.b64encode(f.read()).decode()})
        for p in sorted(glob.glob(os.path.join(src, "tables", "*.csv"))):
            files.append({"path": f"{sym}/tables/{os.path.basename(p)}", "text": text(p)})
        files.append({"path": f"{sym}/pipeline_log.txt", "text": text(os.path.join(src, "log.txt"))})
        files.append({"path": f"{sym}/README.txt", "text": STOCK_README.format(sym=sym)})
        write_json(os.path.join(out, f"{sym}.json"), {"files": files})

    files = [{"path": f"quantpredict_all_results/summary/{os.path.basename(p)}", "text": text(p)}
             for p in sorted(glob.glob(os.path.join(OUTPUTS, "_summary", "*.csv")))]
    for s, _ in stocks:
        sym = s["symbol"]
        for p in all_results_tables(sym):
            files.append({"path": f"quantpredict_all_results/stocks/{sym}/{os.path.basename(p)}", "text": text(p)})
        files.append({"path": f"quantpredict_all_results/stocks/{sym}/summary.json",
                      "text": text(os.path.join(OUTPUTS, sym, "data", "summary.json"))})
    files.append({"path": "quantpredict_all_results/README.txt", "text": ALL_README})
    write_json(os.path.join(out, "quantpredict_all_results.json"), {"files": files})
    sizes = {os.path.basename(p): os.path.getsize(p) for p in glob.glob(os.path.join(out, "*.json"))}
    print(f"[build_site] artifact bundles: {len(sizes)} files, {sum(sizes.values()) / 1e6:.1f} MB, "
          f"largest {max(sizes.values()) / 1e6:.1f} MB")


STOCK_README = """QuantPredict - full pipeline output for {sym}

plots/   the figures Quantpredict.py draws (price/volume history, model and ensemble
         accuracy, F1, confusion matrices, feature rankings, LSTM/ViT training curves,
         voting comparison, adaptive weights over time)
tables/  results.csv               every model, ensemble and feature-selected variant on the test period
         voting_comparison.csv     majority vs weighted vs adaptive weighted voting (STEP 18)
         validation_scores.csv     walk-forward validation scores (training period only)
         test_predictions.csv      day-by-day P(UP) and UP/DOWN calls for every model and ensemble
         price_history_adjusted.csv OHLCV after split/bonus back-adjustment
data/    summary.json / series.json  the same results in machine-readable form
pipeline_log.txt  the complete console output of the run

Next-day direction is close to a coin flip; nothing here is financial advice.
"""

ALL_README = """QuantPredict - results for every NIFTY-50 stock in the dataset

summary/all_stocks.csv       one row per stock: best model, best ensemble, baseline, key accuracies
summary/all_results.csv      every stock x every model / ensemble / feature-selected variant
summary/method_averages.csv  each method averaged over all stocks
stocks/<SYMBOL>/             that stock's result tables and summary.json

Plots and the adjusted price history for each stock are in the per-stock downloads.
Next-day direction is close to a coin flip; nothing here is financial advice.
"""


def build_page():
    """Inline styles.css + app.js into template.html; write a fragment and a full document."""
    with open(os.path.join(SRC, "template.html"), encoding="utf-8") as f:
        page = f.read()
    with open(os.path.join(SRC, "styles.css"), encoding="utf-8") as f:
        css = f.read()
    with open(os.path.join(SRC, "app.js"), encoding="utf-8") as f:
        js = f.read()
    page = page.replace("/*__STYLES__*/", css).replace("/*__APP__*/", js)

    with open(os.path.join(ROOT, "website", "artifact.html"), "w", encoding="utf-8") as f:
        f.write(page)

    # Full document for ordinary static hosting: move <title>, font links and <style> into <head>.
    head_parts = []
    for pattern in [r"<title>.*?</title>", r"<link [^>]*>", r"<style>.*?</style>"]:
        for m in re.findall(pattern, page, flags=re.S):
            head_parts.append(m)
            page = page.replace(m, "", 1)
    doc = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
           "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1,viewport-fit=cover\">\n"
           + "\n".join(head_parts) + "\n</head>\n<body>\n" + page.strip() + "\n</body>\n</html>\n")
    with open(os.path.join(SITE, "index.html"), "w", encoding="utf-8") as f:
        f.write(doc)


def main():
    if "--page-only" in sys.argv:          # quick rebuild of index.html / artifact.html only
        build_page()
        print("[build_site] page rebuilt")
        return
    subprocess.run([sys.executable, os.path.join(ROOT, "batch", "aggregate.py")], cwd=ROOT, check=True)
    stocks = load_stocks()
    if not stocks:
        raise SystemExit("No finished stocks in outputs/ yet.")
    os.makedirs(SITE, exist_ok=True)

    for s, series in stocks:
        write_json(os.path.join(SITE, "data", "stocks", f"{s['symbol']}.json"),
                   {"summary": s, "series": series})
    sizes = build_zips(stocks)
    build_bundles(stocks)
    overview = build_overview(stocks)
    overview["download_bytes"] = sizes
    write_json(os.path.join(SITE, "data", "overview.json"), overview)
    build_page()
    # GitHub Pages: serve files as-is (no Jekyll processing).
    open(os.path.join(SITE, ".nojekyll"), "w").close()

    total = sum(os.path.getsize(p) for p in glob.glob(os.path.join(SITE, "**", "*"), recursive=True)
                if os.path.isfile(p))
    print(f"[build_site] {len(stocks)} stocks -> {SITE}  ({total / 1e6:.1f} MB total, "
          f"all-results zip {sizes['_all'] / 1e6:.1f} MB, largest stock zip "
          f"{max(v for k, v in sizes.items() if k != '_all') / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
