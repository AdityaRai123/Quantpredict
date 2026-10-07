# QuantPredict website

A static results site for the QuantPredict pipeline: what the project set out to do,
the results across every NIFTY-50 stock, a per-stock explorer, and ZIP downloads of
every figure and table.

## Rebuild from scratch

Run these from the project root.

```bash
py batch/run_all.py
```

Runs `Quantpredict.py` (unchanged) for every stock in `archive/`, 4 at a time with
8 threads each, and writes `outputs/<SYMBOL>/` (plots, tables, data, log). It takes
about 45 minutes on a 32-thread machine. Finished stocks are skipped, so you can stop
it and run it again to resume. Progress is in `outputs/_batch.log`; `--force` re-runs
everything, `--only TCS,INFY` runs a subset, `--workers` / `--threads` tune parallelism.

```bash
py website/build_site.py
```

Turns `outputs/` into the website in `docs/`: `index.html`, `data/*.json` and
`downloads/*.zip`. It also writes `website/artifact.html`, the same page without the
`<html>/<head>` wrapper, plus `website/artifact_files/bundles/*.json`, which are used for
publishing on claude.ai (artifact hosting cannot serve .zip files, so the page builds the ZIP
in the viewer's browser from those bundles).
`--page-only` rebuilds only the HTML after editing `website/src/`.

## Preview locally

```bash
py -m http.server 8765 --directory docs
```

Then open http://localhost:8765. Opening `docs/index.html` straight from disk will not
work, because the page loads its data with `fetch()`.

## Host on GitHub Pages

1. Commit `docs/` and push to GitHub.
2. In the repository: Settings → Pages → Build and deployment → Deploy from a branch →
   `main` / `/docs`.
3. The site appears at `https://<user>.github.io/<repo>/`.

`docs/` is about 100 MB, mostly the per-stock ZIPs. Every file stays under GitHub's
100 MB limit. GitHub Pages needs a public repository on a free plan.

## Where things live

| Path | What it is |
|---|---|
| `batch/run_one.py` | Runs the pipeline for one stock and exports JSON/CSV |
| `batch/run_all.py` | Parallel, resumable batch over all stocks |
| `batch/aggregate.py` | Cross-stock tables in `outputs/_summary/` |
| `website/src/` | Page source: `template.html`, `styles.css`, `app.js` |
| `website/build_site.py` | Builds `docs/` from `outputs/` |
