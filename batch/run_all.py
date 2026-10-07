# #####################################################################################
# QUANTPREDICT - RUN EVERY STOCK IN archive/ (OVERNIGHT BATCH)
#
#   py batch/run_all.py                    # all stocks, 4 at a time, 8 threads each
#   py batch/run_all.py --workers 3 --threads 10
#   py batch/run_all.py --only TCS,INFY    # a subset
#   py batch/run_all.py --force            # re-run stocks that already finished
#
# Each stock runs in its own process (batch/run_one.py), so one failure never
# stops the batch and memory is released between stocks. Finished stocks are
# skipped on a re-run, so the batch can be stopped and resumed at any time.
#
# Progress:  outputs/_batch.log           one line per finished/failed stock
#            outputs/_batch_status.json   machine-readable status
#            outputs/<SYMBOL>/log.txt     full pipeline output for that stock
# At the end batch/aggregate.py builds outputs/_summary/ (cross-stock tables).
# #####################################################################################
import os, sys, json, time, argparse, subprocess, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE = os.path.join(ROOT, "archive")
OUTPUTS = os.path.join(ROOT, "outputs")
NON_STOCK_FILES = {"NIFTY50_all.csv", "stock_metadata.csv"}   # same rule as Quantpredict.py
TIMEOUT_SEC = 4 * 3600

lock = threading.Lock()


def stock_symbols():
    """Every real single-stock CSV, biggest file first (longest runs start earliest)."""
    files = [f for f in os.listdir(ARCHIVE)
             if f.endswith(".csv") and f not in NON_STOCK_FILES
             and os.path.getsize(os.path.join(ARCHIVE, f)) > 1024]
    files.sort(key=lambda f: -os.path.getsize(os.path.join(ARCHIVE, f)))
    return [os.path.splitext(f)[0] for f in files]


def is_done(sym):
    return os.path.exists(os.path.join(OUTPUTS, sym, "data", "summary.json"))


def log_line(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(os.path.join(OUTPUTS, "_batch.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_status(status):
    with open(os.path.join(OUTPUTS, "_batch_status.json"), "w", encoding="utf-8") as f:
        json.dump(status, f, indent=1)


def run_stock(sym, threads, status):
    out_dir = os.path.join(OUTPUTS, sym)
    os.makedirs(out_dir, exist_ok=True)
    with lock:
        status["running"].append(sym)
        write_status(status)
    env = {**os.environ,
           "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads),
           "OPENBLAS_NUM_THREADS": str(threads), "LOKY_MAX_CPU_COUNT": str(threads),
           "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg"}
    t0 = time.time()
    with open(os.path.join(out_dir, "log.txt"), "w", encoding="utf-8") as log:
        try:
            rc = subprocess.run([sys.executable, os.path.join(ROOT, "batch", "run_one.py"), sym],
                                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                timeout=TIMEOUT_SEC).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    ok = rc == 0 and is_done(sym)
    return sym, ok, rc, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--only", default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUTPUTS, exist_ok=True)
    symbols = stock_symbols()
    if args.only:
        wanted = {s.strip().upper() for s in args.only.split(",")}
        symbols = [s for s in symbols if s in wanted]
    todo = [s for s in symbols if args.force or not is_done(s)]

    status = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "total": len(symbols),
              "done": [s for s in symbols if s not in todo], "failed": {}, "running": [],
              "minutes": {}, "finished": None}
    write_status(status)
    log_line(f"BATCH START  {len(todo)} to run, {len(status['done'])} already done, "
             f"workers={args.workers} threads={args.threads}")

    t_batch = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_stock, s, args.threads, status) for s in todo]
        for fut in as_completed(futures):
            sym, ok, rc, secs = fut.result()
            with lock:
                status["running"].remove(sym)
                status["minutes"][sym] = round(secs / 60, 1)
                if ok:
                    status["done"].append(sym)
                    log_line(f"DONE    {sym:<12} {secs / 60:6.1f} min   "
                             f"({len(status['done'])}/{len(symbols)} done)")
                else:
                    status["failed"][sym] = str(rc)
                    log_line(f"FAILED  {sym:<12} {secs / 60:6.1f} min   exit={rc}  "
                             f"(see outputs/{sym}/log.txt)")
                write_status(status)

    status["running"] = []
    status["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_status(status)

    subprocess.run([sys.executable, os.path.join(ROOT, "batch", "aggregate.py")], cwd=ROOT)
    log_line(f"BATCH COMPLETE  {len(status['done'])} done, {len(status['failed'])} failed, "
             f"{(time.time() - t_batch) / 3600:.2f} h")


if __name__ == "__main__":
    main()
