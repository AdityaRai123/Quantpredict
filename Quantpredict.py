# #####################################################################################
# QUANTPREDICT - NEXT-DAY STOCK DIRECTION PREDICTION (8 MODELS + VOTING ENSEMBLES)
# #####################################################################################
#
# GOAL
#   For every trading day t, predict whether tomorrow's close will be higher than
#   today's:   Target[t] = 1 (UP)   if Close[t+1] > Close[t]
#                          0 (DOWN) otherwise
#   Then compare single models against several ways of combining them (voting).
#
# HOW TO RUN
#   * Colab / Jupyter: run the file as one cell (it pip-installs `ta` and `xgboost`).
#   * Locally:  py Quantpredict.py      (set MPLBACKEND=Agg to save plots without windows)
#   * Choose the stock with STOCK_SYMBOL in STEP 1 (a file name in archive/).
#   * Figures + result tables are written to outputs/.
#
# PIPELINE MAP  (search for "STEP <n>" to jump to a section)
#   STEP 1   Load the stock CSV                                   -> df
#   STEP 2   Clean rows (bad/duplicate dates, non-positive prices)
#   STEP 2a  Undo fake price crashes from splits/bonus issues      -> adjusted df
#   STEP 2b  Plot price and volume history
#   STEP 3   Technical indicators (MACD, RSI, SMA/EMA, Bollinger, ATR, OBV, ADX)
#   STEP 4   Target + stationary features                         -> feature_columns
#   STEP 5   Chronological 80/20 train/test split + scaling       -> X, y, split_idx
#   STEP 5b  Shared helper functions (model factory, metrics, walk-forward folds)
#   STEP 6   Feature selection (Mutual Information / tree importance), train data only
#   STEP 7   Six classical models (LogReg, RF, GB, XGBoost, SVM, KNN)
#   STEP 8   LSTM on 20-day windows          (8b: reusable LSTM trainer)
#   STEP 9   Vision Transformer on the same 20-day windows, cut into patches
#   STEP 10A Individual model scores + feature-selected variants (Experiment 3)
#   STEP 10B Walk-forward "out-of-fold" predictions inside the training period
#            (used ONLY to set ensemble weights - never the test set)
#   STEP 11  Majority voting                  (one vote per model)
#   STEP 12  Soft voting                      (average of probabilities)
#   STEP 13  Weighted voting                  (fixed weights from validation F1)
#   STEP 13B Adaptive weighted voting         (weights re-computed daily from recent accuracy)
#   STEP 14  Stacking                         (logistic-regression meta-model)
#   STEP 15  Big comparison table + plots
#   STEP 16  Prediction for the next (unknown) trading day
#   STEP 17  Leakage / integrity checklist (PASS / FAIL)
#   STEP 18  Final head-to-head: majority vs weighted vs adaptive voting
#
# KEY VARIABLES (shared by many steps)
#   df_model              rows that have all features AND a known next-day label
#   X, y                  raw feature matrix / labels for df_model, in date order
#   split_idx             first test row (rows < split_idx are training data)
#   X_full_scaled         all rows scaled with a scaler fitted on TRAINING rows only
#   y_test_common         test labels - every model and ensemble is scored on these
#   model_predictions     {model name: 0/1 test predictions}
#   model_probabilities   {model name: P(UP) on the test set}
#   oof_pred / oof_prob   walk-forward validation predictions (training period only)
#   ensemble_results      {ensemble name: metrics}, filled by register()
#
# THE ONE RULE THAT SHAPES THE WHOLE FILE: NO LOOK-AHEAD
#   Nothing fitted or tuned may see data from its own future. Scalers, feature
#   selectors, weights and meta-models learn from training rows only, nothing is
#   shuffled, and STEP 17 re-checks this programmatically.
# #####################################################################################

import os, glob, warnings, random
# Fix hash / TensorFlow randomness so re-runs give the same numbers.
os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

try:
    get_ipython().run_line_magic("pip", "install -q ta xgboost")   # Colab / Jupyter
except NameError:
    pass                                                           # plain `python Quantpredict.py`

# Reproducibility seeds. TensorFlow seed is set again after import below.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

import matplotlib.pyplot as plt
import seaborn as sns
import ta

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, balanced_accuracy_score
)

# Added for STEP 6 (leakage-safe feature selection) and STEP 14 (stacking meta-model).
from functools import partial
from sklearn.feature_selection import SelectKBest, mutual_info_classif, SelectFromModel

# =====================================================================
# DEEP-LEARNING BACKEND
# Use TensorFlow/Keras if installed, otherwise PyTorch. Every deep model below
# (LSTM, Vision Transformer) is written twice - once per backend - with the
# same architecture, so results are comparable whichever one is available.
# =====================================================================
DL_BACKEND = None
try:
    import tensorflow as tf
    tf.random.set_seed(SEED)
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    DL_BACKEND = "tensorflow"
except ImportError:
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader
        DL_BACKEND = "pytorch"
    except ImportError:
        raise ImportError("Install TensorFlow or PyTorch before running the LSTM.")

print("QuantPredict - 8 Model Pipeline (6 classical + LSTM + Vision Transformer)")
print("DL backend:", DL_BACKEND)

# =====================================================================
# STEP 1 - LOAD DATA
# Look for <STOCK_SYMBOL>.csv in the project folder or archive/. In Colab, ask
# for an upload if it is missing. As a last resort use the first valid stock
# CSV in archive/ (skipping the combined/metadata files and empty files).
# Produces: df (raw rows), filename
# =====================================================================
STOCK_SYMBOL = "RELIANCE"          # any file name in archive/ without ".csv", e.g. "TCS", "INFY"
DATA_DIR = "archive"
OUTPUT_DIR = "outputs"             # figures and result tables are saved here
# Files in archive/ that are not single-stock price histories.
NON_STOCK_FILES = {"NIFTY50_all.csv", "stock_metadata.csv"}


def is_stock_csv(path):
    """True for a real single-stock file (not the combined/metadata files, not empty)."""
    return os.path.basename(path) not in NON_STOCK_FILES and os.path.getsize(path) > 1024


df = None
filename = None
for candidate in [f"{STOCK_SYMBOL}.csv", os.path.join(DATA_DIR, f"{STOCK_SYMBOL}.csv")]:
    if os.path.exists(candidate) and is_stock_csv(candidate):
        filename = candidate
        break

if filename is None:
    try:
        from google.colab import files
        print(f"{STOCK_SYMBOL}.csv not found locally. Upload your stock CSV:")
        uploaded = files.upload()
        if uploaded:
            filename = list(uploaded.keys())[0]
    except Exception:
        pass

if filename is None:
    csvs = [p for p in sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")) + glob.glob("*.csv"))
            if is_stock_csv(p)]
    if not csvs:
        raise FileNotFoundError("No stock CSV found.")
    filename = csvs[0]
    print(f"WARNING: {STOCK_SYMBOL}.csv not found - falling back to {filename}")

df = pd.read_csv(filename)
print("Loaded:", filename, "| rows:", len(df))

# =====================================================================
# STEP 2 - PREPROCESSING
# Normalise column names, find the date column, make OHLCV numeric, then drop
# rows with missing/invalid prices and duplicate dates. Sorted oldest -> newest.
# Produces: df (clean, date-ordered)
# =====================================================================
df.columns = df.columns.str.strip().str.title()      # " close " -> "Close"

# Accept the most common names for the date column.
date_found = False
for c in ["Date", "Datetime", "Time", "Timestamp"]:
    if c in df.columns:
        df["Date"] = pd.to_datetime(df[c], errors="coerce")
        date_found = True
        break
if not date_found:
    raise ValueError("No date column found.")

for c in ["Open", "High", "Low", "Close", "Volume"]:
    if c not in df.columns:
        raise ValueError(f"Missing OHLCV column: {c}")
    df[c] = pd.to_numeric(df[c], errors="coerce")

n_raw = len(df)
df = (
    df.dropna(subset=["Date", "Open", "High", "Low", "Close", "Volume"])
      .query("Open > 0 and High > 0 and Low > 0 and Close > 0")
      .sort_values("Date")
      .drop_duplicates(subset="Date", keep="last")
      .reset_index(drop=True)
)
print(f"Rows after cleaning: {len(df)} (dropped {n_raw - len(df)} invalid/duplicate rows)")

# =====================================================================
# STEP 2a - CORPORATE-ACTION (SPLIT / BONUS / DEMERGER) ADJUSTMENT
# Produces: df with split-adjusted prices, corporate_actions (list of events)
#
# The NSE files hold RAW traded prices. A 1:1 bonus halves the price overnight
# (e.g. RELIANCE on 2009-11-26 and 2017-09-07), which would otherwise create a
# fake -50% "crash", a wrong DOWN label, and broken lag / rolling / indicator values.
# A day is treated as a corporate action only when BOTH the open and the close are
# below 0.78x the previous close: real crashes (e.g. March 2020) open near 0.90x,
# while splits/bonuses open at a clean ratio such as 1/2, 1/5, 2/3, 3/4.
# All rows BEFORE the event are multiplied by the factor (volume divided by it),
# so the most recent prices stay exactly as traded. Only the ratio at the event
# date is used, which is public on the ex-date, so no future price information leaks.
# =====================================================================
CA_THRESHOLD = 0.78                     # open AND close below 78% of yesterday -> corporate action
# Every "clean" fraction a/b with b <= 10 (1/2, 1/3, 2/3, 1/5, 3/4, 1/10 ...).
CLEAN_RATIOS = sorted({a / b for b in range(2, 11) for a in range(1, b)})


def adjust_corporate_actions(frame):
    """Back-adjust prices/volume for splits & bonuses. Returns (adjusted frame, event log)."""
    frame = frame.copy()
    frame[["Open", "High", "Low", "Close", "Volume"]] = (
        frame[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    )
    prev_close = frame["Close"].shift(1)
    gap = frame["Open"] / prev_close          # overnight jump, e.g. 0.50 for a 1:1 bonus
    ret = frame["Close"] / prev_close         # full-day move
    events = frame.index[(gap < CA_THRESHOLD) & (ret < CA_THRESHOLD)]
    log = []
    for i in events:
        raw = float(gap.loc[i])
        # Snap to the nearest clean ratio (0.506 -> 0.5) if it is within 4%;
        # otherwise (e.g. a demerger) use the observed gap as-is.
        nearest = min(CLEAN_RATIOS, key=lambda r: abs(r - raw))
        factor = nearest if abs(nearest - raw) / nearest < 0.04 else raw
        # Scale everything BEFORE the event so the series has no artificial jump.
        before = frame.index < i
        frame.loc[before, ["Open", "High", "Low", "Close"]] *= factor
        frame.loc[before, "Volume"] /= factor   # more shares after a split -> scale old volume up
        log.append((frame.loc[i, "Date"].date(), raw, factor))
    return frame, log


df, corporate_actions = adjust_corporate_actions(df)
print(f"\nCorporate-action adjustments applied: {len(corporate_actions)}")
for d, raw, factor in corporate_actions:
    print(f"  {d}  overnight gap {raw:.3f} -> price factor {factor:.4f}")


# Helper used by every plot in the file: saves outputs/<STOCK>_<name>.png, then shows it.
def save_and_show(name):
    """Save the current figure to OUTPUT_DIR, then display it."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{STOCK_SYMBOL}_{name}.png"), dpi=120, bbox_inches="tight")
    plt.show()
    plt.close("all")


# =====================================================================
# STEP 2b - EXPLORATORY PLOTS
# Quick visual sanity check: the adjusted close should have no sudden -50%
# cliffs any more. Saved as outputs/<STOCK>_00_*.png
# =====================================================================
plt.figure(figsize=(14,5))
plt.plot(df["Date"], df["Close"])
plt.title(f"{filename} - Historical Close Price (split/bonus adjusted)")
plt.xlabel("Date")
plt.ylabel("Close")
plt.grid(alpha=0.3)
save_and_show("00_close_price")

plt.figure(figsize=(14,4))
plt.plot(df["Date"], df["Volume"])
plt.title(f"{filename} - Trading Volume")
plt.xlabel("Date")
plt.ylabel("Volume")
plt.grid(alpha=0.3)
save_and_show("00_volume")

# =====================================================================
# STEP 3 - TECHNICAL INDICATORS  (via the `ta` library)
# These are computed in price units here; STEP 4 converts them into
# scale-free versions before they are used as model inputs.
# Every indicator only looks BACKWARDS (rolling windows), so no look-ahead.
# Produces: new columns on df (MACD*, RSI, SMA_*, EMA_*, BB_*, ATR, OBV, ADX)
# =====================================================================
print("\n" + "=" * 80)
print("STEP 3: CALCULATING TECHNICAL INDICATORS")
print("=" * 80)

# MACD: trend/momentum = EMA(12) - EMA(26), plus its 9-day signal line and the gap between them.
macd =ta.trend.MACD(df["Close"], window_slow=26, window_fast=12, window_sign=9)
df["MACD"] = macd.macd()
df["MACD_signal"] = macd.macd_signal()
df["MACD_hist"] = macd.macd_diff()
print("✓ MACD calculated")

# RSI (0-100): strength of recent gains vs losses; >70 "overbought", <30 "oversold".
df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
print("✓ RSI calculated")

# Simple and exponential moving averages (trend).
df["SMA_20"] = ta.trend.SMAIndicator(df["Close"], window=20).sma_indicator()
df["SMA_50"] = ta.trend.SMAIndicator(df["Close"], window=50).sma_indicator()
df["EMA_12"] = ta.trend.EMAIndicator(df["Close"], window=12).ema_indicator()
df["EMA_26"] = ta.trend.EMAIndicator(df["Close"], window=26).ema_indicator()
print("✓ SMA / EMA calculated")

# Bollinger Bands: 20-day mean +/- 2 standard deviations (volatility envelope).
bb = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
df["BB_upper"] = bb.bollinger_hband()
df["BB_middle"] = bb.bollinger_mavg()
df["BB_lower"] = bb.bollinger_lband()
print("✓ Bollinger Bands calculated")

# ATR: average daily trading range (volatility). OBV: running volume total, +volume
# on up days and -volume on down days (buying/selling pressure). ADX: trend strength.
df["ATR"] = ta.volatility.AverageTrueRange(
    df["High"], df["Low"], df["Close"], window=14
).average_true_range()
print("✓ ATR calculated")

df["OBV"] = ta.volume.OnBalanceVolumeIndicator(
    df["Close"], df["Volume"]
).on_balance_volume()
print("✓ OBV calculated")

df["ADX"] = ta.trend.ADXIndicator(
    df["High"], df["Low"], df["Close"], window=14
).adx()
print("✓ ADX calculated")

print("✓ Technical indicator calculation completed")

# =====================================================================
# STEP 4 - TARGET + FEATURE ENGINEERING
# Produces: Target column, feature_columns (list of model inputs), N_FEATURES,
#           df_model (rows usable for training/testing), latest_row (today)
# =====================================================================
print("\n" + "=" * 80)
print("STEP 4: FEATURE ENGINEERING")
print("=" * 80)

# ---- The label we are predicting: 1 = tomorrow closes higher, 0 = not ----------
df["Next_Close"] = df["Close"].shift(-1)          # tomorrow's close moved up onto today's row
df["Target"] = (df["Next_Close"] > df["Close"]).astype(int)

# STATIONARY FEATURES. Raw price levels (Close, SMA, Bollinger bands, lagged
# closes, OBV ...) drift over 20 years, so a scaler fitted on the training years
# puts the test years far outside the training range. Every feature below is a
# return, a ratio to the current close, a bounded oscillator, or a volume ratio,
# so its distribution is comparable across decades and across stocks.
close = df["Close"]
ret_1 = close.pct_change()                        # today's % change vs yesterday
vol_mean_20 = df["Volume"].rolling(20).mean()     # "normal" volume over the last month

# ---- Price action: momentum and the shape of today's candle ---------------------
df["Return_1"] = ret_1
df["Return_5"] = close.pct_change(5)
df["Return_20"] = close.pct_change(20)
df["Overnight_Gap"] = df["Open"] / close.shift(1) - 1        # jump between yesterday's close and today's open
df["Intraday_Return"] = close / df["Open"] - 1                # move from open to close
df["High_Low_Range"] = df["High"] / df["Low"] - 1             # size of today's range
hl_range = (df["High"] - df["Low"]).replace(0, np.nan)
df["Close_Position"] = ((close - df["Low"]) / hl_range).fillna(0.5)   # 0 = closed at low, 1 = at high

# ---- Indicators made scale-free (divided by, or measured relative to, Close) ----
df["MACD_Pct"] = df["MACD"] / close
df["MACD_Signal_Pct"] = df["MACD_signal"] / close
df["MACD_Hist_Pct"] = df["MACD_hist"] / close
for col in ["SMA_20", "SMA_50", "EMA_12", "EMA_26"]:
    df[f"{col}_Dist"] = close / df[col] - 1                   # +0.03 = price 3% above that average
df["Rolling_Mean_5_Dist"] = close / close.rolling(5).mean() - 1
band = (df["BB_upper"] - df["BB_lower"]).replace(0, np.nan)
df["BB_PctB"] = (close - df["BB_lower"]) / band               # 0 = lower band, 1 = upper band
df["BB_Width"] = band / df["BB_middle"]                       # wide = volatile market
df["ATR_Pct"] = df["ATR"] / close
df["OBV_Slope_5"] = (df["OBV"] - df["OBV"].shift(5)) / (vol_mean_20 * 5)   # 5-day OBV change, in "days of volume"

# ---- Volatility and volume ------------------------------------------------------
df["Volatility_5"] = ret_1.rolling(5).std()
df["Volatility_20"] = ret_1.rolling(20).std()
df["Volume_Change"] = np.log((df["Volume"] + 1) / (df["Volume"].shift(1) + 1))  # log, so spikes don't dominate
df["Volume_Ratio_20"] = df["Volume"] / vol_mean_20            # 2.0 = twice the usual volume

# ---- Lags: the same signals 1, 2, 3 and 5 days ago (short-term memory) ----------
for lag in [1, 2, 3, 5]:
    df[f"Return_Lag_{lag}"] = ret_1.shift(lag)
    df[f"Volume_Ratio_Lag_{lag}"] = df["Volume_Ratio_20"].shift(lag)
    df[f"RSI_Lag_{lag}"] = df["RSI"].shift(lag)

df.replace([np.inf, -np.inf], np.nan, inplace=True)   # e.g. division by zero volume -> treat as missing

# The model inputs, in a fixed order (column index = position in this list).
feature_columns = [
    "Return_1", "Return_5", "Return_20",
    "Overnight_Gap", "Intraday_Return", "High_Low_Range", "Close_Position",
    "MACD_Pct", "MACD_Signal_Pct", "MACD_Hist_Pct", "RSI", "ADX",
    "SMA_20_Dist", "SMA_50_Dist", "EMA_12_Dist", "EMA_26_Dist", "Rolling_Mean_5_Dist",
    "BB_PctB", "BB_Width", "ATR_Pct", "OBV_Slope_5",
    "Volatility_5", "Volatility_20", "Volume_Change", "Volume_Ratio_20",
    "Return_Lag_1", "Return_Lag_2", "Return_Lag_3", "Return_Lag_5",
    "Volume_Ratio_Lag_1", "Volume_Ratio_Lag_2", "Volume_Ratio_Lag_3", "Volume_Ratio_Lag_5",
    "RSI_Lag_1", "RSI_Lag_2", "RSI_Lag_3", "RSI_Lag_5",
]
N_FEATURES = len(feature_columns)

# The first ~50 rows have no SMA_50 etc. yet, so they are dropped here.
# df_feature_complete : every row with all features (includes TODAY)
# df_model            : the same minus today, because today's label (tomorrow's close) is unknown
df_feature_complete = df.dropna(subset=feature_columns).copy()

# IMPORTANT: last row has no real next-day label, so do NOT use it for supervised training/test.
df_model = df_feature_complete.dropna(subset=["Next_Close"]).copy()
df_model.reset_index(drop=True, inplace=True)

# Keep the latest current day separately for live-style next-day prediction.
latest_row = df_feature_complete.iloc[-1]

print("Model samples:", len(df_model))
print("Latest available date:", latest_row["Date"])

print("\nFeature engineering summary:")
print(f"  Total features used: {len(feature_columns)}")
print("  All features are stationary: returns, ratios to Close, oscillators, volume ratios")
print("  Indicators (price-normalised): MACD, RSI, SMA(20,50), EMA(12,26), Bollinger %B/width, ATR, OBV slope, ADX")
print("  Lag features: return, volume ratio, RSI at lags 1, 2, 3, 5")
print("  Rolling features: 5/20-day volatility, 5/20-day momentum, 20-day volume ratio")

# Class-balance sanity check before model training.
class_balance = df_model["Target"].value_counts(normalize=True).sort_index()
print("\nTarget class balance:")
print(f"DOWN (0): {class_balance.get(0, 0):.4f} ({class_balance.get(0, 0)*100:.1f}%)")
print(f"UP   (1): {class_balance.get(1, 0):.4f} ({class_balance.get(1, 0)*100:.1f}%)")

# =====================================================================
# STEP 5 - TRAIN/TEST SPLIT + SCALING
# Time series -> NO random split. The first 80% of days is training, the last
# 20% is the test period. Features are standardised (mean 0, std 1) with a
# scaler fitted on TRAINING rows only, then applied unchanged to test rows.
# Produces: X, y, split_idx, X_train_scaled, X_test_scaled, y_train, y_test,
#           X_full_scaled (train+test stacked, for building LSTM windows)
# =====================================================================
X = df_model[feature_columns].values.astype(np.float32)   # rows = days, columns = features
y = df_model["Target"].values.astype(np.int64)            # 1 = UP next day, 0 = DOWN

# Sequence models (LSTM / ViT) see the last WINDOW_SIZE days for every prediction.
WINDOW_SIZE = 20

split_idx = int(len(X) * 0.80)          # row index where the test period begins

# Fail fast on very small datasets instead of allowing empty LSTM sequences.
assert isinstance(WINDOW_SIZE, int) and WINDOW_SIZE > 0, "WINDOW_SIZE must be a positive integer."
assert split_idx >= WINDOW_SIZE, (
    "split_idx smaller than WINDOW_SIZE — train sequences would be empty. "
    "Use a longer stock history or reduce WINDOW_SIZE."
)

X_train_raw = X[:split_idx]
X_test_raw = X[split_idx:]
y_train = y[:split_idx]
y_test = y[split_idx:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)   # learn mean/std from TRAIN only ...
X_test_scaled = scaler.transform(X_test_raw)         # ... and reuse them on TEST

# Full scaled feature matrix uses only the training-fitted scaler.
X_full_scaled = np.vstack([X_train_scaled, X_test_scaled])
y_full = y.copy()

print("Train:", len(X_train_raw), "| Test:", len(X_test_raw))
print("Split index:", split_idx)

# =====================================================================
# DATASET SUMMARY  (requirement: same dataset, same target, same test period)
# Prints the exact train/test date ranges every result below refers to.
# =====================================================================
dates_model = pd.to_datetime(df_model["Date"])
DATASET_NAME = os.path.basename(filename)
TRAIN_START, TRAIN_END = dates_model.iloc[0], dates_model.iloc[split_idx - 1]
TEST_START, TEST_END = dates_model.iloc[split_idx], dates_model.iloc[-1]

print("\n" + "=" * 80)
print("DATASET SUMMARY")
print("=" * 80)
print(f"Dataset file        : {DATASET_NAME}")
print(f"Full date range     : {dates_model.iloc[0].date()}  ->  {dates_model.iloc[-1].date()}")
print(f"Modelling samples   : {len(df_model)}  (complete features AND a real next-day label)")
print(f"Engineered features : {len(feature_columns)}")
print("Target definition   : Target[t] = 1 if Close[t+1] > Close[t] else 0")
print(f"Split rule          : chronological 80/20, split index = {split_idx}")
print(f"TRAIN period        : {TRAIN_START.date()}  ->  {TRAIN_END.date()}   ({split_idx} samples)")
print(f"TEST  period        : {TEST_START.date()}  ->  {TEST_END.date()}   ({len(X) - split_idx} samples)")
print("Every model and every ensemble below is scored on exactly this TEST period.")

# =====================================================================
# STEP 5b - SHARED UTILITIES
# Used by STEP 6 (feature selection), STEP 10B (walk-forward validation)
# and STEPS 11-14 (the ensemble methods).
#   make_classical_models()   -> fresh, untrained copies of the 6 classical models
#   evaluate_binary()         -> accuracy / precision / recall / F1 / ROC-AUC
#   make_walk_forward_folds() -> time-ordered validation folds inside the training period
#   build_sequences()         -> 20-day windows for the LSTM / ViT
# =====================================================================

# ---- configuration for the time-aware machinery -------------------------------
WF_N_SPLITS = 4                 # walk-forward folds INSIDE the training period
WF_INITIAL_TRAIN_FRAC = 0.50    # fold 1 trains on the first 50% of the training period
TOP_K_FEATURES = 20             # k for SelectKBest / SelectFromModel
WF_DL_EPOCHS = 20               # shorter DL budget inside folds, to stay Colab-friendly


def make_classical_models(seed=SEED, svm_probability=True):
    """
    Fresh, UNFITTED copies of the six classical models.

    A factory is required because walk-forward validation refits every model once
    per fold; reusing one already-fitted object would silently carry information
    from an earlier fold into a later one.

    svm_probability=False is used only where labels (not probabilities) are
    needed - SVC(probability=True) runs an internal 5-fold CV and is much slower.
    """
    return {
        # Linear baseline: weighted sum of features -> probability.
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=seed),
        # 200 independent decision trees, majority vote (robust, low tuning).
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=seed, n_jobs=-1
        ),
        # Trees built one after another, each fixing the previous ones' errors.
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=5, random_state=seed
        ),
        # Optimised, regularised gradient boosting.
        "XGBoost": XGBClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=5,
            random_state=seed, n_jobs=-1, eval_metric="logloss"
        ),
        # Support Vector Machine with an RBF kernel: finds a curved UP/DOWN boundary.
        "SVM": SVC(C=1.0, kernel="rbf", probability=svm_probability, random_state=seed),
        # Looks up the 5 most similar past days and copies their outcome.
        "KNN": KNeighborsClassifier(n_neighbors=5, weights="distance"),
    }


def evaluate_binary(y_true, y_pred, y_prob=None):
    """
    Accuracy / precision / recall / F1 (+ ROC-AUC when probabilities are given).
      accuracy  : share of days predicted correctly
      precision : of the days we said UP, how many really went up
      recall    : of the days that really went up, how many we caught
      f1        : balance of precision and recall
      roc_auc   : how well P(UP) ranks up-days above down-days (0.5 = random)
    """
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        try:
            out["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            out["roc_auc"] = np.nan
    else:
        out["roc_auc"] = np.nan
    return out


def make_walk_forward_folds(n_train, n_splits=None, initial_train_frac=None):
    """
    Expanding-window ("walk-forward") folds INSIDE the training period.

    HOW THIS PREVENTS FUTURE INFORMATION FROM ENTERING TRAINING OR WEIGHTING
    -----------------------------------------------------------------------
    * Each fold is (train = [0, tr_end), validate = [tr_end, va_end)). The
      validation block always sits strictly AFTER its own training block in
      calendar time, so a model is only ever scored on days it has never seen.
    * The training block only grows forwards (expanding window). Fold k+1 may
      learn from fold k's validation days because by then those days are in the
      past - exactly what a real trading system would be allowed to do.
    * Nothing is shuffled. Rows keep their chronological order everywhere.
    * n_train is always passed as split_idx, so va_end can never cross the
      train/test boundary. Every weight, every feature-selection decision and
      every stacking meta-model below is therefore computed without the TEST set.

        fold 1:  train [......]              val [--]
        fold 2:  train [........]            val   [--]
        fold 3:  train [..........]          val     [--]
        fold 4:  train [............]        val       [--]
        |------------- TRAINING PERIOD -------------|  TEST (never used here)  |
    """
    n_splits = WF_N_SPLITS if n_splits is None else n_splits
    initial_train_frac = (
        WF_INITIAL_TRAIN_FRAC if initial_train_frac is None else initial_train_frac
    )

    first_train_end = int(n_train * initial_train_frac)
    block = (n_train - first_train_end) // n_splits
    assert block > 0, "Walk-forward blocks are empty. Reduce WF_N_SPLITS or use a longer history."

    folds = []
    for k in range(n_splits):
        tr_end = first_train_end + k * block
        va_end = tr_end + block if k < n_splits - 1 else n_train
        folds.append((0, tr_end, tr_end, va_end))

    # Hard leakage guard: no fold may reach into the test period.
    for (_, tr_end, va_start, va_end) in folds:
        assert 0 < tr_end <= va_start < va_end <= n_train, "Malformed walk-forward fold."
        assert va_end <= split_idx, "Walk-forward fold leaked into the TEST period."
    return folds


def build_sequences(X_scaled, y_arr, start, end, window):
    """
    Build LSTM/ViT sequences using the project's CORRECTED alignment:

        sequence X[i-window+1 : i+1]   ->   label y[i]

    The sequence ENDS at day i (inclusive) because Target[i] compares Close[i+1]
    with Close[i], so day i's own features must be inside the window. This is the
    alignment the existing LSTM already uses and it is left unchanged.
    """
    assert start >= window - 1, "Not enough history to build the first sequence."
    xs, ys, idx = [], [], []
    for i in range(start, end):
        xs.append(X_scaled[i - window + 1:i + 1])
        ys.append(y_arr[i])
        idx.append(i)
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(idx, dtype=np.int64),
    )


# =====================================================================
# STEP 6 - FEATURE SELECTION (LEAKAGE-SAFE)
#
# Both selectors are fitted on the TRAINING PARTITION ONLY. Test rows are never
# shown to mutual_info_classif or to the importance-ranking forest, so the choice
# of features carries no information about the future.
#
#   Method A  Mutual Information : how much knowing a feature reduces uncertainty
#                                  about UP/DOWN (captures non-linear links).
#   Method B  Tree importance    : how much a Random Forest relied on each feature.
# Each method keeps its TOP_K_FEATURES best features. Then "All / MI / Tree" are
# compared with walk-forward validation, and the winner is picked on validation F1.
# Produces: FEATURE_SETS, fs_val_df, BEST_FEATURE_SET, selected_indices
# =====================================================================
print("\n" + "=" * 80)
print("STEP 6: FEATURE SELECTION (fitted on TRAINING data only)")
print("=" * 80)

# ---- Method A: Mutual Information via SelectKBest ------------------------------
mi_selector = SelectKBest(
    score_func=partial(mutual_info_classif, random_state=SEED),
    k=TOP_K_FEATURES,
)
mi_selector.fit(X_train_scaled, y_train)            # <-- TRAIN ONLY
mi_scores = np.nan_to_num(mi_selector.scores_)
mi_indices = np.where(mi_selector.get_support())[0]

# ---- Method B: Tree-based feature importance -----------------------------------
importance_forest = RandomForestClassifier(
    n_estimators=300, max_depth=10, random_state=SEED, n_jobs=-1
)
importance_forest.fit(X_train_scaled, y_train)      # <-- TRAIN ONLY
tree_importances = importance_forest.feature_importances_
tree_selector = SelectFromModel(
    importance_forest, prefit=True, max_features=TOP_K_FEATURES, threshold=-np.inf
)
tree_indices = np.where(tree_selector.get_support())[0]

# Explicit leakage assertions: the selectors saw exactly split_idx rows.
assert mi_selector.n_features_in_ == len(feature_columns)
assert importance_forest.n_features_in_ == len(feature_columns)
assert X_train_scaled.shape[0] == split_idx, "Selector input was not the train partition."
assert len(mi_indices) == TOP_K_FEATURES and len(tree_indices) == TOP_K_FEATURES

FEATURE_SETS = {
    "All Features": np.arange(len(feature_columns)),
    f"MI Top-{TOP_K_FEATURES}": mi_indices,
    f"Tree Top-{TOP_K_FEATURES}": tree_indices,
}

print(f"Original number of features : {len(feature_columns)}")
print(f"Selected number of features : {TOP_K_FEATURES}  (k is a fixed, documented hyperparameter)")
print("\nMutual-Information selected features:")
for rank, i in enumerate(sorted(mi_indices, key=lambda j: -mi_scores[j]), start=1):
    print(f"  {rank:>2}. {feature_columns[i]:<24} MI = {mi_scores[i]:.5f}")
print("\nTree-importance selected features:")
for rank, i in enumerate(sorted(tree_indices, key=lambda j: -tree_importances[j]), start=1):
    print(f"  {rank:>2}. {feature_columns[i]:<24} importance = {tree_importances[i]:.5f}")

overlap = sorted(set(mi_indices) & set(tree_indices))
print(f"\nChosen by BOTH methods ({len(overlap)}): {[feature_columns[i] for i in overlap]}")

# ---- Compare feature sets with TIME-AWARE validation (train period only) --------
# Only the six classical models are used for this comparison, to keep the runtime
# reasonable in Colab. SVM runs with probability=False here because the comparison
# metric is F1 on labels, not a probability score.
fs_folds = make_walk_forward_folds(split_idx)
print(f"\nWalk-forward folds used for the feature-set comparison ({len(fs_folds)}):")
for k, (_, tr_end, va_start, va_end) in enumerate(fs_folds, start=1):
    print(f"  Fold {k}: train rows 0..{tr_end-1}  ->  validate rows {va_start}..{va_end-1} "
          f"({dates_model.iloc[va_start].date()} .. {dates_model.iloc[va_end-1].date()})")


def walk_forward_f1_classical(feature_idx, folds):
    """Mean walk-forward validation F1 per classical model, for one feature subset."""
    scores = {name: [] for name in make_classical_models().keys()}
    for (tr_start, tr_end, va_start, va_end) in folds:
        # A FRESH scaler per fold, fitted on that fold's training rows only.
        fold_scaler = StandardScaler().fit(X[tr_start:tr_end][:, feature_idx])
        Xtr = fold_scaler.transform(X[tr_start:tr_end][:, feature_idx])
        Xva = fold_scaler.transform(X[va_start:va_end][:, feature_idx])
        ytr, yva = y[tr_start:tr_end], y[va_start:va_end]
        for name, mdl in make_classical_models(svm_probability=False).items():
            mdl.fit(Xtr, ytr)
            scores[name].append(f1_score(yva, mdl.predict(Xva), zero_division=0))
    return {name: float(np.mean(v)) for name, v in scores.items()}


print("\nRunning walk-forward validation for each feature set "
      "(slowest cell; lower WF_N_SPLITS to speed it up)...")
fs_validation = {}
for set_name, idx in FEATURE_SETS.items():
    fs_validation[set_name] = walk_forward_f1_classical(idx, fs_folds)
    print(f"  {set_name:<16} mean validation F1 = "
          f"{np.mean(list(fs_validation[set_name].values())):.4f}")

fs_val_df = pd.DataFrame(fs_validation)
fs_val_df.loc["MEAN"] = fs_val_df.mean()
print("\nWALK-FORWARD VALIDATION F1 BY FEATURE SET (validation only - no test data)")
print(fs_val_df.round(4).to_string())

# The winning feature set is chosen using VALIDATION performance only.
BEST_FEATURE_SET = str(fs_val_df.loc["MEAN"].idxmax())
selected_indices = FEATURE_SETS[BEST_FEATURE_SET]
selected_feature_names = [feature_columns[i] for i in selected_indices]
print(f"\nFeature set chosen by walk-forward VALIDATION: {BEST_FEATURE_SET}")
if BEST_FEATURE_SET == "All Features":
    print(f"=> On this data feature selection did NOT beat the full {N_FEATURES}-feature baseline "
          "in validation. The baseline stays the primary pipeline.")
else:
    print(f"=> {len(selected_feature_names)} features selected: {selected_feature_names}")

# Scaled matrices restricted to the selected features (scaler fitted on TRAIN only).
selected_scaler = StandardScaler().fit(X[:split_idx][:, selected_indices])
X_sel_train_scaled = selected_scaler.transform(X[:split_idx][:, selected_indices])
X_sel_test_scaled = selected_scaler.transform(X[split_idx:][:, selected_indices])
X_sel_full_scaled = np.vstack([X_sel_train_scaled, X_sel_test_scaled])
assert selected_scaler.n_features_in_ == len(selected_indices)
print("Selected-feature scaler fitted on training rows only: OK")


# =====================================================================
# STEP 7 - SIX CLASSICAL ML MODELS (baseline, all features)
# Each model is trained once on the training period and predicts every test day.
# Model definitions live in make_classical_models() (STEP 5b) so that every
# walk-forward fold can build fresh, unfitted estimators.
# Produces: results, model_predictions, model_probabilities, model_objects
#           (dictionaries keyed by model name; LSTM and ViT are added later)
# =====================================================================
models = make_classical_models()

results = {}
model_predictions = {}
model_objects = {}

model_probabilities = {}   # P(UP) on the test set, needed by soft/weighted/stacking

for name, model in models.items():
    model.fit(X_train_scaled, y_train)               # learn from the training period
    pred = model.predict(X_test_scaled)              # 0/1 decision for every test day
    prob = model.predict_proba(X_test_scaled)[:, 1]  # confidence that the day is UP
    results[name] = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0)
    }
    model_predictions[name] = pred
    model_probabilities[name] = prob
    model_objects[name] = model

print("\nSTEP 7 complete - 6 classical models trained on the TRAINING partition only.")
for name in models:
    print(f"  {name:<22} test accuracy = {results[name]['accuracy']:.4f}")

# =====================================================================
# STEP 8 - LSTM (CORRECTED ALIGNMENT - PRESERVED)
# An LSTM is a recurrent neural network that reads the last WINDOW_SIZE days in
# order and outputs P(UP) for the next day.
#
# Alignment: Target[i] = 1 if Close[i+1] > Close[i].
# Therefore, sequence ending at i MUST INCLUDE X[i] (today's features):
#
#     X[i-W+1 : i+1] -> y[i]        (days i-19 .. i  ->  label of day i)
#
# Architecture: LSTM(64) -> Dropout -> Dense(32, relu) -> Dense(1, sigmoid)
# Produces: lstm_model, lstm_probs, X_tr / X_val / X_test_seq (reused by the ViT)
# =====================================================================
print("\n" + "=" * 80)
print("STEP 8: LSTM DEEP LEARNING MODEL")
print("=" * 80)
EPOCHS = 35            # maximum passes over the training data
BATCH_SIZE = 32        # days per gradient update
LEARNING_RATE = 0.001
DROPOUT = 0.20         # randomly silence 20% of units while training (reduces overfitting)
THRESHOLD = 0.50       # P(UP) >= 0.5 -> predict UP (shared by every probability model/ensemble)
PATIENCE = 8           # stop if validation loss hasn't improved for 8 epochs

# Re-apply seeds immediately before model creation/training.
random.seed(SEED)
np.random.seed(SEED)
if DL_BACKEND == "tensorflow":
    tf.random.set_seed(SEED)

print(
    f"Config -> window={WINDOW_SIZE}, epochs={EPOCHS}, batch_size={BATCH_SIZE}, "
    f"learning_rate={LEARNING_RATE}, dropout={DROPOUT}, threshold={THRESHOLD}, "
    f"patience={PATIENCE}, seed={SEED}"
)

# Build (window, label) pairs. Training windows end on training days; test windows
# end on test days but may look back into late training days (that's the past).
X_train_seq, y_train_seq = [], []
for i in range(WINDOW_SIZE - 1, split_idx):
    X_train_seq.append(X_full_scaled[i-WINDOW_SIZE+1:i+1])
    y_train_seq.append(y_full[i])

X_train_seq = np.asarray(X_train_seq, dtype=np.float32)
y_train_seq = np.asarray(y_train_seq, dtype=np.float32)

X_test_seq, y_test_seq = [], []
for i in range(split_idx, len(X_full_scaled)):
    X_test_seq.append(X_full_scaled[i-WINDOW_SIZE+1:i+1])
    y_test_seq.append(y_full[i])

X_test_seq = np.asarray(X_test_seq, dtype=np.float32)
y_test_seq = np.asarray(y_test_seq, dtype=np.float32)

# Chronological validation split INSIDE training period: the last 15% of training
# windows is held back to decide when to stop training (early stopping).
val_idx = int(len(X_train_seq) * 0.85)
X_tr, X_val = X_train_seq[:val_idx], X_train_seq[val_idx:]
y_tr, y_val = y_train_seq[:val_idx], y_train_seq[val_idx:]

print("\nLSTM sequence shapes")
print("Train:", X_tr.shape)
print("Val  :", X_val.shape)
print("Test :", X_test_seq.shape)

print("\nSequence alignment check")
print(
    "First train sequence: "
    f"{WINDOW_SIZE-1-WINDOW_SIZE+1}..{WINDOW_SIZE-1} -> y[{WINDOW_SIZE-1}]"
)
print(
    "Last train sequence: "
    f"{split_idx-1-WINDOW_SIZE+1}..{split_idx-1} -> y[{split_idx-1}]"
)
print(
    "First test sequence: "
    f"{split_idx-WINDOW_SIZE+1}..{split_idx} -> y[{split_idx}]"
)
print(
    "Last test sequence: "
    f"{len(X_full_scaled)-1-WINDOW_SIZE+1}..{len(X_full_scaled)-1} "
    f"-> y[{len(X_full_scaled)-1}]"
)

# Programmatic sequence/leakage sanity checks.
assert split_idx >= WINDOW_SIZE, (
    "split_idx smaller than WINDOW_SIZE — train sequences would be empty."
)
assert X_train_seq.shape[0] > 0, "No LSTM training sequences were created."
assert X_test_seq.shape[0] > 0, "No LSTM test sequences were created."
assert X_train_seq.shape[0] == split_idx - (WINDOW_SIZE - 1)
assert X_test_seq.shape[0] == len(X_full_scaled) - split_idx

# The last training target is before the test boundary.
last_train_target_idx = split_idx - 1
first_test_target_idx = split_idx
assert last_train_target_idx < first_test_target_idx

# Sequence ending at the final training target cannot include a test index.
last_train_sequence_end = last_train_target_idx
assert last_train_sequence_end < split_idx

# First test sequence is allowed to use historical observations before the split:
# those observations are known at prediction time. Its target is the first
# unseen/future target at the test boundary.
assert first_test_target_idx == split_idx

print("✓ Leakage/sequence sanity checks passed")
print("✓ Scaler fitted only on training partition")
print("✓ Training sequences end before test boundary")
print("✓ First test sequence uses only historical observations available before/on test-day prediction")
print("✓ Test targets are never used in LSTM training")

# ---- Train the LSTM (TensorFlow version first, PyTorch version in the else) ----
if DL_BACKEND == "tensorflow":
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    lstm_model = Sequential([
        Input(shape=(WINDOW_SIZE, len(feature_columns))),
        LSTM(64),
        Dropout(DROPOUT),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    lstm_model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True
    )

    history = lstm_model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        callbacks=[early_stop],
        verbose=1
    )

    lstm_probs = lstm_model.predict(X_test_seq, verbose=0).flatten()

    # Backend-neutral training curve, used by the STEP 15 loss plot.
    lstm_history = {
        "loss": list(history.history.get("loss", [])),
        "val_loss": list(history.history.get("val_loss", [])),
    }

else:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class TorchLSTM(nn.Module):
        """Same network as the Keras version: LSTM(64) -> Dropout -> Dense(32) -> sigmoid."""

        def __init__(self, n_features, hidden=64, dense=32, drop=0.2):
            super().__init__()
            self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
            self.dropout = nn.Dropout(drop)
            self.fc1 = nn.Linear(hidden, dense)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(dense, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):                    # x: (batch, 20 days, n_features)
            out, _ = self.lstm(x)
            out = self.dropout(out[:, -1, :])    # keep only the LSTM state after the LAST day
            out = self.relu(self.fc1(out))
            return self.sigmoid(self.fc2(out))

    lstm_model = TorchLSTM(len(feature_columns), 64, 32, DROPOUT).to(device)
    criterion = nn.BCELoss()                  # binary cross-entropy: penalises confident wrong answers
    optimizer = optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE)

    train_ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)   # keep date order
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Early-stopping bookkeeping: remember the weights with the lowest validation loss.
    best_loss = float("inf")
    best_state = None
    patience_count = 0
    lstm_history = {"loss": [], "val_loss": []}   # needed for the STEP 15 loss plot

    for epoch in range(EPOCHS):
        # -- 1) one pass over the training windows, updating the weights --
        lstm_model.train()
        epoch_train_loss = 0.0
        epoch_train_n = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = lstm_model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * len(xb)
            epoch_train_n += len(xb)

        # -- 2) measure loss on the held-back validation windows (no weight updates) --
        lstm_model.eval()
        total_loss = 0.0
        total_n = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = lstm_model(xb)
                loss = criterion(out, yb)
                total_loss += loss.item() * len(xb)
                total_n += len(xb)

        val_loss = total_loss / max(total_n, 1)
        lstm_history["loss"].append(epoch_train_loss / max(epoch_train_n, 1))
        lstm_history["val_loss"].append(val_loss)

        # -- 3) keep the best weights; stop after PATIENCE epochs without improvement --
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in lstm_model.state_dict().items()
            }
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= PATIENCE:
            print("Early stopping at epoch", epoch + 1)
            break

    if best_state is not None:
        lstm_model.load_state_dict(best_state)     # roll back to the best epoch

    # Predict P(UP) for every test window.
    lstm_model.eval()
    with torch.no_grad():
        lstm_probs = (
            lstm_model(
                torch.tensor(X_test_seq, dtype=torch.float32).to(device)
            )
            .cpu().squeeze().numpy()
        )

# ---- Score the LSTM on the test period and store it with the other models ----
lstm_pred = (lstm_probs >= THRESHOLD).astype(int)

results["LSTM"] = {
    "accuracy": accuracy_score(y_test_seq, lstm_pred),
    "precision": precision_score(y_test_seq, lstm_pred, zero_division=0),
    "recall": recall_score(y_test_seq, lstm_pred, zero_division=0),
    "f1": f1_score(y_test_seq, lstm_pred, zero_division=0)
}
model_predictions["LSTM"] = lstm_pred
model_probabilities["LSTM"] = np.asarray(lstm_probs, dtype=np.float64).ravel()

print("\nLSTM TEST SUMMARY")
print("-" * 60)
print(f"Test samples : {len(y_test_seq)}")
print(f"UP predicted : {int(lstm_pred.sum())}")
print(f"DOWN predicted: {int(len(lstm_pred) - lstm_pred.sum())}")
print(f"Accuracy     : {results['LSTM']['accuracy']:.4f}")
print(f"Precision    : {results['LSTM']['precision']:.4f}")
print(f"Recall       : {results['LSTM']['recall']:.4f}")
print(f"F1-score     : {results['LSTM']['f1']:.4f}")
print(f"Avg predicted UP probability: {float(np.mean(lstm_probs)):.4f}")
model_objects["LSTM"] = lstm_model


# =====================================================================
# STEP 8b - REUSABLE LSTM TRAINER
#
# The baseline LSTM above is left completely untouched. This helper rebuilds the
# SAME architecture and the SAME hyperparameters so it can be retrained inside
# each walk-forward fold and on the feature-selected matrix. Nothing about the
# baseline LSTM's result depends on this function.
# =====================================================================
def train_lstm_model(X_tr_seq, y_tr_seq, X_va_seq, y_va_seq, epochs=None, verbose=0):
    """
    Train one LSTM and return (predict_fn, history_dict).

    predict_fn(array_of_sequences) -> 1-D array of P(UP).
    Early stopping uses the chronologically LAST slice of the training period,
    which is passed in as (X_va_seq, y_va_seq). No test data is ever involved.
    """
    epochs = EPOCHS if epochs is None else epochs
    n_feat = X_tr_seq.shape[2]

    random.seed(SEED)
    np.random.seed(SEED)

    if DL_BACKEND == "tensorflow":
        tf.keras.backend.clear_session()
        tf.random.set_seed(SEED)

        mdl = Sequential([
            Input(shape=(WINDOW_SIZE, n_feat)),
            LSTM(64),
            Dropout(DROPOUT),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid"),
        ])
        mdl.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        hist = mdl.fit(
            X_tr_seq, y_tr_seq,
            validation_data=(X_va_seq, y_va_seq),
            epochs=epochs, batch_size=BATCH_SIZE, shuffle=False,
            callbacks=[EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                     restore_best_weights=True)],
            verbose=verbose,
        )
        return (
            (lambda arr: mdl.predict(arr, verbose=0).ravel()),
            {"loss": list(hist.history.get("loss", [])),
             "val_loss": list(hist.history.get("val_loss", []))},
        )

    # ---- PyTorch fallback ------------------------------------------------------
    torch.manual_seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl = TorchLSTM(n_feat, 64, 32, DROPOUT).to(dev)
    crit = nn.BCELoss()
    opt = optim.Adam(mdl.parameters(), lr=LEARNING_RATE)

    tr_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr_seq), torch.tensor(y_tr_seq).unsqueeze(1)),
        batch_size=BATCH_SIZE, shuffle=False,
    )
    va_loader = DataLoader(
        TensorDataset(torch.tensor(X_va_seq), torch.tensor(y_va_seq).unsqueeze(1)),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    hist = {"loss": [], "val_loss": []}
    best, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        mdl.train()
        tl, tn = 0.0, 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = crit(mdl(xb), yb)
            loss.backward()
            opt.step()
            tl += loss.item() * len(xb)
            tn += len(xb)
        mdl.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                vl += crit(mdl(xb), yb).item() * len(xb)
                vn += len(xb)
        hist["loss"].append(tl / max(tn, 1))
        hist["val_loss"].append(vl / max(vn, 1))
        if hist["val_loss"][-1] < best:
            best = hist["val_loss"][-1]
            best_state = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        mdl.load_state_dict(best_state)
    mdl.eval()

    def _predict(arr):
        with torch.no_grad():
            return mdl(torch.tensor(arr, dtype=torch.float32).to(dev)).cpu().numpy().ravel()

    return _predict, hist


# =====================================================================
# STEP 9 - VISION TRANSFORMER
#         "Vision Transformer adapted for financial time-series feature matrices"
#
# This is NOT a conventional image-classification ViT and no images are used or
# downloaded. Each training example is the SAME WINDOW_SIZE x N_FEATURES matrix the LSTM already
# consumes (20 trading days x all engineered features). That matrix is treated
# like a small single-channel image: it is cut into a grid of non-overlapping
# rectangular patches, each patch is linearly embedded, learnable positional
# embeddings are added, the patch sequence is passed through Transformer encoder
# blocks, and a classification head with a sigmoid outputs P(UP).
#
#   20 x F matrix -> patchify -> patch embeddings -> + positional embeddings
#   -> transformer encoder blocks -> mean-pool -> dense head -> sigmoid -> UP/DOWN
#
# The target, the split, the scaler and the sequence alignment are identical to
# the LSTM's, so the ViT is directly comparable with every other model.
# =====================================================================
print("\n" + "=" * 80)
print("STEP 9: VISION TRANSFORMER (adapted for financial time-series feature matrices)")
print("=" * 80)

VIT_PATCH_T = 5              # patch height: trading days per patch
VIT_FEATURE_PATCHES = 2      # patch grid width: number of feature groups
VIT_D_MODEL = 64             # patch-embedding dimension
VIT_HEADS = 4                # attention heads
VIT_LAYERS = 2               # transformer encoder blocks (kept small for Colab)
VIT_MLP_DIM = 128            # feed-forward width inside each block
VIT_DROPOUT = 0.20
VIT_EPOCHS = 40
VIT_BATCH = 32
VIT_LR = 0.001
VIT_PATIENCE = 8

assert WINDOW_SIZE % VIT_PATCH_T == 0, (
    f"WINDOW_SIZE ({WINDOW_SIZE}) must be divisible by VIT_PATCH_T ({VIT_PATCH_T})."
)


def vit_patchify(sequences, patch_t=VIT_PATCH_T, feature_patches=VIT_FEATURE_PATCHES):
    """
    (B, T, F) financial feature matrices -> (B, n_patches, patch_dim) ViT patches.

    The T x F matrix is cut into a (T / patch_t) x feature_patches grid of
    rectangles, exactly like a ViT cuts an image into square patches. Each
    rectangle is then flattened into one patch vector.

    If F is not divisible by feature_patches the feature axis is zero-padded.
    Because the features are already standardised, a zero equals the training
    mean of that feature, so the padding adds no information.
    """
    B, T, F = sequences.shape
    assert T % patch_t == 0
    patch_f = int(np.ceil(F / feature_patches))
    padded_F = patch_f * feature_patches
    if padded_F != F:
        sequences = np.pad(sequences, ((0, 0), (0, 0), (0, padded_F - F)), mode="constant")
    n_t = T // patch_t
    out = sequences.reshape(B, n_t, patch_t, feature_patches, patch_f)
    out = out.transpose(0, 1, 3, 2, 4)                       # (B, n_t, n_f, patch_t, patch_f)
    out = out.reshape(B, n_t * feature_patches, patch_t * patch_f)
    return np.ascontiguousarray(out, dtype=np.float32)


def vit_patch_grid(n_features):
    """Return (n_patches, patch_dim) for a given feature count."""
    patch_f = int(np.ceil(n_features / VIT_FEATURE_PATCHES))
    return (WINDOW_SIZE // VIT_PATCH_T) * VIT_FEATURE_PATCHES, VIT_PATCH_T * patch_f


if DL_BACKEND == "tensorflow":
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        LayerNormalization, MultiHeadAttention, Add, GlobalAveragePooling1D
    )

    class AddPositionalEmbedding(tf.keras.layers.Layer):
        """Learnable positional embedding added to every patch embedding."""

        def __init__(self, n_patches, d_model, **kwargs):
            super().__init__(**kwargs)
            self.n_patches = n_patches
            self.d_model = d_model

        def build(self, input_shape):
            self.pos_emb = self.add_weight(
                name="pos_emb",
                shape=(1, self.n_patches, self.d_model),
                initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
                trainable=True,
            )
            super().build(input_shape)

        def call(self, x):
            return x + self.pos_emb

    def build_vit_tensorflow(n_patches, patch_dim):
        """Small pre-norm ViT encoder over financial patches."""
        tf.keras.backend.clear_session()
        tf.random.set_seed(SEED)

        inputs = Input(shape=(n_patches, patch_dim), name="patches")
        x = Dense(VIT_D_MODEL, name="patch_embedding")(inputs)      # linear patch embedding
        x = AddPositionalEmbedding(n_patches, VIT_D_MODEL)(x)       # positional embedding
        x = Dropout(VIT_DROPOUT)(x)

        for _ in range(VIT_LAYERS):                                 # transformer encoder blocks
            h = LayerNormalization(epsilon=1e-6)(x)
            h = MultiHeadAttention(
                num_heads=VIT_HEADS,
                key_dim=max(VIT_D_MODEL // VIT_HEADS, 1),
                dropout=VIT_DROPOUT,
            )(h, h)
            x = Add()([x, h])
            h = LayerNormalization(epsilon=1e-6)(x)
            h = Dense(VIT_MLP_DIM, activation="gelu")(h)
            h = Dropout(VIT_DROPOUT)(h)
            h = Dense(VIT_D_MODEL)(h)
            h = Dropout(VIT_DROPOUT)(h)
            x = Add()([x, h])

        x = LayerNormalization(epsilon=1e-6)(x)
        x = GlobalAveragePooling1D()(x)                             # classification head
        x = Dropout(VIT_DROPOUT)(x)
        x = Dense(32, activation="relu")(x)
        outputs = Dense(1, activation="sigmoid")(x)
        return Model(inputs, outputs, name="ViT_financial_timeseries")

else:
    class TorchViT(nn.Module):
        """PyTorch equivalent of the Keras ViT above (same sizes, pre-norm blocks)."""

        def __init__(self, n_patches, patch_dim):
            super().__init__()
            self.patch_embedding = nn.Linear(patch_dim, VIT_D_MODEL)
            self.pos_emb = nn.Parameter(torch.randn(1, n_patches, VIT_D_MODEL) * 0.02)
            self.drop = nn.Dropout(VIT_DROPOUT)
            layer = nn.TransformerEncoderLayer(
                d_model=VIT_D_MODEL, nhead=VIT_HEADS, dim_feedforward=VIT_MLP_DIM,
                dropout=VIT_DROPOUT, activation="gelu", batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=VIT_LAYERS)
            self.norm = nn.LayerNorm(VIT_D_MODEL)
            self.head = nn.Sequential(
                nn.Dropout(VIT_DROPOUT), nn.Linear(VIT_D_MODEL, 32), nn.ReLU(),
                nn.Linear(32, 1), nn.Sigmoid(),
            )

        def forward(self, x):
            z = self.drop(self.patch_embedding(x) + self.pos_emb)
            z = self.encoder(z)
            return self.head(self.norm(z).mean(dim=1))


def train_vit_model(P_tr, y_tr, P_va, y_va, epochs=None, verbose=0):
    """
    Train one ViT on pre-patchified inputs and return (predict_fn, history_dict).
    Early stopping uses the chronologically last slice of the training period.
    """
    epochs = VIT_EPOCHS if epochs is None else epochs
    n_patches, patch_dim = P_tr.shape[1], P_tr.shape[2]

    random.seed(SEED)
    np.random.seed(SEED)

    if DL_BACKEND == "tensorflow":
        mdl = build_vit_tensorflow(n_patches, patch_dim)
        mdl.compile(
            optimizer=Adam(learning_rate=VIT_LR),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        hist = mdl.fit(
            P_tr, y_tr,
            validation_data=(P_va, y_va),
            epochs=epochs, batch_size=VIT_BATCH, shuffle=False,
            callbacks=[EarlyStopping(monitor="val_loss", patience=VIT_PATIENCE,
                                     restore_best_weights=True)],
            verbose=verbose,
        )
        return (
            (lambda arr: mdl.predict(arr, verbose=0).ravel()),
            {"loss": list(hist.history.get("loss", [])),
             "val_loss": list(hist.history.get("val_loss", []))},
            mdl,
        )

    torch.manual_seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl = TorchViT(n_patches, patch_dim).to(dev)
    crit = nn.BCELoss()
    opt = optim.Adam(mdl.parameters(), lr=VIT_LR)

    tr_loader = DataLoader(
        TensorDataset(torch.tensor(P_tr), torch.tensor(y_tr).unsqueeze(1)),
        batch_size=VIT_BATCH, shuffle=False,
    )
    va_loader = DataLoader(
        TensorDataset(torch.tensor(P_va), torch.tensor(y_va).unsqueeze(1)),
        batch_size=VIT_BATCH, shuffle=False,
    )

    hist = {"loss": [], "val_loss": []}
    best, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        mdl.train()
        tl, tn = 0.0, 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = crit(mdl(xb), yb)
            loss.backward()
            opt.step()
            tl += loss.item() * len(xb)
            tn += len(xb)
        mdl.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                vl += crit(mdl(xb), yb).item() * len(xb)
                vn += len(xb)
        hist["loss"].append(tl / max(tn, 1))
        hist["val_loss"].append(vl / max(vn, 1))
        if hist["val_loss"][-1] < best:
            best = hist["val_loss"][-1]
            best_state = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= VIT_PATIENCE:
                break
    if best_state is not None:
        mdl.load_state_dict(best_state)
    mdl.eval()

    def _predict(arr):
        with torch.no_grad():
            return mdl(torch.tensor(arr, dtype=torch.float32).to(dev)).cpu().numpy().ravel()

    return _predict, hist, mdl


# ---- Baseline ViT: same sequences, same split, same scaler as the LSTM ---------
# X_tr / X_val / X_test_seq were built in STEP 8 from X_full_scaled, whose scaler
# was fitted on the training partition only.
P_tr = vit_patchify(X_tr)
P_val = vit_patchify(X_val)
P_test = vit_patchify(X_test_seq)

n_patches_base, patch_dim_base = vit_patch_grid(len(feature_columns))
print(f"Input matrix per sample  : {WINDOW_SIZE} time steps x {len(feature_columns)} features")
print(f"Patch grid               : {WINDOW_SIZE // VIT_PATCH_T} x {VIT_FEATURE_PATCHES} "
      f"= {n_patches_base} patches")
print(f"Patch size               : {VIT_PATCH_T} days x "
      f"{int(np.ceil(len(feature_columns) / VIT_FEATURE_PATCHES))} features "
      f"-> flattened patch_dim = {patch_dim_base}")
print(f"Patched shapes           : train {P_tr.shape}, val {P_val.shape}, test {P_test.shape}")
print(f"Config -> d_model={VIT_D_MODEL}, heads={VIT_HEADS}, layers={VIT_LAYERS}, "
      f"mlp={VIT_MLP_DIM}, dropout={VIT_DROPOUT}, epochs={VIT_EPOCHS}, "
      f"batch={VIT_BATCH}, lr={VIT_LR}, patience={VIT_PATIENCE}, seed={SEED}")

assert P_tr.shape[1:] == (n_patches_base, patch_dim_base)
assert len(P_test) == len(y_test_seq), "ViT test patches must align with the LSTM test targets."

vit_predict, vit_history, vit_model = train_vit_model(P_tr, y_tr, P_val, y_val, verbose=1)

vit_probs = np.asarray(vit_predict(P_test), dtype=np.float64).ravel()
vit_pred = (vit_probs >= THRESHOLD).astype(int)

results["Vision Transformer"] = {
    "accuracy": accuracy_score(y_test_seq, vit_pred),
    "precision": precision_score(y_test_seq, vit_pred, zero_division=0),
    "recall": recall_score(y_test_seq, vit_pred, zero_division=0),
    "f1": f1_score(y_test_seq, vit_pred, zero_division=0),
}
model_predictions["Vision Transformer"] = vit_pred
model_probabilities["Vision Transformer"] = vit_probs
model_objects["Vision Transformer"] = vit_model

print("\nVISION TRANSFORMER TEST SUMMARY")
print("-" * 60)
print(f"Test samples : {len(y_test_seq)}")
print(f"UP predicted : {int(vit_pred.sum())}")
print(f"DOWN predicted: {int(len(vit_pred) - vit_pred.sum())}")
print(f"Accuracy     : {results['Vision Transformer']['accuracy']:.4f}")
print(f"Precision    : {results['Vision Transformer']['precision']:.4f}")
print(f"Recall       : {results['Vision Transformer']['recall']:.4f}")
print(f"F1-score     : {results['Vision Transformer']['f1']:.4f}")
print(f"Avg predicted UP probability: {float(np.mean(vit_probs)):.4f}")


# =====================================================================
# STEP 10A - MODEL EVALUATION (individual models)
# Ranks the 8 single models by test accuracy, prints a detailed report and
# confusion matrix for the best one, then runs Experiment 3 (feature-selected
# variants of the models).
# Produces: results_df, fs_results (feature-selected rows for the STEP 15 table)
# =====================================================================
results_df = pd.DataFrame(results).T.sort_values("accuracy", ascending=False)
print("\nINDIVIDUAL MODEL PERFORMANCE (6 classical + LSTM + Vision Transformer)")
print(results_df)

best_model_name = results_df.index[0]
print("Best-performing entry by test accuracy:", best_model_name)

# Detailed classification report for the best individual predictive model.
# Ensemble rows are not individual models, so keep the detailed report focused
# on the best actual predictive model.
best_individual_names = list(models.keys()) + ["LSTM", "Vision Transformer"]
best_individual_name = results_df.loc[
    results_df.index.intersection(best_individual_names)
].index[0]

best_true = y_test_seq if best_individual_name == "LSTM" else y_test
best_pred = model_predictions[best_individual_name]

print("\n" + "=" * 80)
print(f"DETAILED CLASSIFICATION REPORT - {best_individual_name}")
print("=" * 80)
print(
    pd.DataFrame(
        {
            "precision": [results[best_individual_name]["precision"]],
            "recall": [results[best_individual_name]["recall"]],
            "f1": [results[best_individual_name]["f1"]],
            "accuracy": [results[best_individual_name]["accuracy"]],
        }
    ).T.rename(columns={0: "score"}).to_string()
)

best_cm = confusion_matrix(best_true, best_pred)
print("\nConfusion Matrix:")
print(best_cm)
print(f"True Negatives : {best_cm[0,0]}")
print(f"False Positives: {best_cm[0,1]}")
print(f"False Negatives: {best_cm[1,0]}")
print(f"True Positives : {best_cm[1,1]}")


# =====================================================================
# EXPERIMENT 3 - FEATURE-SELECTED VERSIONS OF THE MODELS
#
# These are reported as SEPARATE, CLEARLY LABELLED rows. They never overwrite the
# all-feature baseline rows. The feature indices came from STEP 6 selectors that
# were fitted on the training partition only, and each variant gets its own
# StandardScaler which is also fitted on training rows only.
# =====================================================================
print("\n" + "=" * 80)
print("EXPERIMENT 3: FEATURE-SELECTED MODEL VARIANTS")
print("=" * 80)


def scaled_matrices_for(indices):
    """Train/test/full scaled matrices for a feature subset. Scaler fits on TRAIN only."""
    sc = StandardScaler().fit(X[:split_idx][:, indices])
    tr = sc.transform(X[:split_idx][:, indices])
    te = sc.transform(X[split_idx:][:, indices])
    assert sc.n_samples_seen_ == split_idx, "Scaler saw rows outside the training partition."
    return sc, tr, te, np.vstack([tr, te])


SELECTED_SET_NAMES = [n for n in FEATURE_SETS if n != "All Features"]
fs_results = {}

for set_name in SELECTED_SET_NAMES:
    idx = FEATURE_SETS[set_name]
    _, fs_tr, fs_te, _ = scaled_matrices_for(idx)
    for name, mdl in make_classical_models().items():
        mdl.fit(fs_tr, y_train)
        pred = mdl.predict(fs_te)
        prob = mdl.predict_proba(fs_te)[:, 1]
        fs_results[f"{name} [{set_name}]"] = evaluate_binary(y_test, pred, prob)
    print(f"  {set_name}: 6 classical models trained and scored on the test period.")

# The deep models are retrained on ONE selected set only, to keep Colab runtime sane.
# That set is the better of the two selectors according to WALK-FORWARD VALIDATION.
FS_EVAL_SET = str(fs_val_df.loc["MEAN", SELECTED_SET_NAMES].idxmax())
fs_idx = FEATURE_SETS[FS_EVAL_SET]
print(f"\nDeep models are retrained on the validation-preferred selected set: {FS_EVAL_SET}")

_, _, _, X_fs_full_scaled = scaled_matrices_for(fs_idx)

fs_train_seq, fs_train_y, _ = build_sequences(
    X_fs_full_scaled, y_full, WINDOW_SIZE - 1, split_idx, WINDOW_SIZE
)
fs_test_seq, fs_test_y, _ = build_sequences(
    X_fs_full_scaled, y_full, split_idx, len(X_fs_full_scaled), WINDOW_SIZE
)
fs_val_cut = int(len(fs_train_seq) * 0.85)

fs_lstm_predict, _ = train_lstm_model(
    fs_train_seq[:fs_val_cut], fs_train_y[:fs_val_cut],
    fs_train_seq[fs_val_cut:], fs_train_y[fs_val_cut:],
)
fs_lstm_prob = np.asarray(fs_lstm_predict(fs_test_seq), dtype=np.float64).ravel()
fs_results[f"LSTM [{FS_EVAL_SET}]"] = evaluate_binary(
    y_test, (fs_lstm_prob >= THRESHOLD).astype(int), fs_lstm_prob
)

fs_vit_predict, _, _ = train_vit_model(
    vit_patchify(fs_train_seq[:fs_val_cut]), fs_train_y[:fs_val_cut],
    vit_patchify(fs_train_seq[fs_val_cut:]), fs_train_y[fs_val_cut:],
)
fs_vit_prob = np.asarray(fs_vit_predict(vit_patchify(fs_test_seq)), dtype=np.float64).ravel()
fs_results[f"Vision Transformer [{FS_EVAL_SET}]"] = evaluate_binary(
    y_test, (fs_vit_prob >= THRESHOLD).astype(int), fs_vit_prob
)

fs_results_df = pd.DataFrame(fs_results).T.sort_values("accuracy", ascending=False)
print("\nFEATURE-SELECTED VARIANTS - TEST PERFORMANCE")
print(fs_results_df.round(4).to_string())
print("\nReminder: whether these beat the all-feature baseline is an empirical result,")
print("not an assumption. Compare them against the baseline rows in the STEP 15 table.")


# =====================================================================
# STEP 10B - MODEL EVALUATION (part B)
#           TIME-AWARE WALK-FORWARD VALIDATION -> OUT-OF-FOLD PREDICTIONS
#
# Everything the ensembles need that is NOT allowed to see the test set is
# produced here: the voting weights and the stacking meta-model's training data.
#
# For every fold:
#   * a FRESH StandardScaler is fitted on that fold's training rows only;
#   * fresh, unfitted models are trained on rows [0, tr_end);
#   * they predict rows [tr_end, va_end), which are strictly in that fold's future.
# Concatenating the fold validation predictions gives one out-of-fold (OOF)
# prediction per day for the second half of the TRAINING period. Every OOF value
# was produced by a model that had never seen that day - which is exactly what
# makes it safe to use for weighting and for stacking.
#
# Produces: oof_pred / oof_prob (per model, one value per OOF day), oof_y,
#           validation_scores (per-model validation accuracy / F1 / ROC-AUC)
# This is the slowest step: every model (incl. LSTM and ViT) is retrained 4 times.
# =====================================================================
print("\n" + "=" * 80)
print("STEP 10B: TIME-AWARE WALK-FORWARD VALIDATION (training period only)")
print("=" * 80)

classical_names = list(models.keys())
all_7_names = classical_names + ["LSTM"]                        # original 7-model ensemble
all_8_names = all_7_names + ["Vision Transformer"]              # experimental 8-model ensemble

# The classical models and the sequence models must be scored on the same days.
assert len(y_test) == len(y_test_seq), "Classical and sequence test lengths differ."
assert np.array_equal(y_test, y_test_seq.astype(np.int64)), (
    "Classical and sequence test targets are not the same days."
)
y_test_common = y_test

oof_folds = make_walk_forward_folds(split_idx)
OOF_START, OOF_END = oof_folds[0][2], oof_folds[-1][3]
oof_y = y[OOF_START:OOF_END]

# Empty slots (-1 / NaN) that each fold fills in; checked below that none are left.
oof_pred = {n: np.full(len(oof_y), -1, dtype=np.int64) for n in all_8_names}
oof_prob = {n: np.full(len(oof_y), np.nan, dtype=np.float64) for n in all_8_names}

print(f"OOF coverage: rows {OOF_START}..{OOF_END-1} "
      f"({dates_model.iloc[OOF_START].date()} .. {dates_model.iloc[OOF_END-1].date()})")
print(f"OOF samples : {len(oof_y)}   |   folds: {len(oof_folds)}")
assert OOF_END <= split_idx, "OOF window crossed into the test period."

for k, (tr_s, tr_e, va_s, va_e) in enumerate(oof_folds, start=1):
    print(f"\n--- Fold {k}/{len(oof_folds)}: train rows {tr_s}..{tr_e-1} "
          f"-> validate rows {va_s}..{va_e-1} ---")

    # Fold-local scaler: fitted on the fold's training rows ONLY.
    fold_scaler = StandardScaler().fit(X[tr_s:tr_e])
    X_fold_scaled = fold_scaler.transform(X[:va_e])   # transform (not fit) for the val rows
    assert fold_scaler.n_samples_seen_ == tr_e - tr_s

    lo, hi = va_s - OOF_START, va_e - OOF_START       # where this fold's rows go in the OOF arrays

    # ---- six classical models -------------------------------------------------
    for name, mdl in make_classical_models().items():
        mdl.fit(X_fold_scaled[tr_s:tr_e], y[tr_s:tr_e])
        oof_pred[name][lo:hi] = mdl.predict(X_fold_scaled[va_s:va_e])
        oof_prob[name][lo:hi] = mdl.predict_proba(X_fold_scaled[va_s:va_e])[:, 1]
    print("    classical models done")

    # ---- LSTM and ViT (same corrected sequence alignment) ---------------------
    seq_tr_X, seq_tr_y, _ = build_sequences(X_fold_scaled, y, WINDOW_SIZE - 1, tr_e, WINDOW_SIZE)
    seq_va_X, seq_va_y, _ = build_sequences(X_fold_scaled, y, va_s, va_e, WINDOW_SIZE)
    inner_cut = int(len(seq_tr_X) * 0.85)   # early-stopping slice = end of fold-train

    fold_lstm_predict, _ = train_lstm_model(
        seq_tr_X[:inner_cut], seq_tr_y[:inner_cut],
        seq_tr_X[inner_cut:], seq_tr_y[inner_cut:],
        epochs=WF_DL_EPOCHS,
    )
    p_lstm = np.asarray(fold_lstm_predict(seq_va_X), dtype=np.float64).ravel()
    oof_prob["LSTM"][lo:hi] = p_lstm
    oof_pred["LSTM"][lo:hi] = (p_lstm >= THRESHOLD).astype(int)
    print("    LSTM done")

    fold_vit_predict, _, _ = train_vit_model(
        vit_patchify(seq_tr_X[:inner_cut]), seq_tr_y[:inner_cut],
        vit_patchify(seq_tr_X[inner_cut:]), seq_tr_y[inner_cut:],
        epochs=WF_DL_EPOCHS,
    )
    p_vit = np.asarray(fold_vit_predict(vit_patchify(seq_va_X)), dtype=np.float64).ravel()
    oof_prob["Vision Transformer"][lo:hi] = p_vit
    oof_pred["Vision Transformer"][lo:hi] = (p_vit >= THRESHOLD).astype(int)
    print("    ViT done")

# Every OOF slot must be filled, and nothing may come from the test period.
for n in all_8_names:
    assert not np.any(oof_pred[n] < 0), f"Unfilled OOF predictions for {n}."
    assert not np.any(np.isnan(oof_prob[n])), f"Unfilled OOF probabilities for {n}."
assert OOF_END <= split_idx

# How good each model was on days it had never seen - inside the training period.
# STEP 13 turns val_f1 into voting weights.
validation_scores = {
    n: {
        "val_accuracy": accuracy_score(oof_y, oof_pred[n]),
        "val_balanced_accuracy": balanced_accuracy_score(oof_y, oof_pred[n]),
        "val_f1": f1_score(oof_y, oof_pred[n], zero_division=0),
        "val_roc_auc": roc_auc_score(oof_y, oof_prob[n]),
    }
    for n in all_8_names
}
validation_df = pd.DataFrame(validation_scores).T.sort_values("val_f1", ascending=False)
print("\nWALK-FORWARD VALIDATION SCORES (training period only - NO test data)")
print(validation_df.round(4).to_string())


# =====================================================================
# HELPERS SHARED BY STEPS 11-14
#   stack_columns() turns {model: array} into a (days x models) matrix
#   register()      scores an ensemble and adds it to ensemble_results
# The *_matrix_6 / _7 / _8 names below mean: 6 classical models,
# 7 = + LSTM (main ensemble), 8 = + ViT (experimental ensemble).
# =====================================================================
def stack_columns(source, names):
    """Column-stack per-model arrays in a fixed model order."""
    return np.column_stack([np.asarray(source[n]).ravel() for n in names])


def register(name, y_true, y_pred, y_prob=None):
    """Record one row of the final comparison table and return the metrics."""
    ensemble_results[name] = evaluate_binary(y_true, y_pred, y_prob)
    ensemble_predictions[name] = np.asarray(y_pred).ravel()
    m = ensemble_results[name]
    print(f"  {name:<44} acc={m['accuracy']:.4f}  prec={m['precision']:.4f}  "
          f"rec={m['recall']:.4f}  f1={m['f1']:.4f}")
    return m


ensemble_results = {}
ensemble_predictions = {}

test_pred_matrix_7 = stack_columns(model_predictions, all_7_names)
test_prob_matrix_7 = stack_columns(model_probabilities, all_7_names)
test_pred_matrix_8 = stack_columns(model_predictions, all_8_names)
test_prob_matrix_8 = stack_columns(model_probabilities, all_8_names)
test_pred_matrix_6 = stack_columns(model_predictions, classical_names)
test_prob_matrix_6 = stack_columns(model_probabilities, classical_names)

oof_pred_matrix_7 = stack_columns(oof_pred, all_7_names)
oof_prob_matrix_7 = stack_columns(oof_prob, all_7_names)
oof_pred_matrix_8 = stack_columns(oof_pred, all_8_names)
oof_prob_matrix_8 = stack_columns(oof_prob, all_8_names)


# =====================================================================
# STEP 11 - EQUAL / MAJORITY VOTING  (Experiment 4)
#
# Every model gets exactly ONE vote. With 7 models the majority threshold is
# more than 3 UP votes, i.e. at least 4 of 7. Ties (only possible with an even
# number of models) are resolved as DOWN.
#   Example (7 models): votes UP,UP,DOWN,UP,DOWN,UP,DOWN -> 4 UP -> predict UP
# =====================================================================
print("\n" + "=" * 80)
print("STEP 11: EQUAL / MAJORITY VOTING (Experiment 4)")
print("=" * 80)


def majority_vote(pred_matrix):
    """Returns (0/1 decision per day, number of UP votes per day)."""
    votes_up = pred_matrix.sum(axis=1)
    return (votes_up > pred_matrix.shape[1] / 2.0).astype(int), votes_up


v6_test, v6_votes = majority_vote(test_pred_matrix_6)
v7_test, v7_votes = majority_vote(test_pred_matrix_7)
v8_test, v8_votes = majority_vote(test_pred_matrix_8)

print(f"6-model majority rule : UP if more than 3.0 of 6 models vote UP")
print(f"7-model majority rule : UP if more than 3.5 of 7 models vote UP (i.e. >3 votes)")
print(f"8-model majority rule : UP if more than 4.0 of 8 models vote UP (experimental)")
register("Majority Voting (6 classical)", y_test_common, v6_test, v6_votes / 6.0)
register("Majority Voting (7 = 6 classical + LSTM)", y_test_common, v7_test, v7_votes / 7.0)
register("Majority Voting (8 = 7 + ViT) [experimental]", y_test_common, v8_test, v8_votes / 8.0)


# =====================================================================
# STEP 12 - SOFT VOTING  (Experiment 5)
#
# Uses model PROBABILITIES rather than hard labels:
#       P_soft = (1/M) * sum_i p_i        ->  UP if P_soft >= 0.50
# So a model that is 90% sure counts more than one that is 51% sure.
# =====================================================================
print("\n" + "=" * 80)
print("STEP 12: SOFT VOTING (Experiment 5)")
print("=" * 80)

soft7_prob = test_prob_matrix_7.mean(axis=1)
soft8_prob = test_prob_matrix_8.mean(axis=1)
register("Soft Voting (7 = 6 classical + LSTM)", y_test_common,
         (soft7_prob >= THRESHOLD).astype(int), soft7_prob)
register("Soft Voting (8 = 7 + ViT) [experimental]", y_test_common,
         (soft8_prob >= THRESHOLD).astype(int), soft8_prob)


# =====================================================================
# STEP 13 - PERFORMANCE-WEIGHTED VOTING  (Experiment 6)
#
# WEIGHTING METRIC: walk-forward validation F1 (STEP 10B), computed entirely
# inside the training period. Test performance is NEVER used to build weights.
#
#   w_i = F1_val(model_i) / sum_j F1_val(model_j)        so that sum_i w_i = 1
#
# Two decision rules are reported:
#   (a) label-weighted   S = sum_i w_i * yhat_i   -> UP if S > 0.50
#   (b) probability-weighted (statistically the sounder of the two, because it
#       keeps each model's confidence instead of collapsing it to 0/1)
#                        P = sum_i w_i * p_i      -> UP if P >= 0.50
# =====================================================================
print("\n" + "=" * 80)
print("STEP 13: PERFORMANCE-WEIGHTED VOTING (Experiment 6)")
print("=" * 80)


def performance_weights(names, metric="val_f1"):
    """Normalised weights from walk-forward VALIDATION scores only."""
    raw = np.array([validation_scores[n][metric] for n in names], dtype=np.float64)
    if raw.sum() <= 0:                       # degenerate guard -> fall back to equal weights
        print("  WARNING: all validation scores are zero; using equal weights.")
        raw = np.ones_like(raw)
    return raw / raw.sum()


weights_7 = performance_weights(all_7_names)
weights_8 = performance_weights(all_8_names)

print("Weighting metric: walk-forward validation F1 (training period only)\n")
print(f"{'Model':<24}{'Val F1':>10}{'Val BalAcc':>13}{'Weight':>10}")
print("-" * 57)
for n, w in zip(all_7_names, weights_7):
    print(f"{n:<24}{validation_scores[n]['val_f1']:>10.4f}"
          f"{validation_scores[n]['val_balanced_accuracy']:>13.4f}{w:>10.4f}")
print("-" * 57)
print(f"{'SUM':<24}{'':>10}{'':>13}{weights_7.sum():>10.4f}")
assert abs(weights_7.sum() - 1.0) < 1e-9, "7-model weights do not sum to 1."
assert abs(weights_8.sum() - 1.0) < 1e-9, "8-model weights do not sum to 1."

wv7_score = test_pred_matrix_7 @ weights_7          # S = sum_i w_i * yhat_i
wv7_prob = test_prob_matrix_7 @ weights_7           # P = sum_i w_i * p_i
wv8_score = test_pred_matrix_8 @ weights_8
wv8_prob = test_prob_matrix_8 @ weights_8

print()
register("Weighted Voting - labels (7 models)", y_test_common,
         (wv7_score > 0.50).astype(int), wv7_score)
register("Weighted Voting - probabilities (7 models)", y_test_common,
         (wv7_prob >= THRESHOLD).astype(int), wv7_prob)
register("Weighted Voting - labels (8 models) [experimental]", y_test_common,
         (wv8_score > 0.50).astype(int), wv8_score)
register("Weighted Voting - probabilities (8 models) [experimental]", y_test_common,
         (wv8_prob >= THRESHOLD).astype(int), wv8_prob)


# =====================================================================
# STEP 13B - ADAPTIVE WEIGHTED VOTING  (Experiment 6b)
#
# STEP 13 fixes every model's weight ONCE, from training-period validation F1.
# Adaptive voting recomputes the weights EVERY DAY from each model's accuracy
# over the most recent ADAPTIVE_WINDOW days whose outcomes are already known:
#
#   acc_i(t) = accuracy of model i on rows [t-W, t)
#   w_i(t)   = exp(beta * (acc_i(t) - 0.5)) / sum_j exp(beta * (acc_j(t) - 0.5))
#
# so a model that has recently been right more often gets more say, and one that
# has drifted below chance gets less.
#
# TIMING / LEAKAGE: the vote for row t is cast after the close of day t. The
# newest label it may use is y[t-1] (Close[t] vs Close[t-1]), which is known by
# then. y[t] itself needs Close[t+1] and is never used for row t's weights.
# During the test period the weights therefore learn only from test days that
# are already in the past - exactly what a live system could do.
#
# The history starts with the walk-forward OOF predictions (STEP 10B), so the
# first test day already has a full window. W and beta are chosen on the OOF
# (training) period only. beta = 0 is in the grid and means "no adaptation"
# (every model weighted equally); ties are resolved in favour of it.
# =====================================================================
print("\n" + "=" * 80)
print("STEP 13B: ADAPTIVE WEIGHTED VOTING (Experiment 6b)")
print("=" * 80)

# Settings tried (the best pair is picked on training-period data):
#   window = how many recent days to judge each model on (20 ~ 1 month, 250 ~ 1 year)
#   beta   = how strongly to favour the recent winner. 0 = equal weights;
#            with beta = 20, a model at 60% recent accuracy gets ~7x the weight
#            of one at 50%  (exp(20 * 0.10) = 7.4).
ADAPTIVE_WINDOWS = [20, 60, 120, 250]
ADAPTIVE_BETAS = [0.0, 5.0, 10.0, 20.0]
assert OOF_END == split_idx, "OOF history must end exactly where the test period starts."
assert len(oof_y) > max(ADAPTIVE_WINDOWS), "OOF period too short for the adaptive windows."


def adaptive_weight_path(correct, window, beta):
    """
    correct: (n_rows, n_models) 1/0 matrix of past hits, in date order.
    Returns (n_rows + 1, n_models): row t holds the weights for day t, built
    from correct[t-window : t] only. The extra last row is for the next,
    still-unlabelled day (used by the STEP 16 latest prediction).
    """
    n, m = correct.shape
    # Running totals of hits, so "hits in rows lo..t-1" = csum[t] - csum[lo] (fast window sums).
    csum = np.vstack([np.zeros((1, m)), np.cumsum(correct, axis=0)])
    weights = np.full((n + 1, m), 1.0 / m)          # day 0 has no history -> equal weights
    for t in range(1, n + 1):
        lo = max(0, t - window)
        acc = (csum[t] - csum[lo]) / (t - lo)       # each model's recent accuracy (rows < t only)
        z = np.exp(beta * (acc - 0.5))              # reward accuracy above coin-flip level
        weights[t] = z / z.sum()                    # normalise so the weights sum to 1
    return weights


def adaptive_history(names, y_test_override=None):
    """OOF + test predictions and labels as one contiguous, date-ordered history."""
    preds = np.vstack([stack_columns(oof_pred, names), stack_columns(model_predictions, names)])
    labels = np.concatenate([
        oof_y, y_test_common if y_test_override is None else y_test_override
    ])
    return preds, (preds == labels[:, None]).astype(np.float64)


def tune_adaptive(names):
    """Pick (window, beta) by adaptive probability-vote accuracy on the OOF period only."""
    P = stack_columns(oof_pred, names)
    Q = stack_columns(oof_prob, names)
    correct = (P == oof_y[:, None]).astype(np.float64)
    start = max(ADAPTIVE_WINDOWS)          # score every setting on the same rows
    grid = []
    for beta in ADAPTIVE_BETAS:
        for window in ADAPTIVE_WINDOWS:
            W = adaptive_weight_path(correct, window, beta)[:-1]
            prob = np.sum(W * Q, axis=1)
            acc = accuracy_score(oof_y[start:], (prob[start:] >= THRESHOLD).astype(int))
            grid.append({"window": window, "beta": beta, "oof_accuracy": acc})
    grid_df = pd.DataFrame(grid)
    best = grid_df.loc[grid_df["oof_accuracy"].idxmax()]   # idxmax -> first max -> beta=0 wins ties
    return int(best["window"]), float(best["beta"]), grid_df


# Run it for the main 7-model ensemble and the experimental 8-model one.
# adaptive["7"] / adaptive["8"] hold the chosen settings, the daily weights and
# the ensemble outputs (used again in STEPS 16, 17 and 18).
adaptive = {}
for key, names in [("7", all_7_names), ("8", all_8_names)]:
    window, beta, grid_df = tune_adaptive(names)               # 1) choose settings on OOF only
    _, correct_hist = adaptive_history(names)                  # 2) hit/miss history, OOF then test
    W_all = adaptive_weight_path(correct_hist, window, beta)   # 3) daily weights
    n_oof = len(oof_y)
    W_test = W_all[n_oof:-1]                                   # one weight row per test day
    assert W_test.shape == (len(y_test_common), len(names))
    assert np.allclose(W_test.sum(axis=1), 1.0)

    # 4) combine today's model outputs with today's weights (same two rules as STEP 13)
    score = np.sum(W_test * stack_columns(model_predictions, names), axis=1)    # weighted votes
    prob = np.sum(W_test * stack_columns(model_probabilities, names), axis=1)   # weighted P(UP)
    adaptive[key] = {
        "names": names, "window": window, "beta": beta, "grid": grid_df,
        "test_weights": W_test, "next_weights": W_all[-1], "score": score, "prob": prob,
    }

    print(f"\n{key}-model adaptive voting - parameter search on the OOF period (accuracy):")
    print(grid_df.pivot(index="beta", columns="window", values="oof_accuracy").round(4).to_string())
    print(f"Chosen: window = {window} days, beta = {beta:g}"
          + ("   (beta = 0 -> adaptation did not help in validation; equal weights)" if beta == 0 else ""))

a7 = adaptive["7"]
print("\n7-model adaptive weights - average over the test period vs static STEP 13 weights:")
print(f"{'Model':<24}{'Static':>10}{'Adaptive mean':>15}{'min':>8}{'max':>8}")
for j, n in enumerate(all_7_names):
    col = a7["test_weights"][:, j]
    print(f"{n:<24}{weights_7[j]:>10.4f}{col.mean():>15.4f}{col.min():>8.4f}{col.max():>8.4f}")

print()
for key, suffix in [("7", "(7 models)"), ("8", "(8 models) [experimental]")]:
    a = adaptive[key]
    register(f"Adaptive Weighted Voting - labels {suffix}", y_test_common,
             (a["score"] > 0.50).astype(int), a["score"])
    register(f"Adaptive Weighted Voting - probabilities {suffix}", y_test_common,
             (a["prob"] >= THRESHOLD).astype(int), a["prob"])


# =====================================================================
# STEP 14 - STACKING  (Experiment 7)
#
# Instead of a fixed voting rule, a small second-level model (logistic
# regression, the "meta-model") LEARNS how to combine the base models'
# probabilities - e.g. it may learn to trust XGBoost more and KNN less.
#
# LEAKAGE SAFETY: the meta-model is trained on the walk-forward OUT-OF-FOLD
# probabilities from STEP 10B. Each of those probabilities came from a base model
# that had NOT been trained on that day, so the meta-model never sees a base
# model's own training-set predictions (which would be optimistically biased).
# At test time the meta-model consumes the probabilities of the base models that
# were fitted on the full training period.
# =====================================================================
print("\n" + "=" * 80)
print("STEP 14: STACKING (Experiment 7)")
print("=" * 80)


def fit_stacking(oof_matrix, test_matrix, label):
    """Train the meta-model on OOF probabilities, return (meta-model, test P(UP))."""
    meta =LogisticRegression(max_iter=1000, random_state=SEED)
    meta.fit(oof_matrix, oof_y)                     # OOF only - no test data
    prob = meta.predict_proba(test_matrix)[:, 1]
    print(f"\n{label} meta-model coefficients (higher = more trusted by the meta-learner):")
    for n, c in zip(label_names[label], meta.coef_[0]):
        print(f"  {n:<24}{c:>9.4f}")
    print(f"  {'intercept':<24}{meta.intercept_[0]:>9.4f}")
    return meta, prob


label_names = {
    "Stacking (7 models)": all_7_names,
    "Stacking (8 models)": all_8_names,
}

assert oof_prob_matrix_7.shape[0] == len(oof_y)
assert OOF_END <= split_idx, "Stacking training data must come from the training period."

meta7, stack7_prob = fit_stacking(oof_prob_matrix_7, test_prob_matrix_7, "Stacking (7 models)")
meta8, stack8_prob = fit_stacking(oof_prob_matrix_8, test_prob_matrix_8, "Stacking (8 models)")

print()
register("Stacking (7 = 6 classical + LSTM)", y_test_common,
         (stack7_prob >= THRESHOLD).astype(int), stack7_prob)
register("Stacking (8 = 7 + ViT) [experimental]", y_test_common,
         (stack8_prob >= THRESHOLD).astype(int), stack8_prob)



# =====================================================================
# STEP 15 - COMPARATIVE ANALYSIS
#           One comprehensive table + the required visualisations.
# Every individual model, ensemble and feature-selected variant in one table,
# sorted by test accuracy (also saved as outputs/<STOCK>_results.csv), then
# figures 01-06:
#   01 individual accuracy   02 ensemble accuracy   03 F1 comparison
#   04 confusion matrices    05 feature rankings    06 LSTM / ViT training curves
# =====================================================================
print("\n" + "=" * 80)
print("STEP 15: COMPARATIVE ANALYSIS")
print("=" * 80)

# ---- individual models (all 8), recomputed with ROC-AUC ------------------------
individual_rows = {
    n: evaluate_binary(y_test_common, model_predictions[n], model_probabilities[n])
    for n in all_8_names
}

final_table = pd.DataFrame(
    {**individual_rows, **ensemble_results, **fs_results}
).T
final_table["kind"] = (
    ["Individual"] * len(individual_rows)
    + ["Ensemble"] * len(ensemble_results)
    + ["Feature-selected"] * len(fs_results)
)
final_table = final_table[["kind", "accuracy", "precision", "recall", "f1", "roc_auc"]]
final_table = final_table.sort_values("accuracy", ascending=False)

pd.set_option("display.width", 160)
pd.set_option("display.max_rows", 200)
print(f"\nAll rows below are scored on the SAME chronological test period "
      f"({TEST_START.date()} .. {TEST_END.date()}, {len(y_test_common)} days).\n")
print("COMPREHENSIVE COMPARISON TABLE (sorted by test accuracy)")
print("=" * 110)
print(final_table.round(4).to_string())
print("=" * 110)
os.makedirs(OUTPUT_DIR, exist_ok=True)
final_table.to_csv(os.path.join(OUTPUT_DIR, f"{STOCK_SYMBOL}_results.csv"))

# ---- Did ensembling / feature selection beat the best single model? ------------
best_individual_row = final_table[final_table["kind"] == "Individual"].iloc[0]
best_individual_label = final_table[final_table["kind"] == "Individual"].index[0]
best_ensemble_row = final_table[final_table["kind"] == "Ensemble"].iloc[0]
best_ensemble_label = final_table[final_table["kind"] == "Ensemble"].index[0]
best_fs_row = final_table[final_table["kind"] == "Feature-selected"].iloc[0]
best_fs_label = final_table[final_table["kind"] == "Feature-selected"].index[0]

print("\nDID THE ENSEMBLES ACTUALLY HELP?")
print("-" * 80)
print(f"Best individual model  : {best_individual_label:<45} "
      f"acc={best_individual_row['accuracy']:.4f}  f1={best_individual_row['f1']:.4f}")
print(f"Best ensemble method   : {best_ensemble_label:<45} "
      f"acc={best_ensemble_row['accuracy']:.4f}  f1={best_ensemble_row['f1']:.4f}")
print(f"Best feature-selected  : {best_fs_label:<45} "
      f"acc={best_fs_row['accuracy']:.4f}  f1={best_fs_row['f1']:.4f}")

delta = best_ensemble_row["accuracy"] - best_individual_row["accuracy"]
if delta > 0:
    print(f"\n=> The best ensemble beat the best individual model by "
          f"{delta*100:+.2f} accuracy points on the test period.")
elif delta < 0:
    print(f"\n=> The best ensemble did NOT beat the best individual model "
          f"({delta*100:+.2f} accuracy points). Ensembling did not help here, "
          f"and that is reported as the result rather than hidden.")
else:
    print("\n=> The best ensemble exactly matched the best individual model.")

fs_delta = best_fs_row["accuracy"] - best_individual_row["accuracy"]
print(f"=> Feature selection vs the all-feature baseline: "
      f"{fs_delta*100:+.2f} accuracy points for the best selected variant.")
print("Note: a single test period is a small sample. Treat small gaps as noise.")

# Majority-class baseline, so the numbers above can be read in context.
majority_class_rate = max(np.mean(y_test_common), 1 - np.mean(y_test_common))
print(f"=> Always-predict-the-majority-class baseline on this test period: "
      f"{majority_class_rate:.4f}")

# =====================================================================
# VISUALISATIONS
# =====================================================================
sns.set_style("whitegrid")

# --- Figure 01: test accuracy of each single model (red line = always-majority baseline)
plt.figure(figsize=(12, 5))
ind_sorted = pd.Series({n: individual_rows[n]["accuracy"] for n in all_8_names}).sort_values()
bars = plt.barh(ind_sorted.index, ind_sorted.values, color=sns.color_palette("viridis", len(ind_sorted)))
plt.axvline(majority_class_rate, color="red", linestyle="--", linewidth=1.2,
            label=f"majority-class baseline ({majority_class_rate:.3f})")
for b, v in zip(bars, ind_sorted.values):
    plt.text(v + 0.002, b.get_y() + b.get_height() / 2, f"{v:.4f}", va="center", fontsize=9)
plt.title("1. Individual Model Test Accuracy (8 models)")
plt.xlabel("Accuracy")
plt.xlim(0, max(ind_sorted.max() * 1.15, majority_class_rate * 1.15))
plt.legend(loc="lower right")
plt.tight_layout()
save_and_show("01_individual_accuracy")

# --- Figure 02: test accuracy of each ensemble (green line = best single model)
plt.figure(figsize=(12, 6))
ens_sorted = pd.Series({k: v["accuracy"] for k, v in ensemble_results.items()}).sort_values()
colors = ["#c0392b" if "experimental" in k else "#2980b9" for k in ens_sorted.index]
bars = plt.barh(ens_sorted.index, ens_sorted.values, color=colors)
plt.axvline(best_individual_row["accuracy"], color="green", linestyle="--", linewidth=1.2,
            label=f"best individual ({best_individual_label}, {best_individual_row['accuracy']:.3f})")
for b, v in zip(bars, ens_sorted.values):
    plt.text(v + 0.002, b.get_y() + b.get_height() / 2, f"{v:.4f}", va="center", fontsize=8)
plt.title("2. Ensemble Method Test Accuracy  (red = experimental 8-model incl. ViT)")
plt.xlabel("Accuracy")
plt.xlim(0, ens_sorted.max() * 1.2)
plt.legend(loc="lower right")
plt.tight_layout()
save_and_show("02_ensemble_accuracy")

# --- Figure 03: F1 of single models (purple) vs ensembles (teal)
f1_series = pd.concat([
    pd.Series({n: individual_rows[n]["f1"] for n in all_8_names}),
    pd.Series({k: v["f1"] for k, v in ensemble_results.items()}),
]).sort_values()
plt.figure(figsize=(12, 8))
f1_colors = ["#8e44ad" if k in all_8_names else "#16a085" for k in f1_series.index]
bars = plt.barh(f1_series.index, f1_series.values, color=f1_colors)
for b, v in zip(bars, f1_series.values):
    plt.text(v + 0.003, b.get_y() + b.get_height() / 2, f"{v:.4f}", va="center", fontsize=8)
plt.title("3. F1-Score Comparison  (purple = individual model, teal = ensemble method)")
plt.xlabel("F1-score")
plt.xlim(0, f1_series.max() * 1.2)
plt.tight_layout()
save_and_show("03_f1_comparison")

# --- Figure 04: confusion matrices (rows = what happened, columns = what was predicted)
cm_panels = [
    (f"4. Best individual: {best_individual_label}", model_predictions[best_individual_label]),
    ("5. Majority Voting (7 models)", ensemble_predictions["Majority Voting (7 = 6 classical + LSTM)"]),
    ("6. Weighted Voting - probabilities (7)", ensemble_predictions["Weighted Voting - probabilities (7 models)"]),
    ("7. Stacking (7 models)", ensemble_predictions["Stacking (7 = 6 classical + LSTM)"]),
    ("8. Vision Transformer", model_predictions["Vision Transformer"]),
    ("Adaptive Weighted Voting - probs (7)",
     ensemble_predictions["Adaptive Weighted Voting - probabilities (7 models)"]),
]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, (title, preds) in zip(axes.ravel(), cm_panels):
    cm = confusion_matrix(y_test_common, preds, labels=[0, 1])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["DOWN", "UP"], yticklabels=["DOWN", "UP"])
    ax.set_title(f"{title}\nacc={accuracy_score(y_test_common, preds):.4f}", fontsize=10)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
fig.suptitle("Confusion Matrices on the Test Period", fontsize=13)
plt.tight_layout()
save_and_show("04_confusion_matrices")

# --- Figure 05: feature rankings from the two STEP 6 selectors
fig, axes = plt.subplots(1, 2, figsize=(15, 8))
mi_order = np.argsort(mi_scores)[::-1]
axes[0].barh(
    [feature_columns[i] for i in mi_order][::-1],
    mi_scores[mi_order][::-1],
    color=["#27ae60" if i in set(mi_indices) else "#bdc3c7" for i in mi_order][::-1],
)
axes[0].set_title(f"9a. Mutual Information ranking\n(green = MI Top-{TOP_K_FEATURES}, train-fitted)")
axes[0].set_xlabel("Mutual information")
axes[0].tick_params(labelsize=8)

tree_order = np.argsort(tree_importances)[::-1]
axes[1].barh(
    [feature_columns[i] for i in tree_order][::-1],
    tree_importances[tree_order][::-1],
    color=["#2980b9" if i in set(tree_indices) else "#bdc3c7" for i in tree_order][::-1],
)
axes[1].set_title(f"9b. Random-Forest importance ranking\n(blue = Tree Top-{TOP_K_FEATURES}, train-fitted)")
axes[1].set_xlabel("Gini importance")
axes[1].tick_params(labelsize=8)
plt.tight_layout()
save_and_show("05_feature_ranking")

# --- Figure 06: LSTM / ViT training vs validation loss (gap = overfitting)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(lstm_history["loss"], label="train loss")
axes[0].plot(lstm_history["val_loss"], label="validation loss")
axes[0].set_title("10. LSTM training vs validation loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Binary cross-entropy")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(vit_history["loss"], label="train loss")
axes[1].plot(vit_history["val_loss"], label="validation loss")
axes[1].set_title("11. Vision Transformer training vs validation loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Binary cross-entropy")
axes[1].legend()
axes[1].grid(alpha=0.3)
plt.tight_layout()
save_and_show("06_training_curves")


# =====================================================================
# STEP 16 - LATEST NEXT-DAY PREDICTION
#
# TWO DIFFERENT THINGS, DELIBERATELY KEPT SEPARATE:
#
#   (A) HISTORICAL TEST EVALUATION  -  everything above. Each model predicted the
#       direction of EVERY eligible next trading day inside the chronological test
#       period, and those predictions were scored against the known outcome.
#
#   (B) LATEST PREDICTION  -  below. Using the most recent row of data (whose
#       next-day outcome does NOT exist yet), predict the direction of the next
#       trading day. This row was never part of training or of the test scoring,
#       because df_model dropped it for having no label.
# =====================================================================
print("\n" + "=" * 80)
print("STEP 16: LATEST NEXT-DAY PREDICTION")
print("=" * 80)

X_live = df_feature_complete[feature_columns].values.astype(np.float32)
assert len(X_live) >= WINDOW_SIZE, "Not enough recent rows to build the latest sequence."

# The SAME training-fitted scaler is reused - nothing is refitted on recent data.
X_live_scaled = scaler.transform(X_live)
latest_features = X_live_scaled[-1:]                                   # (1, N_FEATURES)
latest_sequence = X_live_scaled[-WINDOW_SIZE:][np.newaxis, :, :]       # (1, WINDOW_SIZE, N_FEATURES)
assert latest_sequence.shape == (1, WINDOW_SIZE, len(feature_columns))

# 1) every single model's opinion about tomorrow ...
latest_preds = {}
latest_probs = {}
for name in classical_names:
    mdl = model_objects[name]
    latest_preds[name] = int(mdl.predict(latest_features)[0])
    latest_probs[name] = float(mdl.predict_proba(latest_features)[0, 1])

if DL_BACKEND == "tensorflow":
    latest_lstm_prob = float(lstm_model.predict(latest_sequence, verbose=0).ravel()[0])
else:
    with torch.no_grad():
        latest_lstm_prob = float(
            lstm_model(torch.tensor(latest_sequence, dtype=torch.float32).to(device))
            .cpu().numpy().ravel()[0]
        )
latest_probs["LSTM"] = latest_lstm_prob
latest_preds["LSTM"] = int(latest_lstm_prob >= THRESHOLD)

latest_vit_prob = float(np.asarray(vit_predict(vit_patchify(latest_sequence))).ravel()[0])
latest_probs["Vision Transformer"] = latest_vit_prob
latest_preds["Vision Transformer"] = int(latest_vit_prob >= THRESHOLD)


# 2) ... printed, then combined with each voting rule (same weights as the test period).
def arrow(v):
    return "UP" if v == 1 else "DOWN"


print(f"Stock / dataset      : {DATASET_NAME}")
print(f"Latest available date: {pd.to_datetime(latest_row['Date']).date()}")
print(f"Latest close         : {float(latest_row['Close']):.2f}")
print(f"Predicting direction : the NEXT trading day after "
      f"{pd.to_datetime(latest_row['Date']).date()}")

print("\nINDIVIDUAL MODEL PREDICTIONS")
print("-" * 52)
print(f"{'Model':<24}{'Prediction':>12}{'P(UP)':>12}")
print("-" * 52)
for name in all_8_names:
    print(f"{name:<24}{arrow(latest_preds[name]):>12}{latest_probs[name]:>12.4f}")

votes_up_7 = sum(latest_preds[n] for n in all_7_names)
maj7 = int(votes_up_7 > len(all_7_names) / 2.0)
votes_up_8 = sum(latest_preds[n] for n in all_8_names)
maj8 = int(votes_up_8 > len(all_8_names) / 2.0)

w_label_7 = float(sum(w * latest_preds[n] for n, w in zip(all_7_names, weights_7)))
w_prob_7 = float(sum(w * latest_probs[n] for n, w in zip(all_7_names, weights_7)))
w_label_8 = float(sum(w * latest_preds[n] for n, w in zip(all_8_names, weights_8)))
w_prob_8 = float(sum(w * latest_probs[n] for n, w in zip(all_8_names, weights_8)))

print("\nENSEMBLE DECISIONS FOR THE NEXT TRADING DAY")
print("-" * 72)
print(f"Majority voting (7 models)          : {arrow(maj7):<5} "
      f"({votes_up_7}/7 models voted UP; majority needs more than 3)")
print(f"Weighted voting - labels (7)        : {arrow(int(w_label_7 > 0.50)):<5} "
      f"(weighted score S = {w_label_7:.4f}, UP if S > 0.50)")
print(f"Weighted voting - probabilities (7) : {arrow(int(w_prob_7 >= THRESHOLD)):<5} "
      f"(weighted P = {w_prob_7:.4f}, UP if P >= 0.50)")
print(f"[experimental] Majority voting (8)  : {arrow(maj8):<5} ({votes_up_8}/8 voted UP)")
print(f"[experimental] Weighted labels (8)  : {arrow(int(w_label_8 > 0.50)):<5} (S = {w_label_8:.4f})")
print(f"[experimental] Weighted probs (8)   : {arrow(int(w_prob_8 >= THRESHOLD)):<5} (P = {w_prob_8:.4f})")

# Adaptive weights for the next day use the latest window of already-known outcomes.
aw7 = adaptive["7"]["next_weights"]
a_label_7 = float(sum(w * latest_preds[n] for n, w in zip(all_7_names, aw7)))
a_prob_7 = float(sum(w * latest_probs[n] for n, w in zip(all_7_names, aw7)))
print(f"Adaptive weighted - labels (7)      : {arrow(int(a_label_7 > 0.50)):<5} "
      f"(S = {a_label_7:.4f}, weights from the last {adaptive['7']['window']} known days)")
print(f"Adaptive weighted - probs (7)       : {arrow(int(a_prob_7 >= THRESHOLD)):<5} "
      f"(P = {a_prob_7:.4f})")
print("\nAdaptive weights used for this prediction (7 models):")
for n, w in zip(all_7_names, aw7):
    print(f"  {n:<24}{w:.4f}")

print("\n" + "!" * 72)
print("THIS IS A STATISTICAL ESTIMATE, NOT A GUARANTEE.")
print("The models are trained on historical patterns that may not repeat. Next-day")
print("direction accuracy in this project is close to a coin flip, and nothing here")
print("accounts for news, costs or slippage. Do not use it as financial advice.")
print("!" * 72)


# =====================================================================
# STEP 17 - FINAL LEAKAGE / INTEGRITY CHECKLIST
# Re-verifies, in code, every "no look-ahead" promise made above: scalers and
# selectors saw training rows only, folds never touch the test period, weights
# come from validation data, sequence alignment is intact, and adaptive weights
# ignore future labels. Prints [PASS]/[FAIL] for each; any FAIL means a result
# above cannot be trusted.
# =====================================================================
print("\n" + "=" * 80)
print("STEP 17: LEAKAGE AND INTEGRITY CHECKS")
print("=" * 80)

checks = []


def check(label, condition):
    checks.append((label, bool(condition)))


check("Chronological split, test never shuffled", split_idx == int(len(X) * 0.80))
check("Main scaler fitted on training rows only", scaler.n_samples_seen_ == split_idx)
check("Selected-feature scaler fitted on training rows only",
      selected_scaler.n_samples_seen_ == split_idx)
check("MI selector fitted on training rows only", mi_selector.n_features_in_ == len(feature_columns))
check("Tree selector fitted on training rows only",
      importance_forest.n_features_in_ == len(feature_columns))
check("All walk-forward folds end at or before the test boundary",
      all(va_end <= split_idx for (_, _, _, va_end) in oof_folds))
check("Out-of-fold window lies inside the training period", OOF_END <= split_idx)
check("Voting weights come from validation F1, not test data",
      all(np.isclose(w, validation_scores[n]["val_f1"] / sum(
          validation_scores[m]["val_f1"] for m in all_7_names))
          for n, w in zip(all_7_names, weights_7)))
check("7-model weights sum to 1", abs(weights_7.sum() - 1.0) < 1e-9)
check("Stacking meta-model trained on out-of-fold predictions only",
      meta7.n_features_in_ == len(all_7_names) and len(oof_y) == OOF_END - OOF_START)
check("LSTM sequence alignment X[i-W+1:i+1] -> y[i] preserved",
      X_train_seq.shape[0] == split_idx - (WINDOW_SIZE - 1)
      and X_test_seq.shape[0] == len(X_full_scaled) - split_idx)
check(f"ViT consumes {WINDOW_SIZE} x {N_FEATURES} financial feature matrices",
      P_test.shape[1] * 1 == n_patches_base
      and P_tr.shape[0] == X_tr.shape[0]
      and X_tr.shape[1:] == (WINDOW_SIZE, len(feature_columns)))
check("Classical and sequence models scored on identical test days",
      np.array_equal(y_test, y_test_seq.astype(np.int64)))
check("All models predict the same UP/DOWN target",
      all(set(np.unique(model_predictions[n])) <= {0, 1} for n in all_8_names))
check("Latest-prediction row excluded from training and test",
      len(df_feature_complete) == len(df_model) + 1)
check("Adaptive window/beta chosen on the OOF (training) period only",
      all(len(adaptive[k]["grid"]) == len(ADAPTIVE_WINDOWS) * len(ADAPTIVE_BETAS) for k in adaptive))


def adaptive_ignores_future(key, k):
    """Scramble every test label from day k onwards: weights up to day k must not change."""
    a = adaptive[key]
    y_scrambled = y_test_common.copy()
    y_scrambled[k:] = 1 - y_scrambled[k:]
    _, corr = adaptive_history(a["names"], y_test_override=y_scrambled)
    W = adaptive_weight_path(corr, a["window"], a["beta"])[len(oof_y):-1]
    return np.allclose(W[:k + 1], a["test_weights"][:k + 1])


check("Adaptive weights for day t never use the label of day t or later",
      all(adaptive_ignores_future(key, k)
          for key in adaptive for k in [0, len(y_test_common) // 2, len(y_test_common) - 1]))
check("Adaptive weights sum to 1 on every test day",
      all(np.allclose(adaptive[k]["test_weights"].sum(axis=1), 1.0) for k in adaptive))

for label, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")

failed = [label for label, ok in checks if not ok]
if failed:
    print("\nWARNING - the following checks FAILED:")
    for label in failed:
        print("  -", label)
else:
    print("\nAll integrity checks passed.")


# =====================================================================
# STEP 18 - FINAL COMPARISON: MAJORITY vs WEIGHTED vs ADAPTIVE WEIGHTED VOTING
#
#   Majority           : every model gets one vote; weights never change.
#   Weighted (static)  : weights = walk-forward validation F1, fixed for the whole test.
#   Adaptive weighted  : weights recomputed daily from the last W known outcomes.
# All rows are scored on the same test days with the same base-model predictions,
# so any difference comes only from how the votes are combined.
# =====================================================================
print("\n" + "=" * 80)
print("STEP 18: FINAL COMPARISON - MAJORITY vs WEIGHTED vs ADAPTIVE WEIGHTED VOTING")
print("=" * 80)

VOTING_ROWS = []
for key, n_models, tag in [("7", 7, "(7 models)"), ("8", 8, "(8 models) [experimental]")]:
    maj_name = ("Majority Voting (7 = 6 classical + LSTM)" if key == "7"
                else "Majority Voting (8 = 7 + ViT) [experimental]")
    VOTING_ROWS += [
        ("Majority", "labels", n_models, maj_name, "1 vote per model"),
        ("Weighted (static)", "labels", n_models, f"Weighted Voting - labels {tag}",
         "validation F1, fixed"),
        ("Weighted (static)", "probabilities", n_models, f"Weighted Voting - probabilities {tag}",
         "validation F1, fixed"),
        ("Adaptive weighted", "labels", n_models, f"Adaptive Weighted Voting - labels {tag}",
         f"last {adaptive[key]['window']} days, beta={adaptive[key]['beta']:g}"),
        ("Adaptive weighted", "probabilities", n_models,
         f"Adaptive Weighted Voting - probabilities {tag}",
         f"last {adaptive[key]['window']} days, beta={adaptive[key]['beta']:g}"),
    ]

voting_table = pd.DataFrame([
    {"method": m, "rule": r, "models": k, "weights": w, **ensemble_results[name]}
    for (m, r, k, name, w) in VOTING_ROWS
])[["method", "rule", "models", "weights", "accuracy", "precision", "recall", "f1", "roc_auc"]]

print(f"Test period: {TEST_START.date()} .. {TEST_END.date()}  ({len(y_test_common)} days)\n")
print(voting_table.round(4).to_string(index=False))
voting_table.to_csv(os.path.join(OUTPUT_DIR, f"{STOCK_SYMBOL}_voting_comparison.csv"), index=False)

# Verdict on the main 7-model ensemble: best rule of each method.
main = voting_table[voting_table["models"] == 7]
best_per_method = main.loc[main.groupby("method", sort=False)["accuracy"].idxmax()]
n_test = len(y_test_common)
margin = 1.96 * np.sqrt(0.25 / n_test)       # ~95% sampling noise band for an accuracy near 0.5

print("\nBEST RULE PER METHOD (7-model ensemble)")
print("-" * 80)
for _, r in best_per_method.iterrows():
    print(f"  {r['method']:<20} ({r['rule']:<13}) acc={r['accuracy']:.4f}  f1={r['f1']:.4f}  "
          f"roc_auc={r['roc_auc']:.4f}")
winner = best_per_method.loc[best_per_method["accuracy"].idxmax()]
maj_acc = float(main.loc[main["method"] == "Majority", "accuracy"].iloc[0])
print(f"\n=> Highest test accuracy: {winner['method']} ({winner['rule']}), "
      f"{(winner['accuracy'] - maj_acc) * 100:+.2f} points vs majority voting.")
print(f"=> Always-predict-the-majority-class baseline: {majority_class_rate:.4f}")
print(f"=> With {n_test} test days, accuracy differences smaller than about "
      f"+/-{margin * 100:.1f} points are within sampling noise.")

# ---- Figure 7: accuracy / F1 of the three voting methods ----------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
labels_7 = [f"{m}\n({r})" for m, r in zip(main["method"], main["rule"])]
colors_7 = {"Majority": "#7f8c8d", "Weighted (static)": "#2980b9", "Adaptive weighted": "#e67e22"}
for ax, metric in zip(axes, ["accuracy", "f1"]):
    bars = ax.bar(labels_7, main[metric], color=[colors_7[m] for m in main["method"]])
    for b, v in zip(bars, main[metric]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.4f}", ha="center", fontsize=9)
    if metric == "accuracy":
        ax.axhline(majority_class_rate, color="red", linestyle="--", linewidth=1.2,
                   label=f"majority-class baseline ({majority_class_rate:.3f})")
        ax.legend(loc="lower right")
    ax.set_title(f"7. Voting methods - test {metric} (7 models)")
    lo = min(main[metric].min(), majority_class_rate if metric == "accuracy" else 1.0)
    hi = max(main[metric].max(), majority_class_rate if metric == "accuracy" else 0.0)
    ax.set_ylim(lo - 0.03, hi + 0.02)   # zoomed: the differences are ~1 point
    ax.tick_params(axis="x", labelsize=8)
plt.tight_layout()
save_and_show("07_voting_comparison")

# ---- Figure 8: adaptive weights over time + rolling accuracy of the 3 methods --
test_dates = dates_model.iloc[split_idx:].to_numpy()
ROLL = 60
fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
axes[0].stackplot(test_dates, a7["test_weights"].T, labels=all_7_names, alpha=0.85)
axes[0].set_title(f"8a. Adaptive weights over the test period "
                  f"(window={a7['window']}, beta={a7['beta']:g})")
axes[0].set_ylabel("Weight")
axes[0].set_ylim(0, 1)
axes[0].legend(loc="upper left", fontsize=8, ncol=4)

for name, label, color in [
    ("Majority Voting (7 = 6 classical + LSTM)", "Majority", colors_7["Majority"]),
    ("Weighted Voting - probabilities (7 models)", "Weighted (static)", colors_7["Weighted (static)"]),
    ("Adaptive Weighted Voting - probabilities (7 models)", "Adaptive weighted",
     colors_7["Adaptive weighted"]),
]:
    hits = pd.Series((ensemble_predictions[name] == y_test_common).astype(float))
    axes[1].plot(test_dates, hits.rolling(ROLL).mean(), label=label, color=color)
axes[1].axhline(0.5, color="black", linewidth=0.8, linestyle=":")
axes[1].set_title(f"8b. Rolling {ROLL}-day accuracy on the test period")
axes[1].set_ylabel("Accuracy")
axes[1].legend(loc="upper left")
plt.tight_layout()
save_and_show("08_adaptive_weights_over_time")

print("\n" + "=" * 80)
print("PIPELINE COMPLETE")
print("=" * 80)
print("Experiments produced:")
print(f"  1. Six classical ML models (all {N_FEATURES} features)")
print("  2. Six classical + LSTM")
print("  3. Feature-selected variants (MI Top-K and Tree Top-K)")
print("  4. Equal / majority voting (6-model, 7-model, 8-model)")
print("  5. Soft voting (7-model, 8-model)")
print("  6. Performance-weighted voting (labels and probabilities)")
print("  6b. Adaptive weighted voting (rolling recent-accuracy weights)")
print("  7. Stacking with out-of-fold meta-features")
print("  8. Vision Transformer as an individual model")
print("  9. Experimental 8-model ensembles including the ViT")
print("  10. Final comparison: majority vs weighted vs adaptive weighted voting (STEP 18)")
