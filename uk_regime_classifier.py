\#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UK Market Regime Classifier (Rates-Focused)
-------------------------------------------
Classifies next-5D market regime for UK rates into:
- Bull-Flattening (10y down, 2s10s slope down)
- Bull-Steepening (10y down, 2s10s slope up)
- Bear-Flattening (10y up, 2s10s slope down)
- Bear-Steepening (10y up, 2s10s slope up)

Data sources:
- Primary: Bloomberg via blpapi (Terminal or B-PIPE required)
- Fallback: user-provided CSVs

Tickers used (typical Bloomberg mnemonics; verify on your terminal):
- GUKG2 Index   : UK Generic 2Y Gilt Yield (%)
- GUKG5 Index   : UK Generic 5Y Gilt Yield (%)
- GUKG10 Index  : UK Generic 10Y Gilt Yield (%)
- GUKG30 Index  : UK Generic 30Y Gilt Yield (%)
- UKX Index     : FTSE 100
- VFTSE Index   : FTSE 100 Volatility Index

Optional macro (uncomment to use):
- UKRPCJYR Index: UK CPI YoY

Outputs:
- Plots in ./output/
- feature_importance.csv
- classification_report.txt
- confusion_matrix.png

Usage:
- With Bloomberg: `python uk_regime_classifier.py`
- Without Bloomberg: prepare CSVs in ./data/ matching names below and run.
"""

import os
import sys
import json
import math
import time
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Modeling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit

# Plotting
import matplotlib.pyplot as plt

OUTDIR = os.path.join(os.getcwd(), "output")
DATADIR = os.path.join(os.getcwd(), "data")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(DATADIR, exist_ok=True)

START_DATE = "2012-01-01"
END_DATE   = datetime.today().strftime("%Y-%m-%d")

# -----------------------------
# Bloomberg fetch (if available)
# -----------------------------

def have_blpapi():
    try:
        import blpapi  # noqa: F401
        return True
    except Exception:
        return False

def bbg_history(tickers, fields, start_date, end_date, periodicity="DAILY"):
    """
    Minimal Bloomberg blpapi historical data fetcher.
    Returns a wide DataFrame with multiindex columns (ticker, field).
    """
    import blpapi
    from blpapi import SessionOptions, Session, Name
    
    opts = SessionOptions()
    # Desktop API defaults
    opts.setServerHost("localhost")
    opts.setServerPort(8194)
    session = Session(opts)
    if not session.start():
        raise RuntimeError("Failed to start Bloomberg session.")
    if not session.openService("//blp/refdata"):
        raise RuntimeError("Failed to open //blp/refdata service.")
    refdata = session.getService("//blp/refdata")
    
    req = refdata.createRequest("HistoricalDataRequest")
    if isinstance(tickers, str):
        tickers = [tickers]
    if isinstance(fields, str):
        fields = [fields]
    for t in tickers:
        req.append("securities", t)
    for f in fields:
        req.append("fields", f)
    req.set("periodicitySelection", periodicity)
    req.set("startDate", start_date.replace("-", ""))
    req.set("endDate", end_date.replace("-", ""))
    req.set("maxDataPoints", 1000000)
    req.set("adjustmentSplit", False)
    req.set("adjustmentAbnormal", False)
    req.set("adjustmentNormal", False)
    
    cid = session.sendRequest(req)
    data = {}
    done = False
    while not done:
        ev = session.nextEvent(500)
        for msg in ev:
            if msg.messageType() == Name("HistoricalDataResponse"):
                sec = msg.getElementAsString(Name("securityData.security"))
                flds = {}
                for bar in msg.getElement(Name("securityData.fieldData")).values():
                    dt = bar.getElementAsDatetime("date")
                    dts = dt.strftime("%Y-%m-%d")
                    for f in fields:
                        if bar.hasElement(f):
                            flds.setdefault(f, {})[dts] = bar.getElementAsFloat(f)
                for f in fields:
                    ser = pd.Series(flds.get(f, {}), name=(sec, f))
                    data[(sec, f)] = ser
            if ev.eventType() == blpapi.Event.RESPONSE:
                done = True
    session.stop()
    if not data:
        raise RuntimeError("No data returned from Bloomberg.")
    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df

# -----------------------------
# CSV fallback helpers
# -----------------------------

def read_csv_or_none(path, parse_dates=True):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if parse_dates and 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    elif parse_dates and df.columns[0].lower() in ('date', 'time', 'datetime'):
        df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
        df.set_index(df.columns[0], inplace=True)
    return df

def save_df_csv(df, path):
    df_out = df.copy()
    df_out.index.name = "date"
    df_out.to_csv(path)

# -----------------------------
# Data assembly
# -----------------------------

def get_data(start=START_DATE, end=END_DATE, use_bloomberg=True):
    tickers = {
        "GUKG2 Index": "PX_LAST",
        "GUKG5 Index": "PX_LAST",
        "GUKG10 Index": "PX_LAST",
        "GUKG30 Index": "PX_LAST",
        "UKX Index": "PX_LAST",
        "VFTSE Index": "PX_LAST",
        # "UKRPCJYR Index": "PX_LAST",  # optional CPI YoY
    }
    if use_bloomberg and have_blpapi():
        print("Fetching data from Bloomberg via blpapi...")
        df = bbg_history(list(tickers.keys()), list(set(tickers.values())), start, end)
        # Pivot to simple wide columns
        wide = pd.DataFrame(index=df.index.unique())
        for tkr, fld in df.columns:
            series = df[(tkr, fld)].dropna()
            col = tkr.replace(" ", "_")
            wide[col] = series
        # Save cache
        save_df_csv(wide, os.path.join(DATADIR, "uk_market_data_cache.csv"))
        return wide
    else:
        print("Bloomberg not available. Looking for CSV fallback in ./data ...")
        # Expect a CSV with columns:
        # date, GUKG2_Index, GUKG5_Index, GUKG10_Index, GUKG30_Index, UKX_Index, VFTSE_Index
        path = os.path.join(DATADIR, "uk_market_data_cache.csv")
        df = read_csv_or_none(path)
        if df is None:
            raise FileNotFoundError(
                f"CSV fallback not found at {path}. "
                "Provide a CSV with columns: date,GUKG2_Index,GUKG5_Index,GUKG10_Index,"
                "GUKG30_Index,UKX_Index,VFTSE_Index"
            )
        return df

# -----------------------------
# Feature engineering & labels
# -----------------------------

def build_features(df_raw):
    df = df_raw.copy().dropna()
    # Rename columns to simpler names
    cols_map = {
        "GUKG2_Index": "y2",
        "GUKG5_Index": "y5",
        "GUKG10_Index": "y10",
        "GUKG30_Index": "y30",
        "UKX_Index": "ukx",
        "VFTSE_Index": "vftse",
    }
    # Auto-rename if data came from bbg fetcher (spaces replaced with underscores)
    for k in list(cols_map.keys()):
        if k not in df.columns and k.replace("_", " ") in df.columns:
            df[k] = df[k.replace("_", " ")]
    df = df.rename(columns=cols_map)

    # Basic features
    df["slope_2s10s"] = df["y10"] - df["y2"]
    df["slope_5s30s"] = df["y30"] - df["y5"]

    # Daily changes
    for c in ["y2","y5","y10","y30","ukx","vftse","slope_2s10s","slope_5s30s"]:
        if c in df.columns:
            df[f"d_{c}"] = df[c].diff()

    # Momentum signals (5D)
    for c in ["y10","slope_2s10s","ukx","vftse"]:
        if c in df.columns:
            df[f"mom5_{c}"] = df[c].pct_change(5)

    # Realized vol proxy (on yields)
    df["rv20_y10"] = df["y10"].diff().rolling(20).std()

    # Forward (next 5D) changes for labels
    df["fwd5_dy10"] = df["y10"].shift(-5) - df["y10"]
    df["fwd5_dslope"] = (df["slope_2s10s"].shift(-5) - df["slope_2s10s"])

    # Labels (4-class)
    # Thresholds: sign only; you can add dead-bands if desired
    def label_row(row):
        dy = row["fwd5_dy10"]
        ds = row["fwd5_dslope"]
        if pd.isna(dy) or pd.isna(ds):
            return np.nan
        if dy < 0 and ds < 0:
            return "Bull-Flattening"
        elif dy < 0 and ds > 0:
            return "Bull-Steepening"
        elif dy > 0 and ds < 0:
            return "Bear-Flattening"
        elif dy > 0 and ds > 0:
            return "Bear-Steepening"
        else:
            return np.nan

    df["regime_4c"] = df.apply(label_row, axis=1)

    # Feature set (feel free to tweak)
    feature_cols = [
        "y2","y5","y10","y30",
        "slope_2s10s","slope_5s30s",
        "d_y2","d_y5","d_y10","d_y30",
        "mom5_y10","mom5_slope_2s10s",
        "ukx","d_ukx","mom5_ukx",
        "vftse","d_vftse","mom5_vftse",
        "rv20_y10",
    ]
    feats = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(method="ffill").dropna()
    labels = df["regime_4c"].loc[feats.index]
    # Optional: drop days with tiny moves (dead-band) to reduce noise:
    # mask = (df["fwd5_dy10"].abs() > 0.02) | (df["fwd5_dslope"].abs() > 0.02)
    # feats, labels = feats[mask], labels[mask]
    return feats, labels, df

# -----------------------------
# Modeling
# -----------------------------

def train_model(X, y):
    # Time-aware split
    # Use 70% train, 30% test (no shuffling)
    n = len(X)
    split = int(n * 0.7)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Save metrics
    report = classification_report(y_test, y_pred, digits=3)
    with open(os.path.join(OUTDIR, "classification_report.txt"), "w") as f:
        f.write(report)
    print("\\n=== Classification Report (Test) ===\\n")
    print(report)

    # Confusion matrix
    labels_sorted = ["Bull-Flattening","Bull-Steepening","Bear-Flattening","Bear-Steepening"]
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    fig = plt.figure(figsize=(6,5))
    im = plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix (Test)")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(labels_sorted))
    plt.xticks(tick_marks, labels_sorted, rotation=45, ha="right")
    plt.yticks(tick_marks, labels_sorted)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # Feature importances
    fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    fi.to_csv(os.path.join(OUTDIR, "feature_importance.csv"))
    fig2 = plt.figure(figsize=(8,6))
    fi.head(20).iloc[::-1].plot(kind="barh")  # top-20
    plt.title("Top-20 Feature Importances (Random Forest)")
    plt.tight_layout()
    fig2.savefig(os.path.join(OUTDIR, "feature_importance_top20.png"), dpi=150)
    plt.close(fig2)

    return model, (X_train, X_test, y_train, y_test), fi

# -----------------------------
# Visualization helpers
# -----------------------------

def plot_regimes_timeline(df_full, y_test_idx, y_test_true, y_test_pred):
    """
    Plot 10Y yield with shaded actual regimes and markers for predicted regimes on test set.
    """
    ser = df_full["y10"].dropna()
    fig = plt.figure(figsize=(10,5))
    plt.plot(ser.index, ser.values, label="UK 10Y Yield")
    plt.title("UK 10Y Yield with Regimes (Test Window)")
    plt.xlabel("Date"); plt.ylabel("%")
    # Shade test window
    plt.axvspan(y_test_idx[0], y_test_idx[-1], alpha=0.1)
    # Place small markers at test dates colored by predicted/actual agreement
    match = (y_test_true.values == y_test_pred.values)
    for i, dt in enumerate(y_test_idx):
        if match[i]:
            plt.scatter(dt, ser.loc[:dt].iloc[-1], s=10)  # default color
        else:
            plt.scatter(dt, ser.loc[:dt].iloc[-1], s=10)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "timeline_y10_test.png"), dpi=150)
    plt.close(fig)

# -----------------------------
# Main
# -----------------------------

def main():
    df_raw = get_data()
    X, y, df_full = build_features(df_raw)
    # Align indices
    mask = y.notna()
    X, y = X.loc[mask], y.loc[mask]

    model, splits, fi = train_model(X, y)
    X_train, X_test, y_train, y_test = splits
    # Predictions for timeline
    y_pred = pd.Series(model.predict(X_test), index=X_test.index)
    plot_regimes_timeline(df_full, y_pred.index, y_test, y_pred)

    print(f"Artifacts saved to: {OUTDIR}")
    print("- classification_report.txt")
    print("- confusion_matrix.png")
    print("- feature_importance.csv")
    print("- feature_importance_top20.png")
    print("- timeline_y10_test.png")
    print("\\nTip: Inspect feature_importance.csv to refine features & add macro inputs.")

if __name__ == "__main__":
    main()
