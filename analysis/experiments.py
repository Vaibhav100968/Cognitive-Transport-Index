"""
Post-session analysis: load `data/session_data.csv` and SBP artifacts, write figures
to `analysis/figures/`.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(_ROOT, "analysis", "figures")
DATA_DIR = os.path.join(_ROOT, "data")


def _ensure_dirs():
    os.makedirs(FIG_DIR, exist_ok=True)


def fig_session_timeline():
    path = os.path.join(DATA_DIR, "session_data.csv")
    if not os.path.isfile(path):
        print(f"[experiments] Skip timeline: no {path}")
        return
    df = pd.read_csv(path)
    if df.empty or "timestamp" not in df.columns:
        print("[experiments] Skip timeline: empty or missing timestamp")
        return

    df = df.copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    eeg = df[df["event_type"] == "eeg_energy"].copy()
    if eeg.empty or "CTI" not in eeg.columns:
        print("[experiments] Skip timeline: no eeg_energy rows with CTI")
        return

    eeg["CTI"] = pd.to_numeric(eeg["CTI"], errors="coerce")
    eeg = eeg.dropna(subset=["CTI"])

    fig, ax = plt.subplots(figsize=(10, 4))
    for pid, g in eeg.groupby("participant_id"):
        t0 = g["timestamp"].min()
        ax.plot(
            g["timestamp"] - t0,
            g["CTI"],
            marker="o",
            ms=2,
            lw=1,
            label=str(pid),
            alpha=0.8,
        )
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.axhline(2.0, color="red", ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("Time since first logged point (s)")
    ax.set_ylabel("CTI")
    ax.set_title("Session: CTI over time (eeg_energy rows)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "session_cti_timeline.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[experiments] Wrote {out}")


def fig_streaming_vs_offline():
    stream_path = os.path.join(DATA_DIR, "streaming_sbp_results.csv")
    offline_path = os.path.join(_ROOT, "true_sbp_results.csv")
    if not (os.path.isfile(stream_path) and os.path.isfile(offline_path)):
        print("[experiments] Skip scatter: need data/streaming_sbp_results.csv and true_sbp_results.csv")
        return

    off = pd.read_csv(offline_path)
    st = pd.read_csv(stream_path)
    pts = []
    for _, row in off.iterrows():
        p = row["Participant"]
        sp = int(row["Portion"]) - 1
        sub = st[(st["Participant"] == p) & (st["portion"] == sp)]
        if len(sub) == 0:
            continue
        pts.append(
            {
                "offline": float(row["SBP_Energy"]),
                "streaming": float(sub["raw_energy"].mean()),
            }
        )
    if len(pts) < 2:
        print("[experiments] Skip scatter: not enough merged points")
        return

    dfp = pd.DataFrame(pts)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(dfp["offline"], dfp["streaming"], alpha=0.85, edgecolors="k", lw=0.3)
    lim = [
        min(dfp["offline"].min(), dfp["streaming"].min()) * 0.95,
        max(dfp["offline"].max(), dfp["streaming"].max()) * 1.05,
    ]
    ax.plot(lim, lim, "k--", lw=1, alpha=0.5, label="y=x")
    ax.set_xlabel("Offline SBP energy")
    ax.set_ylabel("Mean streaming raw energy (matched portion)")
    ax.set_title("Streaming vs offline SBP")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "streaming_vs_offline_scatter.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[experiments] Wrote {out}")


def fig_session_event_counts():
    path = os.path.join(DATA_DIR, "session_data.csv")
    if not os.path.isfile(path):
        print(f"[experiments] Skip event counts: no {path}")
        return
    df = pd.read_csv(path)
    if df.empty or "event_type" not in df.columns:
        return
    counts = df["event_type"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="k")
    ax.set_ylabel("Count")
    ax.set_title("Session log: events by type")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "session_event_counts.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[experiments] Wrote {out}")


def main():
    sys.path.insert(0, _ROOT)
    _ensure_dirs()
    print("[experiments] Output directory:", FIG_DIR)
    fig_session_timeline()
    fig_streaming_vs_offline()
    fig_session_event_counts()
    print("[experiments] Done.")


if __name__ == "__main__":
    main()
