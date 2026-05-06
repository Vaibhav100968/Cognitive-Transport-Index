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
    try:
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
        if eeg.empty:
            print("[experiments] Skip timeline: CTI column present but no numeric CTI values")
            return

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
    except Exception as e:
        print(f"[experiments] Warning: timeline figure failed: {e}")


def fig_streaming_vs_offline():
    try:
        stream_path = os.path.join(DATA_DIR, "streaming_sbp_results.csv")
        offline_path = os.path.join(_ROOT, "true_sbp_results.csv")
        if not os.path.isfile(stream_path):
            print(f"[experiments] Skip scatter: no {stream_path}")
            return
        if not os.path.isfile(offline_path):
            print(f"[experiments] Skip scatter: no {offline_path}")
            return

        st = pd.read_csv(stream_path)
        if st.empty:
            print("[experiments] Skip scatter: streaming_sbp_results.csv is empty")
            return
        off = pd.read_csv(offline_path)
        if off.empty:
            print("[experiments] Skip scatter: true_sbp_results.csv is empty")
            return

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
    except Exception as e:
        print(f"[experiments] Warning: scatter figure failed: {e}")


def fig_session_event_counts():
    try:
        path = os.path.join(DATA_DIR, "session_data.csv")
        if not os.path.isfile(path):
            print(f"[experiments] Skip event counts: no {path}")
            return
        df = pd.read_csv(path)
        if df.empty or "event_type" not in df.columns:
            print("[experiments] Skip event counts: empty or missing event_type column")
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
    except Exception as e:
        print(f"[experiments] Warning: event counts figure failed: {e}")


def main():
    sys.path.insert(0, _ROOT)
    _ensure_dirs()
    print("[experiments] Output directory:", FIG_DIR)
    fig_session_timeline()
    fig_streaming_vs_offline()
    fig_session_event_counts()
    print("[experiments] Done.")


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("Files available:")
    for f in [
        "data/streaming_sbp_results.csv",
        "data/session_data.csv",
        "true_sbp_results.csv",
        "cleaned_vegs_data_baseline_z.csv",
    ]:
        path = os.path.join(base, f)
        exists = os.path.exists(path)
        rows = sum(1 for _ in open(path)) - 1 if exists else 0
        print(f"  {'✓' if exists else '✗'} {f} ({rows} rows)")

    main()
