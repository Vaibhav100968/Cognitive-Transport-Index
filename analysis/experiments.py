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
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
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


def fig_easy_vs_hard_group():
    try:
        stream_path = os.path.join(DATA_DIR, "streaming_sbp_results.csv")
        if not os.path.isfile(stream_path):
            print(f"[experiments] Skip easy vs hard: no {stream_path}")
            return
        df = pd.read_csv(stream_path)
        if df.empty:
            print("[experiments] Skip easy vs hard: streaming_sbp_results.csv is empty")
            return

        df = df.copy()
        df["CTI"] = pd.to_numeric(df.get("CTI"), errors="coerce")
        df = df.dropna(subset=["CTI"])
        if df.empty:
            print("[experiments] Skip easy vs hard: no numeric CTI values")
            return

        easy_phases = {"calibration", "easy_test"}
        easy_means = []
        hard_means = []
        for pid in df["Participant"].unique():
            sub = df[df["Participant"] == pid]
            easy = sub[sub["phase"].isin(list(easy_phases))]["CTI"]
            hard = sub[sub["phase"] == "hard_test"]["CTI"]
            if len(easy) == 0 or len(hard) == 0:
                continue
            easy_means.append(float(easy.mean()))
            hard_means.append(float(hard.mean()))

        if len(easy_means) == 0 or len(hard_means) == 0:
            print("[experiments] Skip easy vs hard: no participants with both phases")
            return

        easy_arr = np.array(easy_means, dtype=float)
        hard_arr = np.array(hard_means, dtype=float)

        pooled_std = float(np.sqrt(0.5 * (np.var(easy_arr) + np.var(hard_arr))) + 1e-8)
        cohens_d = float((np.mean(hard_arr) - np.mean(easy_arr)) / pooled_std)

        p_val = None
        stat_name = None
        if len(easy_arr) >= 5 and len(hard_arr) >= 5:
            from scipy.stats import wilcoxon

            stat_name = "wilcoxon"
            _, p_val = wilcoxon(hard_arr, easy_arr)
        else:
            from scipy.stats import mannwhitneyu

            stat_name = "mannwhitneyu"
            _, p_val = mannwhitneyu(hard_arr, easy_arr, alternative="two-sided")

        print(
            "[experiments] Easy CTI mean±std:",
            f"{np.mean(easy_arr):.3f}±{np.std(easy_arr):.3f}",
        )
        print(
            "[experiments] Hard CTI mean±std:",
            f"{np.mean(hard_arr):.3f}±{np.std(hard_arr):.3f}",
        )
        if p_val is not None:
            print(f"[experiments] Stats ({stat_name}): p={p_val:.4g}  d={cohens_d:.3f}")

        try:
            import seaborn as sns

            plot_df = pd.DataFrame(
                {
                    "group": ["easy"] * len(easy_arr) + ["hard"] * len(hard_arr),
                    "CTI_mean": np.concatenate([easy_arr, hard_arr]),
                }
            )
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(data=plot_df, x="group", y="CTI_mean", ax=ax, color="#dddddd")
            sns.stripplot(
                data=plot_df,
                x="group",
                y="CTI_mean",
                ax=ax,
                color="black",
                alpha=0.6,
                jitter=0.15,
            )
            ax.set_xlabel("")
            ax.set_ylabel("Mean CTI per participant")
            title = "Group: easy vs hard (participant means)"
            if p_val is not None:
                title += f" | p={p_val:.3g} d={cohens_d:.2f}"
            ax.set_title(title)
            fig.tight_layout()
            out = os.path.join(FIG_DIR, "fig_easy_vs_hard_group.png")
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f"[experiments] Wrote {out}")
        except Exception as e:
            print(f"[experiments] Warning: easy vs hard plot failed: {e}")
    except Exception as e:
        print(f"[experiments] Warning: easy vs hard group failed: {e}")


def fig_cti_vs_rt():
    try:
        path = os.path.join(DATA_DIR, "session_data.csv")
        if not os.path.isfile(path):
            print(f"[experiments] Skip CTI vs RT: no {path}")
            return
        df = pd.read_csv(path)
        if df.empty:
            print("[experiments] Skip CTI vs RT: session_data.csv is empty")
            return

        choices = df[df["event_type"] == "game_choice"].copy()
        eeg = df[df["event_type"] == "eeg_energy"].copy()
        if choices.empty or eeg.empty:
            print("[experiments] Insufficient data for CTI vs RT")
            return

        choices["timestamp"] = pd.to_numeric(choices.get("timestamp"), errors="coerce")
        choices["reaction_time_ms"] = pd.to_numeric(
            choices.get("reaction_time_ms"), errors="coerce"
        )
        choices = choices.dropna(subset=["timestamp", "reaction_time_ms"])

        eeg["timestamp"] = pd.to_numeric(eeg.get("timestamp"), errors="coerce")
        eeg["CTI"] = pd.to_numeric(eeg.get("CTI"), errors="coerce")
        eeg = eeg.dropna(subset=["timestamp", "CTI"])

        if choices.empty or eeg.empty:
            print("[experiments] Insufficient data for CTI vs RT")
            return

        eeg = eeg.sort_values("timestamp")
        eeg_t = eeg["timestamp"].to_numpy(dtype=float)
        eeg_cti = eeg["CTI"].to_numpy(dtype=float)

        def closest_cti(ts: float, max_diff_sec: float = 30.0):
            idx = int(np.searchsorted(eeg_t, ts))
            best_cti = None
            best_diff = max_diff_sec
            for j in (idx - 1, idx):
                if 0 <= j < len(eeg_t):
                    diff = abs(eeg_t[j] - ts)
                    if diff <= best_diff:
                        best_diff = diff
                        best_cti = float(eeg_cti[j])
            return best_cti

        pairs = []
        for _, row in choices.iterrows():
            ts = float(row["timestamp"])
            cti = closest_cti(ts, max_diff_sec=30.0)
            pairs.append(
                {
                    "reaction_time_ms": float(row["reaction_time_ms"]),
                    "CTI": cti,
                    "difficulty": row.get("difficulty"),
                    "experiment_phase": row.get("experiment_phase"),
                    "scenario_index": row.get("scenario_index"),
                    "choice_timestamp": ts,
                }
            )

        pairs_df = pd.DataFrame(pairs)
        pairs_df["CTI"] = pd.to_numeric(pairs_df.get("CTI"), errors="coerce")
        pairs_df = pairs_df.dropna(subset=["CTI"])
        if len(pairs_df) < 3:
            print("[experiments] Warning: Insufficient matched pairs for CTI vs RT")
            return

        from scipy.stats import pearsonr

        r, pval = pearsonr(pairs_df["reaction_time_ms"], pairs_df["CTI"])
        print(
            f"[experiments] CTI vs RT pairs: n={len(pairs_df)} r={r:.4f} p={pval:.4g}"
        )

        fig, ax = plt.subplots(figsize=(6, 4.5))
        colors = []
        for d in pairs_df["difficulty"].astype(str).str.lower().fillna(""):
            if d == "easy":
                colors.append("#1f77b4")
            elif d == "hard":
                colors.append("#d62728")
            else:
                colors.append("#666666")

        x = pairs_df["reaction_time_ms"].to_numpy(dtype=float)
        y = pairs_df["CTI"].to_numpy(dtype=float)
        ax.scatter(x, y, c=colors, alpha=0.8, edgecolors="k", linewidths=0.2)

        try:
            m, b = np.polyfit(x, y, 1)
            xs = np.linspace(float(np.min(x)), float(np.max(x)), 100)
            ax.plot(xs, m * xs + b, color="black", linewidth=1.5, alpha=0.7)
        except Exception:
            pass

        ax.set_xlabel("Reaction time (ms)")
        ax.set_ylabel("CTI")
        ax.set_title("CTI vs reaction time (timestamp-matched)")
        ax.text(
            0.02,
            0.98,
            f"n={len(pairs_df)}\nr={r:.2f}  p={pval:.3g}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
        )
        fig.tight_layout()
        out = os.path.join(FIG_DIR, "fig_cti_vs_rt.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"[experiments] Wrote {out}")
    except Exception as e:
        print(f"[experiments] Warning: CTI vs RT failed: {e}")


def fig_model_comparison():
    try:
        csv_path = os.path.join(_ROOT, "cleaned_vegs_data_baseline_z.csv")
        stream_path = os.path.join(DATA_DIR, "streaming_sbp_results.csv")

        if not os.path.isfile(csv_path):
            print(f"[experiments] Skip model comparison: no {csv_path}")
            return
        if not os.path.isfile(stream_path):
            print(f"[experiments] Skip model comparison: no {stream_path}")
            return

        features = [
            "Theta",
            "Alpha",
            "BetaL",
            "BetaH",
            "Gamma",
            "Arousal",
            "Valence",
            "Engagement",
        ]

        df = pd.read_csv(csv_path)
        if df.empty:
            print("[experiments] Skip model comparison: VEGS csv empty")
            return

        def make_windows(data, window_size=50, step=10):
            wins = []
            arr = data[features].values
            for i in range(0, len(arr) - window_size, step):
                wins.append(arr[i : i + window_size])
            return wins

        easy_data = df[df["portion"].isin([0, 1])]
        hard_data = df[df["portion"].isin([2, 3])]
        windows_easy = make_windows(easy_data)
        windows_hard = make_windows(hard_data)
        print(f"[experiments] Easy windows: {len(windows_easy)}, Hard windows: {len(windows_hard)}")

        from core.baselines import run_all_baselines

        base_results = run_all_baselines(windows_easy, windows_hard)

        st = pd.read_csv(stream_path)
        if st.empty:
            print("[experiments] Skip model comparison: streaming results empty")
            return
        easy_scores = st[st["phase"].isin(["calibration", "easy_test"])]["raw_energy"].dropna()
        hard_scores = st[st["phase"] == "hard_test"]["raw_energy"].dropna()
        if len(easy_scores) == 0 or len(hard_scores) == 0:
            print("[experiments] Skip model comparison: insufficient SBP phase data")
            return

        from sklearn.metrics import roc_auc_score

        y_true = [0] * len(easy_scores) + [1] * len(hard_scores)
        y_score = list(easy_scores) + list(hard_scores)
        try:
            sbp_auc = float(roc_auc_score(y_true, y_score))
        except Exception:
            sbp_auc = 0.5

        results = {k: v.get("auc", 0.5) for k, v in base_results.items()}
        results["SBP (CTI)"] = sbp_auc

        auc_items = sorted(results.items(), key=lambda x: x[1], reverse=True)
        print("[experiments] AUC table:")
        for name, auc in auc_items:
            print(f"  {name:25s} {auc:.3f}")

        names = [n for n, _ in auc_items]
        aucs = [a for _, a in auc_items]
        colors = ["#1f77b4" if n == "SBP (CTI)" else "#9e9e9e" for n in names]

        fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(names) + 1.5)))
        y = np.arange(len(names))
        bars = ax.barh(y, aucs, color=colors, edgecolor="black", linewidth=0.3)
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.axvline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_xlabel("AUC (easy vs hard)")
        ax.set_title("Model comparison (AUC)")
        for b, auc in zip(bars, aucs):
            ax.text(
                b.get_width() + 0.01,
                b.get_y() + b.get_height() / 2,
                f"{auc:.3f}",
                va="center",
                fontsize=9,
            )
        ax.set_xlim(0.0, min(1.0, max(0.8, max(aucs) + 0.12)))
        fig.tight_layout()
        out = os.path.join(FIG_DIR, "fig_model_comparison.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"[experiments] Wrote {out}")
    except Exception as e:
        print(f"[experiments] Warning: model comparison failed: {e}")


def main():
    _ensure_dirs()
    print("[experiments] Output directory:", FIG_DIR)
    fig_session_timeline()
    fig_streaming_vs_offline()
    fig_session_event_counts()
    fig_easy_vs_hard_group()
    fig_cti_vs_rt()
    fig_model_comparison()
    print("[experiments] Done.")


if __name__ == "__main__":
    import os

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
