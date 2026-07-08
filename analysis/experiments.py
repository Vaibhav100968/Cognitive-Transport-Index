"""
Post-session analysis: load `data/session_data.csv` and SBP artifacts, write figures
to `analysis/figures/`.
"""
import os
import sys

import matplotlib
matplotlib.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
FIG_DIR = os.path.join(_ROOT, "analysis", "figures")
DATA_DIR = os.path.join(_ROOT, "data")

PHASE_COLORS = {
    "calibration": "#aec6cf",
    "easy_test":   "#90ee90",
    "hard_test":   "#ffb3b3",
}
PHASE_LABELS = {
    "calibration": "Calibration",
    "easy_test":   "Easy",
    "hard_test":   "Hard",
}

UNSUPERVISED = {
    "Spectral Band Ratio",
    "Engagement Index",
    "Differential Entropy",
    "RMS Channel Energy",
    "Kalman Filter",
    "SBP (CTI)",
}


def _ensure_dirs():
    os.makedirs(FIG_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Figure 1 – Real-time CTI timeline with phase shading + smoothed overlay
# ---------------------------------------------------------------------------
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

        fig, ax = plt.subplots(figsize=(11, 4))

        for pid, g in eeg.groupby("participant_id"):
            g = g.sort_values("timestamp")
            t0 = g["timestamp"].min()
            t = (g["timestamp"] - t0).to_numpy(dtype=float)
            cti = g["CTI"].to_numpy(dtype=float)

            # Phase background shading
            if "experiment_phase" in g.columns:
                g2 = g.copy()
                g2["t"] = t
                phase_changes = g2[g2["experiment_phase"] != g2["experiment_phase"].shift()].copy()
                phases_used = set()
                for i, (_, pr) in enumerate(phase_changes.iterrows()):
                    ph = str(pr["experiment_phase"])
                    x0 = float(pr["t"])
                    # find end of this phase block
                    next_changes = phase_changes[phase_changes["t"] > x0]
                    x1 = float(next_changes["t"].iloc[0]) if len(next_changes) > 0 else float(t[-1])
                    fc = PHASE_COLORS.get(ph, "#eeeeee")
                    label = PHASE_LABELS.get(ph, ph) if ph not in phases_used else None
                    ax.axvspan(x0, x1, color=fc, alpha=0.35, label=label)
                    phases_used.add(ph)

            # Raw trace (thin, transparent)
            ax.plot(t, cti, color="#1f77b4", lw=0.8, alpha=0.4)
            # Smoothed overlay
            if len(cti) >= 5:
                smooth = pd.Series(cti).rolling(window=7, center=True, min_periods=1).mean().to_numpy()
                ax.plot(t, smooth, color="#1f77b4", lw=2.0, alpha=0.9, label="CTI (smoothed)")
            ax.scatter(t, cti, color="#1f77b4", s=8, alpha=0.5, zorder=3)

        ax.axhline(0, color="gray", ls="--", lw=1, alpha=0.7, label="Baseline (CTI=0)")
        ax.axhline(2.0, color="red", ls="--", lw=1, alpha=0.6, label="High load (CTI=2)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("CTI")
        ax.set_title("Real-Time Cognitive Transport Index During Game Session")
        # Deduplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        seen = {}
        for h, l in zip(handles, labels):
            if l not in seen:
                seen[l] = h
        ax.legend(seen.values(), seen.keys(), loc="upper right", fontsize=9)
        fig.tight_layout()
        out = os.path.join(FIG_DIR, "session_cti_timeline.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[experiments] Wrote {out}")
    except Exception as e:
        print(f"[experiments] Warning: timeline figure failed: {e}")
        import traceback; traceback.print_exc()


# ---------------------------------------------------------------------------
# Figure 2 – Streaming vs offline SBP with regression line + r annotation
# ---------------------------------------------------------------------------
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
        off = pd.read_csv(offline_path)
        if st.empty or off.empty:
            print("[experiments] Skip scatter: empty data files")
            return

        pts = []
        for _, row in off.iterrows():
            p = row["Participant"]
            sp = int(row["Portion"]) - 1
            sub = st[(st["Participant"] == p) & (st["portion"] == sp)]
            if len(sub) == 0:
                continue
            pts.append({
                "offline": float(row["SBP_Energy"]),
                "streaming": float(sub["raw_energy"].mean()),
            })
        if len(pts) < 2:
            print("[experiments] Skip scatter: not enough merged points")
            return

        dfp = pd.DataFrame(pts)
        from scipy.stats import pearsonr
        r, pval = pearsonr(dfp["offline"], dfp["streaming"])

        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.scatter(dfp["offline"], dfp["streaming"], alpha=0.85, edgecolors="k",
                   lw=0.5, s=55, color="#1f77b4", zorder=3)

        # y=x identity line
        lim = [
            min(dfp["offline"].min(), dfp["streaming"].min()) * 0.95,
            max(dfp["offline"].max(), dfp["streaming"].max()) * 1.05,
        ]
        ax.plot(lim, lim, "k--", lw=1, alpha=0.4, label="Identity (y=x)")

        # regression line
        try:
            m, b = np.polyfit(dfp["offline"], dfp["streaming"], 1)
            xs = np.linspace(float(dfp["offline"].min()), float(dfp["offline"].max()), 100)
            ax.plot(xs, m * xs + b, color="#d62728", lw=1.5, alpha=0.8, label="Regression fit")
        except Exception:
            pass

        ax.set_xlabel("Offline SBP Energy")
        ax.set_ylabel("Mean Streaming SBP Energy")
        ax.set_title("Streaming vs. Offline SBP Energy")
        ax.text(0.97, 0.05,
                f"r = {r:.2f}\np = {pval:.3f}\nn = {len(dfp)}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))
        ax.legend(fontsize=9)
        fig.tight_layout()
        out = os.path.join(FIG_DIR, "streaming_vs_offline_scatter.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[experiments] Wrote {out}")
    except Exception as e:
        print(f"[experiments] Warning: scatter figure failed: {e}")


# ---------------------------------------------------------------------------
# Figure 3 – Easy vs Hard CTI boxplot (window-level, not participant means)
# ---------------------------------------------------------------------------
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

        easy_arr = df[df["phase"].isin(["calibration", "easy_test"])]["CTI"].to_numpy(dtype=float)
        hard_arr = df[df["phase"] == "hard_test"]["CTI"].to_numpy(dtype=float)

        if len(easy_arr) == 0 or len(hard_arr) == 0:
            print("[experiments] Skip easy vs hard: missing phase data")
            return

        from scipy.stats import mannwhitneyu
        _, p_val = mannwhitneyu(hard_arr, easy_arr, alternative="two-sided")

        pooled_std = float(np.sqrt(0.5 * (np.var(easy_arr) + np.var(hard_arr))) + 1e-8)
        cohens_d = float((np.mean(hard_arr) - np.mean(easy_arr)) / pooled_std)
        n_easy, n_hard = len(easy_arr), len(hard_arr)

        print(f"[experiments] Easy CTI: {np.mean(easy_arr):.3f}±{np.std(easy_arr):.3f} (n={n_easy} windows)")
        print(f"[experiments] Hard CTI: {np.mean(hard_arr):.3f}±{np.std(hard_arr):.3f} (n={n_hard} windows)")
        print(f"[experiments] Mann-Whitney U: p={p_val:.4g}  d={cohens_d:.3f}")

        try:
            import seaborn as sns

            plot_df = pd.DataFrame({
                "Condition": ["Easy"] * n_easy + ["Hard"] * n_hard,
                "CTI": np.concatenate([easy_arr, hard_arr]),
            })

            fig, ax = plt.subplots(figsize=(5, 5))
            palette = {"Easy": "#4393c3", "Hard": "#d6604d"}
            sns.boxplot(data=plot_df, x="Condition", y="CTI", ax=ax,
                        hue="Condition", palette=palette, width=0.5,
                        linewidth=1.5, legend=False,
                        flierprops=dict(marker="o", ms=2, alpha=0.2))

            y_max = max(np.percentile(easy_arr, 95), np.percentile(hard_arr, 95))
            y_range = y_max - min(np.percentile(easy_arr, 5), np.percentile(hard_arr, 5))
            y_top = y_max + 0.12 * y_range
            ax.plot([0, 0, 1, 1], [y_top - 0.04*y_range, y_top, y_top, y_top - 0.04*y_range],
                    lw=1.2, color="black")
            if p_val < 0.001:
                sig_text = "*** p<0.001"
            elif p_val < 0.01:
                sig_text = f"** p={p_val:.3f}"
            elif p_val < 0.05:
                sig_text = f"* p={p_val:.3f}"
            else:
                sig_text = f"ns  p={p_val:.3f}"
            ax.text(0.5, y_top + 0.01*y_range, sig_text,
                    ha="center", va="bottom", fontsize=11, color="black")

            ax.set_xlabel("")
            ax.set_ylabel("CTI")
            ax.set_title(
                f"CTI: Easy vs. Hard Cognitive Load"
                f" (N={n_easy+n_hard} windows, Cohen's d={cohens_d:.2f})"
            )
            fig.tight_layout()
            out = os.path.join(FIG_DIR, "fig_easy_vs_hard_group.png")
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[experiments] Wrote {out}")
        except Exception as e:
            print(f"[experiments] Warning: easy vs hard plot failed: {e}")
            import traceback; traceback.print_exc()
    except Exception as e:
        print(f"[experiments] Warning: easy vs hard group failed: {e}")


# ---------------------------------------------------------------------------
# Figure 4 – CTI vs Reaction Time (coloured by difficulty, with legend)
# ---------------------------------------------------------------------------
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
        choices["reaction_time_ms"] = pd.to_numeric(choices.get("reaction_time_ms"), errors="coerce")
        choices = choices.dropna(subset=["timestamp", "reaction_time_ms"])

        eeg["timestamp"] = pd.to_numeric(eeg.get("timestamp"), errors="coerce")
        eeg["CTI"] = pd.to_numeric(eeg.get("CTI"), errors="coerce")
        eeg = eeg.dropna(subset=["timestamp", "CTI"]).sort_values("timestamp")

        if choices.empty or eeg.empty:
            print("[experiments] Insufficient data for CTI vs RT")
            return

        eeg_t = eeg["timestamp"].to_numpy(dtype=float)
        eeg_cti = eeg["CTI"].to_numpy(dtype=float)

        def closest_cti(ts: float, max_diff_sec: float = 30.0):
            idx = int(np.searchsorted(eeg_t, ts))
            best_cti, best_diff = None, max_diff_sec
            for j in (idx - 1, idx):
                if 0 <= j < len(eeg_t):
                    diff = abs(eeg_t[j] - ts)
                    if diff <= best_diff:
                        best_diff, best_cti = diff, float(eeg_cti[j])
            return best_cti

        pairs = []
        for _, row in choices.iterrows():
            ts = float(row["timestamp"])
            cti = closest_cti(ts, max_diff_sec=30.0)
            pairs.append({
                "reaction_time_ms": float(row["reaction_time_ms"]),
                "CTI": cti,
                "difficulty": row.get("difficulty"),
            })

        pairs_df = pd.DataFrame(pairs)
        pairs_df["CTI"] = pd.to_numeric(pairs_df["CTI"], errors="coerce")
        pairs_df = pairs_df.dropna(subset=["CTI"])
        if len(pairs_df) < 3:
            print("[experiments] Warning: Insufficient matched pairs for CTI vs RT")
            return

        from scipy.stats import pearsonr
        r, pval = pearsonr(pairs_df["reaction_time_ms"], pairs_df["CTI"])
        print(f"[experiments] CTI vs RT: n={len(pairs_df)} r={r:.4f} p={pval:.4g}")

        diff_col = pairs_df["difficulty"].astype(str).str.lower().fillna("")
        color_map = {"easy": "#1f77b4", "hard": "#d62728"}
        colors = [color_map.get(d, "#888888") for d in diff_col]

        fig, ax = plt.subplots(figsize=(6, 5))
        x = pairs_df["reaction_time_ms"].to_numpy(dtype=float)
        y = pairs_df["CTI"].to_numpy(dtype=float)
        ax.scatter(x, y, c=colors, alpha=0.85, edgecolors="k", linewidths=0.4, s=60, zorder=3)

        # Regression line
        try:
            m, b = np.polyfit(x, y, 1)
            xs = np.linspace(float(np.min(x)), float(np.max(x)), 100)
            ax.plot(xs, m * xs + b, color="black", lw=1.5, alpha=0.7, label="_nolegend_")
        except Exception:
            pass

        # Stats annotation
        ax.text(0.03, 0.97, f"n={len(pairs_df)}\nr={r:.2f}  p={pval:.3f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))

        # Legend for difficulty colours
        legend_handles = [
            mpatches.Patch(facecolor="#1f77b4", edgecolor="k", label="Easy"),
            mpatches.Patch(facecolor="#d62728", edgecolor="k", label="Hard"),
        ]
        ax.legend(handles=legend_handles, title="Difficulty", loc="lower right", fontsize=10)

        ax.set_xlabel("Reaction Time (ms)")
        ax.set_ylabel("CTI")
        ax.set_title("CTI vs. Reaction Time by Cognitive Load")
        fig.tight_layout()
        out = os.path.join(FIG_DIR, "fig_cti_vs_rt.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[experiments] Wrote {out}")
    except Exception as e:
        print(f"[experiments] Warning: CTI vs RT failed: {e}")
        import traceback; traceback.print_exc()


# ---------------------------------------------------------------------------
# Figure 5 – Model comparison AUC bar chart (supervised vs unsupervised grouped)
# ---------------------------------------------------------------------------
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

        features = ["Theta", "Alpha", "BetaL", "BetaH", "Gamma",
                    "Arousal", "Valence", "Engagement"]

        df = pd.read_csv(csv_path)
        if df.empty:
            print("[experiments] Skip model comparison: VEGS csv empty")
            return

        def make_windows(data, window_size=50, step=10):
            arr = data[features].values
            return [arr[i: i + window_size] for i in range(0, len(arr) - window_size, step)]

        easy_data = df[df["portion"].isin([0, 1])]
        hard_data = df[df["portion"].isin([2, 3])]
        windows_easy = make_windows(easy_data)
        windows_hard = make_windows(hard_data)
        print(f"[experiments] Easy windows: {len(windows_easy)}, Hard: {len(windows_hard)}")

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

        # Sort: supervised first (by AUC), then unsupervised (by AUC)
        supervised = {k: v for k, v in results.items() if k not in UNSUPERVISED}
        unsupervised = {k: v for k, v in results.items() if k in UNSUPERVISED}
        supervised_sorted = sorted(supervised.items(), key=lambda x: x[1], reverse=True)
        unsupervised_sorted = sorted(unsupervised.items(), key=lambda x: x[1], reverse=True)
        auc_items = supervised_sorted + unsupervised_sorted

        print("[experiments] AUC table:")
        for name, auc in auc_items:
            tag = "" if name not in UNSUPERVISED else " [unsup]"
            print(f"  {name:25s} {auc:.3f}{tag}")

        names = [n for n, _ in auc_items]
        aucs = [a for _, a in auc_items]

        # Colours: SBP=blue, other unsupervised=teal, supervised=grey
        def bar_color(n):
            if n == "SBP (CTI)":
                return "#1f77b4"
            if n in UNSUPERVISED:
                return "#6baed6"
            return "#bdbdbd"

        colors = [bar_color(n) for n in names]

        n_sup = len(supervised_sorted)
        n_uns = len(unsupervised_sorted)
        fig, ax = plt.subplots(figsize=(9, max(5, 0.42 * len(names) + 1.5)))
        y = np.arange(len(names))
        bars = ax.barh(y, aucs, color=colors, edgecolor="black", linewidth=0.4)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=12)
        ax.invert_yaxis()
        ax.axvline(0.5, color="black", linestyle="--", linewidth=1.2, alpha=0.6)

        # Group bracket annotations
        if n_sup > 0:
            y_mid_sup = (0 + n_sup - 1) / 2
            ax.annotate("Supervised", xy=(-0.28, y_mid_sup), xycoords=("axes fraction", "data"),
                        ha="center", va="center", fontsize=10, rotation=90,
                        fontweight="bold", color="#555555")
            ax.axhline(n_sup - 0.5, color="#aaaaaa", lw=0.8, ls="--")
        if n_uns > 0:
            y_mid_uns = n_sup + (n_uns - 1) / 2
            ax.annotate("Unsupervised", xy=(-0.28, y_mid_uns), xycoords=("axes fraction", "data"),
                        ha="center", va="center", fontsize=10, rotation=90,
                        fontweight="bold", color="#1f77b4")

        # AUC value labels
        for b, auc in zip(bars, aucs):
            ax.text(b.get_width() + 0.005, b.get_y() + b.get_height() / 2,
                    f"{auc:.3f}", va="center", fontsize=10)

        ax.set_xlabel("AUC (Easy vs. Hard Cognitive Load)")
        ax.set_title("Cognitive Load Classification: AUC Comparison\n"
                     "(VEGS Dataset, N=10 Participants)")
        ax.set_xlim(0.0, min(1.0, max(0.8, max(aucs) + 0.12)))

        # Legend
        legend_handles = [
            mpatches.Patch(facecolor="#bdbdbd", edgecolor="k", label="Supervised baselines"),
            mpatches.Patch(facecolor="#6baed6", edgecolor="k", label="Unsupervised baselines"),
            mpatches.Patch(facecolor="#1f77b4", edgecolor="k", label="SBP (CTI) – proposed"),
        ]
        ax.legend(handles=legend_handles, loc="lower right", fontsize=10)

        # Leave space on the left for group labels
        plt.subplots_adjust(left=0.32)
        out = os.path.join(FIG_DIR, "fig_model_comparison.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[experiments] Wrote {out}")
    except Exception as e:
        print(f"[experiments] Warning: model comparison failed: {e}")
        import traceback; traceback.print_exc()


def main():
    _ensure_dirs()
    print("[experiments] Output directory:", FIG_DIR)
    fig_session_timeline()
    fig_streaming_vs_offline()
    fig_easy_vs_hard_group()
    fig_cti_vs_rt()
    fig_model_comparison()
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
