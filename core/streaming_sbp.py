import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import collections

import numpy as np
import pandas as pd
import torch

from core.SBP import ScoreNet, euler_maruyama_sample, train_scores, device


class StreamingSBP:
    """Online Schrödinger-bridge energy estimator with CTI normalization.

    Holds resting prior ``X0``, a sliding feature buffer, and participant-specific
    ``(μ_easy, σ_easy)`` stats. See ``docs/ARCHITECTURE.md`` for the full lifecycle.
    """

    def __init__(self, baseline_rows, features, window_size=50, step_size=10):
        self.features = list(features)
        self.window_size = int(window_size)
        self.step_size = int(step_size)
        # Matches core.SBP: CUDA if available, else MPS, else CPU
        self.device = device

        baseline_rows = np.asarray(baseline_rows, dtype=np.float32)
        if baseline_rows.ndim != 2 or baseline_rows.shape[1] != len(self.features):
            raise ValueError(
                f"baseline_rows must be (N, {len(self.features)}), got {baseline_rows.shape}"
            )

        self.X0 = torch.as_tensor(baseline_rows, dtype=torch.float32, device=self.device)
        self.buffer = collections.deque(maxlen=self.window_size)
        self.mu_easy = None
        self.sigma_easy = None
        self.energy_history = []
        self.phase = "calibration"
        self.window_count = 0

    def set_phase(self, phase):
        """Advance session phase; lock CTI stats when leaving calibration."""
        if (
            self.phase == "calibration"
            and phase != "calibration"
            and len(self.energy_history) > 0
        ):
            # Participant-specific z-score baseline for CTI(t) = (E − μ) / σ
            self.mu_easy = float(np.mean(self.energy_history))
            self.sigma_easy = float(np.std(self.energy_history)) + 1e-8
            print(
                f"[StreamingSBP] Calibration complete: mu={self.mu_easy:.6f} "
                f"sigma={self.sigma_easy:.6f} n={len(self.energy_history)}"
            )
        self.phase = phase

    def add_sample(self, feature_vector):
        v = np.asarray(feature_vector, dtype=np.float32).reshape(-1)
        if v.shape[0] != len(self.features):
            raise ValueError(
                f"feature_vector must have length {len(self.features)}, got {v.shape[0]}"
            )
        self.buffer.append(v)

    def compute_energy(self):
        """Solve a short SBP on the current window; return raw energy and CTI.

        Returns ``None`` until the buffer holds ``window_size`` samples. After a
        successful window, the buffer advances by ``step_size`` (overlap).
        """
        if len(self.buffer) < self.window_size:
            return None

        X1 = torch.as_tensor(np.array(list(self.buffer)), dtype=torch.float32, device=self.device)
        # Match endpoint batch sizes for the bridge (required by train_scores).
        n = min(self.X0.shape[0], self.window_size)
        X0 = self.X0[:n]
        X1 = X1[:n]

        score_f = ScoreNet(len(self.features)).to(self.device)
        score_b = ScoreNet(len(self.features)).to(self.device)
        # Short epoch budget keeps median end-to-end latency ~sub-second on CPU/MPS.
        train_scores(X0, X1, score_f, score_b, epochs=20, batch_size=32)

        traj = euler_maruyama_sample(X0, score_f, steps=100)

        # Monte-Carlo-style mean squared drift along the sampled bridge.
        energy = 0.0
        for i in range(100):
            t = torch.full((traj.shape[1], 1), i / 100.0, device=self.device)
            drift = score_f(
                torch.as_tensor(traj[i], dtype=torch.float32, device=self.device),
                t,
            )
            energy += (drift.norm(dim=1) ** 2).mean().item() / 100.0

        if self.phase == "calibration":
            self.energy_history.append(energy)
            CTI = None
        elif self.mu_easy is not None:
            CTI = (energy - self.mu_easy) / self.sigma_easy
        else:
            CTI = None

        # Overlapping window: drop the oldest ``step_size`` samples, keep the rest.
        buf_list = list(self.buffer)
        self.buffer.clear()
        for item in buf_list[self.step_size :]:
            self.buffer.append(item)

        self.window_count += 1
        return {
            "raw_energy": energy,
            "CTI": CTI,
            "phase": self.phase,
            "window_count": self.window_count,
            "n_calibration_samples": len(self.energy_history),
        }


def simulate_stream(
    csv_path,
    features,
    window_size=50,
    step_size=10,
    output_csv="data/streaming_sbp_results.csv",
):
    df = pd.read_csv(csv_path)
    results = []

    for p in df["Participant"].unique():
        portion0 = df[(df["Participant"] == p) & (df["portion"] == 0)][features].values
        if len(portion0) < 10:
            continue

        sbp = StreamingSBP(portion0, features, window_size, step_size)
        phase_map = {0: "calibration", 1: "easy_test", 2: "hard_test", 3: "hard_test"}

        for portion in [0, 1, 2, 3]:
            rows = df[(df["Participant"] == p) & (df["portion"] == portion)][features].values
            new_phase = phase_map[portion]
            if portion > 0:
                sbp.set_phase(new_phase)

            for i, row in enumerate(rows):
                sbp.add_sample(row)
                if len(sbp.buffer) >= window_size and i % step_size == 0:
                    result = sbp.compute_energy()
                    if result is not None:
                        results.append(
                            {
                                "Participant": p,
                                "portion": portion,
                                "phase": result["phase"],
                                "raw_energy": result["raw_energy"],
                                "CTI": result["CTI"],
                                "window_count": result["window_count"],
                            }
                        )
                        if result["window_count"] % 5 == 0:
                            cti_str = (
                                f"{result['CTI']:.3f}"
                                if result["CTI"] is not None
                                else "N/A"
                            )
                            print(
                                f"  [{p}] portion={portion} window={result['window_count']} "
                                f"CTI={cti_str}"
                            )

    out_dir = os.path.dirname(os.path.abspath(output_csv))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"Saved {len(results)} rows to {output_csv}")
    return results


if __name__ == "__main__":
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
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results = simulate_stream(
        os.path.join(base, "cleaned_vegs_data_baseline_z.csv"),
        features,
        output_csv=os.path.join(base, "data", "streaming_sbp_results.csv"),
    )

    offline = pd.read_csv(os.path.join(base, "true_sbp_results.csv"))
    streaming = pd.DataFrame(results)
    merged = []
    for _, row in offline.iterrows():
        p = row["Participant"]
        streaming_portion = row["Portion"] - 1
        sub = streaming[
            (streaming["Participant"] == p) & (streaming["portion"] == streaming_portion)
        ]
        if len(sub) == 0:
            continue
        mean_e = sub["raw_energy"].mean()
        if not np.isnan(mean_e):
            merged.append({"offline": row["SBP_Energy"], "streaming": mean_e})

    if len(merged) > 1:
        from scipy.stats import pearsonr

        df_m = pd.DataFrame(merged)
        r, pval = pearsonr(df_m["offline"], df_m["streaming"])
        mae = np.mean(np.abs(df_m["offline"] - df_m["streaming"]))
        print("\nValidation: Streaming vs Offline")
        print(f"  Pearson r = {r:.4f}  p = {pval:.4f}")
        print(f"  MAE       = {mae:.8f}")
        print(f"  N pairs   = {len(merged)}")
