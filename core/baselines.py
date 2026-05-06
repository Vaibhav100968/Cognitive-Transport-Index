"""
Twelve baseline cognitive load estimators vs Schrödinger-bridge transport.
Each model consumes a window of shape [50, 8] (Theta, Alpha, BetaL, BetaH,
Gamma, Arousal, Valence, Engagement) and outputs one float (higher ≈ more load).
"""
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ---- CLASSICAL NEUROSCIENCE (no training) ----


def spectral_band_ratio(window: np.ndarray) -> float:
    """(Theta + Alpha) / (BetaL + BetaH + Gamma), window means."""
    num = np.mean(window[:, 0] + window[:, 1])
    den = np.mean(window[:, 2] + window[:, 3] + window[:, 4]) + 1e-8
    return float(num / den)


def engagement_index(window: np.ndarray) -> float:
    """BetaL / (Alpha + Theta), window means."""
    return float(
        np.mean(window[:, 2]) / (np.mean(window[:, 1] + window[:, 0]) + 1e-8)
    )


def differential_entropy(window: np.ndarray) -> float:
    variances = np.var(window, axis=0) + 1e-8
    de = 0.5 * np.log(2 * np.pi * np.e * variances)
    return float(np.mean(de))


def rms_channel_energy(window: np.ndarray) -> float:
    """Root-mean-square across all channels (global activation proxy)."""
    return float(np.sqrt(np.mean(window**2)))


# ---- CLASSICAL ML ----


class SVMBaseline:
    def __init__(self):
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC

        self.model = SVC(kernel="rbf", probability=True, C=1.0)
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, windows_easy, windows_hard):
        X = np.array([w.mean(axis=0) for w in windows_easy + windows_hard])
        y = [0] * len(windows_easy) + [1] * len(windows_hard)
        self.scaler.fit(X)
        self.model.fit(self.scaler.transform(X), y)
        self.fitted = True

    def predict(self, window):
        if not self.fitted:
            return 0.5
        x = window.mean(axis=0, keepdims=True)
        return float(self.model.predict_proba(self.scaler.transform(x))[0][1])


class HMMBaseline:
    def __init__(self, n_states=2):
        try:
            from hmmlearn.hmm import GaussianHMM

            self.model = GaussianHMM(
                n_components=n_states,
                covariance_type="diag",
                n_iter=50,
                random_state=42,
            )
            self.available = True
        except ImportError:
            self.model = None
            self.available = False
        self.fitted = False

    def fit(self, windows_easy, windows_hard):
        if not self.available:
            return
        X = np.vstack(windows_easy + windows_hard)
        lengths = [len(w) for w in windows_easy + windows_hard]
        try:
            self.model.fit(X, lengths)
            self.fitted = True
        except Exception as e:
            print(f"HMM fit error: {e}")

    def predict(self, window):
        if not self.fitted or not self.available:
            return 0.0
        try:
            states = self.model.predict(window)
            return float(
                np.mean(states) / max(1, self.model.n_components - 1)
            )
        except Exception:
            return 0.0


class LDABaseline:
    def __init__(self):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        self.model = LinearDiscriminantAnalysis()
        self.fitted = False

    def fit(self, windows_easy, windows_hard):
        X = np.array([w.mean(axis=0) for w in windows_easy + windows_hard])
        y = [0] * len(windows_easy) + [1] * len(windows_hard)
        try:
            self.model.fit(X, y)
            self.fitted = True
        except Exception as e:
            print(f"LDA fit error: {e}")

    def predict(self, window):
        if not self.fitted:
            return 0.5
        x = window.mean(axis=0, keepdims=True)
        try:
            return float(self.model.predict_proba(x)[0][1])
        except Exception:
            return 0.5


# ---- DEEP SEQUENCE MODELS ----


class LSTMBaseline(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True
        )
        self.head = nn.Linear(hidden_dim, 1)
        self.fitted = False

    def forward(self, x):
        out, _ = self.lstm(x)
        return torch.sigmoid(self.head(out[:, -1, :])).squeeze(-1)

    def fit(self, windows_easy, windows_hard, epochs=30, lr=1e-3):
        X = torch.tensor(
            np.array(windows_easy + windows_hard), dtype=torch.float32
        )
        y = torch.tensor(
            [0.0] * len(windows_easy) + [1.0] * len(windows_hard),
            dtype=torch.float32,
        )
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        self.train()
        for _ in range(epochs):
            pred = self(X)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.fitted = True

    def predict(self, window):
        if not self.fitted:
            return 0.5
        self.eval()
        with torch.no_grad():
            x = torch.tensor(window[np.newaxis], dtype=torch.float32)
            return float(self(x).item())


class GRUBaseline(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)
        self.fitted = False

    def forward(self, x):
        out, _ = self.gru(x)
        return torch.sigmoid(self.head(out[:, -1, :])).squeeze(-1)

    def fit(self, windows_easy, windows_hard, epochs=30, lr=1e-3):
        X = torch.tensor(
            np.array(windows_easy + windows_hard), dtype=torch.float32
        )
        y = torch.tensor(
            [0.0] * len(windows_easy) + [1.0] * len(windows_hard),
            dtype=torch.float32,
        )
        opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        self.train()
        for _ in range(epochs):
            pred = self(X)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.fitted = True

    def predict(self, window):
        if not self.fitted:
            return 0.5
        self.eval()
        with torch.no_grad():
            x = torch.tensor(window[np.newaxis], dtype=torch.float32)
            return float(self(x).item())


class TransformerBaseline(nn.Module):
    def __init__(self, input_dim=8, d_model=16, nhead=4, num_layers=2):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=64,
            batch_first=True,
            dropout=0.1,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers)
        self.head = nn.Linear(d_model, 1)
        self.fitted = False

    def forward(self, x):
        x = self.proj(x)
        x = self.enc(x)
        return torch.sigmoid(self.head(x.mean(dim=1))).squeeze(-1)

    def fit(self, windows_easy, windows_hard, epochs=30, lr=1e-3):
        X = torch.tensor(
            np.array(windows_easy + windows_hard), dtype=torch.float32
        )
        y = torch.tensor(
            [0.0] * len(windows_easy) + [1.0] * len(windows_hard),
            dtype=torch.float32,
        )
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        self.train()
        for _ in range(epochs):
            pred = self(X)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.fitted = True

    def predict(self, window):
        if not self.fitted:
            return 0.5
        self.eval()
        with torch.no_grad():
            x = torch.tensor(window[np.newaxis], dtype=torch.float32)
            return float(self(x).item())


class PatchTSTBaseline(nn.Module):
    def __init__(self, input_dim=8, patch_size=10, d_model=32, nhead=4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(input_dim * patch_size, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=64,
            batch_first=True,
            dropout=0.1,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.head = nn.Linear(d_model, 1)
        self.fitted = False

    def forward(self, x):
        b, t, f = x.shape
        n_patches = t // self.patch_size
        x = x[:, : n_patches * self.patch_size, :]
        x = x.reshape(b, n_patches, self.patch_size * f)
        x = self.proj(x)
        x = self.enc(x)
        return torch.sigmoid(self.head(x.mean(dim=1))).squeeze(-1)

    def fit(self, windows_easy, windows_hard, epochs=30, lr=1e-3):
        X = torch.tensor(
            np.array(windows_easy + windows_hard), dtype=torch.float32
        )
        y = torch.tensor(
            [0.0] * len(windows_easy) + [1.0] * len(windows_hard),
            dtype=torch.float32,
        )
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        self.train()
        for _ in range(epochs):
            pred = self(X)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.fitted = True

    def predict(self, window):
        if not self.fitted:
            return 0.5
        self.eval()
        with torch.no_grad():
            x = torch.tensor(window[np.newaxis], dtype=torch.float32)
            return float(self(x).item())


# ---- PROBABILISTIC ----


class KalmanBaseline:
    def __init__(self):
        self.x = 0.0
        self.P = 1.0
        self.Q = 0.1
        self.R = 1.0

    def predict(self, window):
        estimates = []
        x, p = self.x, self.P
        for row in window:
            obs = float(row[2] + row[3])
            p = p + self.Q
            k_gain = p / (p + self.R)
            x = x + k_gain * (obs - x)
            p = (1 - k_gain) * p
            estimates.append(x)
        return float(np.mean(estimates[-10:]))


# ---- MAIN COMPARISON FUNCTION ----


def run_all_baselines(windows_easy, windows_hard):
    """
    windows_easy: list of np.array [50,8] — easy/calibration phase
    windows_hard: list of np.array [50,8] — hard phase
    Returns dict: {model_name: {'easy_scores':[], 'hard_scores':[], 'auc':float, ...}}
    """
    from sklearn.metrics import roc_auc_score

    n_easy = len(windows_easy)
    n_hard = len(windows_hard)
    split_easy = int(0.7 * n_easy)
    split_hard = int(0.7 * n_hard)
    train_easy = windows_easy[:split_easy]
    train_hard = windows_hard[:split_hard]
    eval_easy = windows_easy[split_easy:]
    eval_hard = windows_hard[split_hard:]

    results = {}

    def evaluate(name, predict_fn: Callable, fit_fn: Optional[Callable] = None):
        if fit_fn is not None:
            fit_fn()
        easy_s = [predict_fn(w) for w in eval_easy]
        hard_s = [predict_fn(w) for w in eval_hard]
        y_true = [0] * len(easy_s) + [1] * len(hard_s)
        y_score = easy_s + hard_s
        try:
            auc = roc_auc_score(y_true, y_score)
        except Exception:
            auc = 0.5
        results[name] = {
            "easy_scores": easy_s,
            "hard_scores": hard_s,
            "auc": auc,
            "mean_easy": float(np.mean(easy_s)),
            "mean_hard": float(np.mean(hard_s)),
        }
        print(
            f"  {name:25s} AUC={auc:.3f} easy={np.mean(easy_s):.3f} "
            f"hard={np.mean(hard_s):.3f}"
        )

    print("Running baselines...")
    evaluate("Spectral Band Ratio", spectral_band_ratio)
    evaluate("Engagement Index", engagement_index)
    evaluate("Differential Entropy", differential_entropy)
    evaluate("RMS Channel Energy", rms_channel_energy)

    svm = SVMBaseline()
    evaluate("SVM", svm.predict, lambda: svm.fit(train_easy, train_hard))

    hmm = HMMBaseline()
    evaluate("HMM", hmm.predict, lambda: hmm.fit(train_easy, train_hard))

    lda = LDABaseline()
    evaluate("LDA", lda.predict, lambda: lda.fit(train_easy, train_hard))

    lstm = LSTMBaseline()
    evaluate("LSTM", lstm.predict, lambda: lstm.fit(train_easy, train_hard))

    gru = GRUBaseline()
    evaluate("GRU", gru.predict, lambda: gru.fit(train_easy, train_hard))

    transformer = TransformerBaseline()
    evaluate(
        "Transformer",
        transformer.predict,
        lambda: transformer.fit(train_easy, train_hard),
    )

    patchtst = PatchTSTBaseline()
    evaluate(
        "PatchTST",
        patchtst.predict,
        lambda: patchtst.fit(train_easy, train_hard),
    )

    kalman = KalmanBaseline()
    evaluate("Kalman Filter", kalman.predict)

    return results


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base, "cleaned_vegs_data_baseline_z.csv")
    df = pd.read_csv(csv_path)
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

    print(f"Easy windows: {len(windows_easy)}, Hard windows: {len(windows_hard)}")
    results = run_all_baselines(windows_easy, windows_hard)

    print("\n=== BASELINE COMPARISON TABLE ===")
    print(f"{'Model':<25} {'AUC':>6} {'Easy':>8} {'Hard':>8}")
    print("-" * 50)
    for name, r in sorted(results.items(), key=lambda x: -x[1]["auc"]):
        print(
            f"{name:<25} {r['auc']:>6.3f} {r['mean_easy']:>8.3f} "
            f"{r['mean_hard']:>8.3f}"
        )
