import argparse

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ScoreNet(nn.Module):
    def __init__(self, dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x, t):
        if len(t.shape) == 1:
            t = t.view(-1, 1)
        t_embed = t.repeat(1, x.shape[1])
        inp = torch.cat([x, t_embed], dim=1)
        return self.net(inp)


def interpolate_samples(x0, x1, t):
    sigma = 1.0
    noise = torch.randn_like(x0) * sigma * torch.sqrt(t * (1 - t))
    return (1 - t) * x0 + t * x1 + noise


def score_matching_loss(score_net, x, t):
    x = x.clone().requires_grad_(True)
    score = score_net(x, t)
    noise_var = (t * (1 - t)).view(-1, 1)
    target = -(x - x.detach()) / noise_var
    return ((score - target) ** 2).mean()


def train_scores(x0, x1, score_f, score_b, epochs=100, batch_size=128):
    optimizer_f = optim.Adam(score_f.parameters(), lr=1e-3)
    optimizer_b = optim.Adam(score_b.parameters(), lr=1e-3)

    n = x0.shape[0]

    for epoch in range(epochs):
        perm = torch.randperm(n)
        loss_f_epoch, loss_b_epoch = 0, 0

        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) == 0:
                continue

            t = torch.rand(len(idx), device=device).unsqueeze(1)

            # Forward
            x_t = interpolate_samples(x0[idx], x1[idx], t)
            optimizer_f.zero_grad()
            loss_f = score_matching_loss(score_f, x_t, t)
            loss_f.backward()
            optimizer_f.step()
            loss_f_epoch += loss_f.item() * len(idx)

            # Backward
            t_rev = 1 - t
            x_t_rev = interpolate_samples(x0[idx], x1[idx], t_rev)
            optimizer_b.zero_grad()
            loss_b = score_matching_loss(score_b, x_t_rev, t_rev)
            loss_b.backward()
            optimizer_b.step()
            loss_b_epoch += loss_b.item() * len(idx)

        print(
            f"Epoch {epoch+1}/{epochs}, "
            f"LossF: {loss_f_epoch/n:.6f}, "
            f"LossB: {loss_b_epoch/n:.6f}"
        )


def euler_maruyama_sample(x0, score_net, T=1.0, steps=100, sigma=1.0):
    dt = T / steps
    x = x0.clone().to(device)
    traj = [x.cpu().numpy()]

    for i in range(steps):
        t = torch.full((x.shape[0], 1), i * dt, device=device)
        with torch.no_grad():
            drift = sigma ** 2 * score_net(x, t)
            noise = torch.randn_like(x) * np.sqrt(dt) * sigma
            x = x + drift * dt + noise
        traj.append(x.cpu().numpy())

    return np.array(traj)


def run_true_sbp(df, features, epochs=100, batch_size=128, output_csv="true_sbp_results.csv"):
    results = []
    scaler = StandardScaler()

    for participant in df['Participant'].unique():

        subset_p1 = df[
            (df['Participant'] == participant) &
            (df['portion'] == 1)
        ][features].values

        if len(subset_p1) < 20:
            continue

        X0_full = scaler.fit_transform(subset_p1)
        X0_full = torch.tensor(X0_full, dtype=torch.float32, device=device)

        for portion in [2, 3, 4]:
            subset_pk = df[
                (df['Participant'] == participant) &
                (df['portion'] == portion)
            ][features].values

            if len(subset_pk) < 20:
                continue

            X1_full = scaler.transform(subset_pk)
            X1_full = torch.tensor(X1_full, dtype=torch.float32, device=device)

            # ✅ MATCH SAMPLE COUNTS (ONLY PLACE THIS SHOULD BE)
            n = min(len(X0_full), len(X1_full))
            X0 = X0_full[:n]
            X1 = X1_full[:n]

            score_f = ScoreNet(len(features)).to(device)
            score_b = ScoreNet(len(features)).to(device)

            train_scores(X0, X1, score_f, score_b,
                         epochs=epochs, batch_size=batch_size)

            traj = euler_maruyama_sample(X0, score_f, steps=100)

            energy = 0
            for i in range(100):
                t = torch.full((traj.shape[1], 1), i / 100, device=device)
                drift = score_f(torch.tensor(traj[i], device=device), t)
                energy += (drift.norm(dim=1) ** 2).mean().item() / 100

            results.append({
                "Participant": participant,
                "Portion": portion,
                "SBP_Energy": energy
            })

    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"SBP results saved to {output_csv}")
    return results


DEFAULT_FEATURES = [
    "Theta",
    "Alpha",
    "BetaL",
    "BetaH",
    "Gamma",
    "Arousal",
    "Valence",
    "Engagement",
]


def normalize_portion_column(df: pd.DataFrame) -> pd.DataFrame:
    """Accept portion as 1–4 or legacy strings like 'portion 1'."""
    out = df.copy()
    p = out["portion"]

    def to_int(v: object) -> int:
        if isinstance(v, (int, np.integer)):
            return int(v)
        s = str(v).strip().lower()
        if s.startswith("portion"):
            return int(s.replace("portion", "", 1).strip())
        return int(float(s))

    if not np.issubdtype(p.dtype, np.number):
        out["portion"] = p.map(to_int)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Train score nets and compute SBP energy per participant/portion.")
    p.add_argument(
        "--input",
        default="generated_eeg_baseline_z.csv",
        help="CSV with Participant, portion (1–4), and feature columns",
    )
    p.add_argument(
        "--output",
        default="generated_sbp_results.csv",
        help="where to write SBP energy results",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    df = normalize_portion_column(pd.read_csv(args.input))
    run_true_sbp(
        df,
        DEFAULT_FEATURES,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_csv=args.output,
    )


if __name__ == "__main__":
    main()
