# Architecture

This document describes how the Cognitive Transport Index (CTI) pipeline is structured, how the Schrödinger Bridge Problem (SBP) is estimated online, and how the MQTT streaming path fits together.

## System overview

```
Emotiv EPOC X (14-ch, 256 Hz)
        │
        ▼
 Cortex WebSocket  ──►  Feature extraction (8-d band power)
        │
        ▼
 MQTT  eeg/features/{pid}
        │
        ▼
 StreamingSBP subscriber  ──►  CTI  ──►  MQTT eeg/energy/{pid}
        │                                    │
        │                                    ├── live_dashboard.py
        │                                    └── data_logger.py → data/session_data.csv
        │
 game/events/{pid}  ◄──  mt_moral_machine (optional)
```

Three concurrent processes form the live loop:

| Process | Role |
|---|---|
| `pipeline/cortex_client.py` | Headset → features → MQTT |
| `pipeline/sbp_subscriber.py` | Features → SBP / CTI → MQTT |
| `pipeline/data_logger.py` + `live_dashboard.py` | Persist and visualize CTI |

`run_session.py` orchestrates subscriber, logger, dashboard, and optional `mock_eeg.py` replay.

## Feature vector

Each MQTT feature message is an 8-dimensional vector:

| Index | Name | Meaning |
|---|---|---|
| 0 | Theta | θ band power |
| 1 | Alpha | α band power |
| 2 | BetaL | Low β |
| 3 | BetaH | High β |
| 4 | Gamma | γ band power |
| 5 | Arousal | Derived: (βL+βH)/(α+θ) |
| 6 | Valence | Frontal α asymmetry proxy |
| 7 | Engagement | Derived: βL/(α+θ) |

Sliding windows use `n_w = 50` vectors with step `s = 10` (paper defaults). Streaming smoke benchmarks may use smaller windows for speed.

## Schrödinger Bridge core (`core/SBP.py`)

### Score network

`ScoreNet` is a small MLP that estimates the score ∇ log p_t(x) of the bridge marginal at diffusion time `t ∈ [0, 1]`:

```
[x ‖ t_embed] → Linear → ReLU → Linear → ReLU → Linear → ℝ^d
```

Time is repeated across feature dimensions and concatenated with `x` (input width `2d`). Forward and backward networks share this architecture but keep independent weights.

### Training

`train_scores(x0, x1, score_f, score_b, …)`:

1. Sample `t ~ U(0,1)`.
2. Form noisy interpolants `x_t = (1−t)x0 + t x1 + σ√(t(1−t)) ε`.
3. Denoising score-matching loss on forward (`t`) and backward (`1−t`) nets.
4. Adam, default `lr = 1e-3`.

Offline paper runs use more epochs (e.g. 50–100); the streaming path uses a short budget (20 epochs) for sub-second updates.

### Transport energy

`euler_maruyama_sample` integrates the forward SDE. Raw transport energy approximates the expected squared drift along the trajectory:

```
E ≈ (1/T) Σ_i ‖ score_f(x_{t_i}, t_i) ‖²
```

This is the scalar “effort” of transporting the resting prior `P0` to the current window `P1`.

## Streaming CTI (`core/streaming_sbp.py`)

`StreamingSBP` holds:

- **`X0`** — resting / calibration feature matrix (prior).
- **`buffer`** — deque of the last `window_size` live feature vectors.
- **`mu_easy`, `sigma_easy`** — CTI normalization stats from calibration (or easy-phase) energies.
- **`phase`** — `calibration` → `easy_test` → `hard_test`.

### Lifecycle

1. **Calibration** — Fill buffer from rest; each completed window appends `raw_energy` to `energy_history`. `CTI` is `None`.
2. **`set_phase(...)` leaving calibration** — Sets `μ_easy = mean(energy_history)`, `σ_easy = std(energy_history) + ε`.
3. **Task phases** — Each window returns:

```
CTI = (E − μ_easy) / σ_easy
```

Positive CTI ⇒ transport energy above the easy / calibration baseline.

After each window, the buffer advances by `step_size` (overlap), matching the paper’s streaming approximation.

## Baselines (`core/baselines.py`)

Twelve comparators in four groups:

1. **Classical neuroscience** — Spectral Band Ratio, Engagement, Differential Entropy, RMS (label-free).
2. **Classical ML** — SVM, HMM, LDA (supervised).
3. **Deep sequence** — LSTM, GRU, Transformer, PatchTST (supervised).
4. **Dynamical** — Kalman filter (unsupervised, session-level).

Classical indices operate on a single `[50, 8]` window and need no training—ideal for unit tests and the synthetic benchmark.

## Analysis (`analysis/experiments.py`)

Post-session scripts load local (gitignored) CSVs when present and write figures under `analysis/figures/`. Missing inputs are skipped with a clear log line so the script stays crash-safe without publishing subject data.

## Testing & quality signals

| Artifact | Purpose |
|---|---|
| `tests/` | Unit tests for baselines, ScoreNet, StreamingSBP / CTI math |
| `benchmarks/run_synthetic_benchmark.py` | End-to-end smoke + sample report (no real EEG) |
| `samples/outputs/` | Checked-in sample benchmark markdown/JSON |
| `scripts/run_live_demo.py` | Headless MQTT → CTI demo |
| `docker-compose.yml` | Mosquitto + demo in one command |

```bash
pytest -q
python benchmarks/run_synthetic_benchmark.py
docker compose up --build   # see docs/DEPLOYMENT.md
```

## Privacy / data policy

Raw EEG, preprocessed VEGS matrices, and session logs are **not** in this repository. Request data from the authors. Local `data/` and `*.csv` are gitignored by default.
