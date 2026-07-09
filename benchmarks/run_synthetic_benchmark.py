#!/usr/bin/env python3
"""
Synthetic CTI smoke benchmark (no real EEG / no MQTT).

Generates rest / easy / hard feature clouds, runs classical baselines, and
optionally a short StreamingSBP pass. Writes a markdown report under
`samples/outputs/` so the repo has visible proof-of-functionality artifacts.

Usage:
  python benchmarks/run_synthetic_benchmark.py
  python benchmarks/run_synthetic_benchmark.py --with-sbp   # slower; trains score nets
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.baselines import (  # noqa: E402
    differential_entropy,
    engagement_index,
    rms_channel_energy,
    spectral_band_ratio,
)

FEATURES = [
    "Theta",
    "Alpha",
    "BetaL",
    "BetaH",
    "Gamma",
    "Arousal",
    "Valence",
    "Engagement",
]


def _make_clouds(rng: np.random.Generator, dim: int = 8):
    rest = rng.normal(0.0, 0.4, size=(80, dim)).astype(np.float32)
    easy = rng.normal(0.3, 0.5, size=(50, dim)).astype(np.float32)
    hard = rng.normal(1.2, 0.8, size=(50, dim)).astype(np.float32)
    return rest, easy, hard


def _baseline_table(easy: np.ndarray, hard: np.ndarray) -> list[dict]:
    rows = []
    for name, fn in [
        ("Spectral Band Ratio", spectral_band_ratio),
        ("Engagement Index", engagement_index),
        ("Differential Entropy", differential_entropy),
        ("RMS Channel Energy", rms_channel_energy),
    ]:
        e = fn(easy)
        h = fn(hard)
        rows.append(
            {
                "model": name,
                "easy": float(e),
                "hard": float(h),
                "hard_minus_easy": float(h - e),
            }
        )
    return rows


def _run_streaming_sbp(rest: np.ndarray, easy: np.ndarray, hard: np.ndarray) -> dict:
    from core.streaming_sbp import StreamingSBP

    # Small window for a fast smoke run
    window_size = 20
    step_size = 10
    sbp = StreamingSBP(rest, FEATURES, window_size=window_size, step_size=step_size)

    t0 = time.perf_counter()
    cal_energies = []
    for i, row in enumerate(rest[:60]):
        sbp.add_sample(row)
        if len(sbp.buffer) >= window_size and i % step_size == 0:
            out = sbp.compute_energy()
            if out is not None:
                cal_energies.append(out["raw_energy"])

    sbp.set_phase("easy_test")
    easy_cti = []
    for i, row in enumerate(easy):
        sbp.add_sample(row)
        if len(sbp.buffer) >= window_size and i % step_size == 0:
            out = sbp.compute_energy()
            if out is not None and out["CTI"] is not None:
                easy_cti.append(out["CTI"])

    sbp.set_phase("hard_test")
    hard_cti = []
    for i, row in enumerate(hard):
        sbp.add_sample(row)
        if len(sbp.buffer) >= window_size and i % step_size == 0:
            out = sbp.compute_energy()
            if out is not None and out["CTI"] is not None:
                hard_cti.append(out["CTI"])

    elapsed = time.perf_counter() - t0
    return {
        "window_size": window_size,
        "step_size": step_size,
        "n_calibration_windows": len(cal_energies),
        "n_easy_cti": len(easy_cti),
        "n_hard_cti": len(hard_cti),
        "mean_easy_cti": float(np.mean(easy_cti)) if easy_cti else None,
        "mean_hard_cti": float(np.mean(hard_cti)) if hard_cti else None,
        "elapsed_sec": round(elapsed, 3),
    }


def _write_report(out_dir: str, payload: dict) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "synthetic_benchmark.json")
    md_path = os.path.join(out_dir, "synthetic_benchmark.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    lines = [
        "# Synthetic CTI benchmark (sample output)",
        "",
        f"_Generated: `{payload['generated_at']}` · seed `{payload['seed']}`_",
        "",
        "This report is produced by `benchmarks/run_synthetic_benchmark.py` using",
        "**synthetic** Gaussian feature clouds (no subject EEG). It demonstrates that",
        "classical baselines and (optionally) StreamingSBP run end-to-end.",
        "",
        "## Classical baselines",
        "",
        "| Model | Easy | Hard | Hard − Easy |",
        "|---|---:|---:|---:|",
    ]
    for row in payload["baselines"]:
        lines.append(
            f"| {row['model']} | {row['easy']:.4f} | {row['hard']:.4f} | "
            f"{row['hard_minus_easy']:.4f} |"
        )

    lines += ["", "## Streaming SBP / CTI", ""]
    sbp = payload.get("streaming_sbp")
    if sbp is None:
        lines.append(
            "_Skipped_ (run with `--with-sbp` to train score networks on synthetic data)."
        )
    else:
        lines += [
            f"- Calibration windows: **{sbp['n_calibration_windows']}**",
            f"- Easy CTI windows: **{sbp['n_easy_cti']}** "
            f"(mean `{sbp['mean_easy_cti']}`)",
            f"- Hard CTI windows: **{sbp['n_hard_cti']}** "
            f"(mean `{sbp['mean_hard_cti']}`)",
            f"- Wall time: **{sbp['elapsed_sec']} s**",
            "",
            "On this synthetic setup, hard-condition mean CTI should typically exceed",
            "easy-condition mean CTI (direction check only; not a paper claim).",
        ]

    lines += [
        "",
        "## How to reproduce",
        "",
        "```bash",
        "python benchmarks/run_synthetic_benchmark.py",
        "python benchmarks/run_synthetic_benchmark.py --with-sbp",
        "```",
        "",
    ]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return json_path, md_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-sbp",
        action="store_true",
        help="Also run a short StreamingSBP pass (trains score nets; slower).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    rest, easy, hard = _make_clouds(rng)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "feature_dim": len(FEATURES),
        "features": FEATURES,
        "baselines": _baseline_table(easy, hard),
        "streaming_sbp": None,
    }

    print("Classical baselines (synthetic easy vs hard):")
    for row in payload["baselines"]:
        print(
            f"  {row['model']:24s}  easy={row['easy']:.4f}  "
            f"hard={row['hard']:.4f}  Δ={row['hard_minus_easy']:.4f}"
        )

    if args.with_sbp:
        print("\nRunning StreamingSBP smoke pass (synthetic)…")
        payload["streaming_sbp"] = _run_streaming_sbp(rest, easy, hard)
        s = payload["streaming_sbp"]
        print(
            f"  easy CTI mean={s['mean_easy_cti']}  "
            f"hard CTI mean={s['mean_hard_cti']}  "
            f"({s['elapsed_sec']} s)"
        )

    out_dir = os.path.join(_ROOT, "samples", "outputs")
    json_path, md_path = _write_report(out_dir, payload)
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
