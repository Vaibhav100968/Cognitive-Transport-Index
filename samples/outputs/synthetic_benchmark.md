# Synthetic CTI benchmark (sample output)

_Generated: `2026-07-09T22:34:46.487442+00:00` · seed `42`_

This report is produced by `benchmarks/run_synthetic_benchmark.py` using
**synthetic** Gaussian feature clouds (no subject EEG). It demonstrates that
classical baselines and (optionally) StreamingSBP run end-to-end.

## Classical baselines

| Model | Easy | Hard | Hard − Easy |
|---|---:|---:|---:|
| Spectral Band Ratio | 0.6634 | 0.5887 | -0.0746 |
| Engagement Index | 0.4113 | 0.5345 | 0.1232 |
| Differential Entropy | 0.6941 | 1.2172 | 0.5232 |
| RMS Channel Energy | 0.5739 | 1.4577 | 0.8838 |

## Streaming SBP / CTI

_Skipped_ (run with `--with-sbp` to train score networks on synthetic data).

## How to reproduce

```bash
python benchmarks/run_synthetic_benchmark.py
python benchmarks/run_synthetic_benchmark.py --with-sbp
```
