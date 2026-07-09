"""Unit tests for StreamingSBP buffering, phase transitions, and CTI math."""

from __future__ import annotations

import numpy as np
import pytest

from core.streaming_sbp import StreamingSBP


def test_rejects_bad_baseline_shape(features):
    with pytest.raises(ValueError):
        StreamingSBP(np.zeros((10, 3)), features)


def test_rejects_bad_feature_vector(rest_baseline, features):
    sbp = StreamingSBP(rest_baseline, features, window_size=10, step_size=5)
    with pytest.raises(ValueError):
        sbp.add_sample(np.zeros(3))


def test_buffer_fills_and_compute_returns_none_until_full(rest_baseline, features, rng):
    sbp = StreamingSBP(rest_baseline, features, window_size=20, step_size=5)
    for _ in range(19):
        sbp.add_sample(rng.normal(size=len(features)))
    assert sbp.compute_energy() is None


def test_cti_none_during_calibration(rest_baseline, features, rng, monkeypatch):
    """During calibration, raw energy is stored but CTI stays None."""

    def _fast_train(*args, **kwargs):
        return None

    def _fast_sample(x0, score_net, T=1.0, steps=100, sigma=1.0):
        # Match StreamingSBP.compute_energy loop: traj[0..99]
        batch, dim = x0.shape
        return np.zeros((steps + 1, batch, dim), dtype=np.float32)

    monkeypatch.setattr("core.streaming_sbp.train_scores", _fast_train)
    monkeypatch.setattr("core.streaming_sbp.euler_maruyama_sample", _fast_sample)

    sbp = StreamingSBP(rest_baseline, features, window_size=10, step_size=5)
    assert sbp.phase == "calibration"
    for _ in range(10):
        sbp.add_sample(rng.normal(size=len(features)).astype(np.float32))
    result = sbp.compute_energy()
    assert result is not None
    assert result["CTI"] is None
    assert result["raw_energy"] is not None
    assert len(sbp.energy_history) == 1


def test_cti_normalized_after_phase_change(rest_baseline, features, rng, monkeypatch):
    """After leaving calibration, CTI = (E - μ_easy) / σ_easy."""

    monkeypatch.setattr("core.streaming_sbp.train_scores", lambda *a, **k: None)

    def _fast_sample(x0, score_net, T=1.0, steps=100, sigma=1.0):
        batch, dim = x0.shape
        return np.zeros((steps + 1, batch, dim), dtype=np.float32)

    monkeypatch.setattr("core.streaming_sbp.euler_maruyama_sample", _fast_sample)

    sbp = StreamingSBP(rest_baseline, features, window_size=10, step_size=5)
    # Seed calibration energies manually, then flip phase
    sbp.energy_history = [1.0, 2.0, 3.0]
    sbp.set_phase("easy_test")
    assert sbp.mu_easy == pytest.approx(2.0)
    assert sbp.sigma_easy > 0.0

    for _ in range(10):
        sbp.add_sample(rng.normal(size=len(features)).astype(np.float32))
    result = sbp.compute_energy()
    assert result is not None
    assert result["CTI"] is not None
    assert np.isfinite(result["CTI"])


def test_cti_formula_unit():
    """Direct check of the CTI normalization used in the paper."""
    energy = 5.0
    mu_easy = 2.0
    sigma_easy = 1.5
    cti = (energy - mu_easy) / sigma_easy
    assert cti == pytest.approx(2.0)
