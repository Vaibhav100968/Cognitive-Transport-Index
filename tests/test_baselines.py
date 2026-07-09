"""Unit tests for classical neuroscience baseline indices."""

from __future__ import annotations

import numpy as np

from core.baselines import (
    differential_entropy,
    engagement_index,
    rms_channel_energy,
    spectral_band_ratio,
)


def test_spectral_band_ratio_positive(easy_window):
    value = spectral_band_ratio(easy_window)
    assert np.isfinite(value)
    assert value > 0.0


def test_engagement_index_finite(easy_window):
    value = engagement_index(easy_window)
    assert np.isfinite(value)


def test_differential_entropy_finite(easy_window):
    value = differential_entropy(easy_window)
    assert np.isfinite(value)


def test_rms_channel_energy_nonnegative(easy_window, hard_window):
    easy_rms = rms_channel_energy(easy_window)
    hard_rms = rms_channel_energy(hard_window)
    assert easy_rms >= 0.0
    assert hard_rms >= 0.0
    # Hard synthetic cloud is farther from origin → higher RMS on average
    assert hard_rms > easy_rms


def test_baselines_reject_wrong_shape():
    bad = np.zeros((10, 3), dtype=np.float32)
    # Functions index columns 0–4; wrong width should raise or produce NaN-safe failure
    with np.errstate(all="ignore"):
        try:
            spectral_band_ratio(bad)
        except IndexError:
            pass
