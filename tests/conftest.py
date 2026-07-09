"""Shared fixtures: synthetic 8-d EEG-like feature windows (no real subject data)."""

from __future__ import annotations

import numpy as np
import pytest

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


@pytest.fixture
def features():
    return list(FEATURES)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def rest_baseline(rng, features):
    """Resting-state prior P0: low-variance Gaussian cloud."""
    return rng.normal(loc=0.0, scale=0.4, size=(80, len(features))).astype(np.float32)


@pytest.fixture
def easy_window(rng, features):
    """Easy-task window: mild shift from rest."""
    return rng.normal(loc=0.3, scale=0.5, size=(50, len(features))).astype(np.float32)


@pytest.fixture
def hard_window(rng, features):
    """Hard-task window: larger shift / higher energy than easy."""
    return rng.normal(loc=1.2, scale=0.8, size=(50, len(features))).astype(np.float32)
