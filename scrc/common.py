"""Configuration and provenance helpers."""

import hashlib
import json
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def config():
    """Load the frozen experiment settings."""
    cfg = yaml.safe_load(
        (ROOT / os.environ.get("SCRC_CONFIG", "configs/default.yaml")).read_text()
    )
    fractions = [
        cfg[k]
        for k in ["train_fraction", "calibration_fraction", "validation_fraction"]
    ]
    if any(x <= 0 for x in fractions) or sum(fractions) >= 1:
        raise ValueError(
            "Chronological split fractions must be positive and leave a test period."
        )
    if (
        not 0 < cfg["alpha"] < 1
        or not 0 < cfg["propensity_min"] < cfg["propensity_max"] < 1
    ):
        raise ValueError("Invalid calibration or propensity threshold.")
    return cfg


def save_json(path, obj):
    """Write standards-compliant, readable research metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, default=str, allow_nan=False), encoding="utf-8"
    )


def digest(path):
    """Return the SHA-256 of an input or configuration."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
