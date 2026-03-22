"""
src/ml/predictor.py
====================
Thin wrapper around the pre-trained Random Forest pipeline.

Usage
-----
    from src.ml.predictor import RiskPredictor, predict_risk

    predictor = RiskPredictor.load()          # loads from models/
    score = predictor.predict(features_dict)  # → float [0, 1]

    # Or use the stateless convenience function (matches the old ui/app.py API):
    models = load_ml_models()
    score  = predict_risk(models, features)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import joblib

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = _PROJECT_ROOT / "models"

FEATURE_COLUMNS = [
    "financial_loss",
    "affected_users",
    "response_time",
    "vulnerability_score",
    "attack_enc",
    "industry_enc",
    "country_enc",
]


# ---------------------------------------------------------------------------
# RiskPredictor class
# ---------------------------------------------------------------------------
class RiskPredictor:
    """Wraps the RF model, scaler and label encoders into a single object."""

    def __init__(self, rf_model, scaler, le_attack, le_industry, le_country):
        self.rf_model = rf_model
        self.scaler = scaler
        self.le_attack = le_attack
        self.le_industry = le_industry
        self.le_country = le_country

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, models_dir: Path | None = None) -> "RiskPredictor":
        """
        Load all five pkl artifacts from *models_dir* (default: ``models/``).

        Raises
        ------
        FileNotFoundError
            If any required .pkl file is missing.
        """
        if models_dir is None:
            models_dir = MODELS_DIR

        required = {
            "rf_model":    models_dir / "rf_model.pkl",
            "scaler":      models_dir / "scaler.pkl",
            "le_attack":   models_dir / "le_attack.pkl",
            "le_industry": models_dir / "le_industry.pkl",
            "le_country":  models_dir / "le_country.pkl",
        }

        missing = [n for n, p in required.items() if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing model artifacts in {models_dir}: {', '.join(missing)}. "
                f"Run `python src/ml/train.py` first."
            )

        return cls(**{name: joblib.load(str(path)) for name, path in required.items()})

    # ------------------------------------------------------------------
    def predict(self, features: Dict[str, Any]) -> float:
        """
        Return the probability of *high risk* for the given incident features.

        Parameters
        ----------
        features : dict with keys
            ``attack_type``, ``industry``, ``country``,
            ``financial_loss``, ``affected_users``,
            ``response_time``, ``vulnerability_score``

        Returns
        -------
        float
            Risk probability in [0.0, 1.0].
        """
        row = [
            features["financial_loss"],
            features["affected_users"],
            features["response_time"],
            features["vulnerability_score"],
            self.le_attack.transform([features["attack_type"]])[0],
            self.le_industry.transform([features["industry"]])[0],
            self.le_country.transform([features["country"]])[0],
        ]
        input_df = pd.DataFrame([row], columns=FEATURE_COLUMNS)
        scaled = self.scaler.transform(input_df)
        return float(self.rf_model.predict_proba(scaled)[0][1])

    # alias kept for backward-compat
    __call__ = predict


# ---------------------------------------------------------------------------
# Stateless convenience function (mirrors the old ui/app.py signature)
# ---------------------------------------------------------------------------
def predict_risk(models: dict, features: Dict[str, Any]) -> float:
    """
    Drop-in replacement for the old ``predict_risk`` defined in ``ui/app.py``.

    Parameters
    ----------
    models : dict
        Dict with keys: ``rf_model``, ``scaler``, ``le_attack``,
        ``le_industry``, ``le_country``.
    features : dict
        Same keys as ``RiskPredictor.predict()``.

    Returns
    -------
    float
    """
    predictor = RiskPredictor(
        rf_model=models["rf_model"],
        scaler=models["scaler"],
        le_attack=models["le_attack"],
        le_industry=models["le_industry"],
        le_country=models["le_country"],
    )
    return predictor.predict(features)
