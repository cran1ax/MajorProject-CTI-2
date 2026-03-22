"""
src/dl/predictor.py
===================
Inference wrapper around the trained ``CyberMLP`` model.

Usage
-----
    from src.dl.predictor import DeepRiskPredictor

    predictor = DeepRiskPredictor()
    predictor.load()                     # loads models/dl_model.pt + models/dl_scaler.pkl
    score = predictor.predict(features)  # → float in [0, 1]
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import joblib

from src.dl.model import CyberMLP

# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = _PROJECT_ROOT / "models"

FEATURE_ORDER = [
    "financial_loss",
    "affected_users",
    "response_time",
    "vulnerability_score",
    "attack_enc",
    "industry_enc",
    "country_enc",
]


class DLNotLoadedError(RuntimeError):
    """Raised when predict() is called before load()."""


class DeepRiskPredictor:
    """
    Wraps ``CyberMLP`` for single-sample and batch inference.

    Parameters
    ----------
    models_dir : Path, optional
        Directory containing ``dl_model.pt`` and ``dl_scaler.pkl``.
        Defaults to the project's ``models/`` directory.
    """

    def __init__(self, models_dir: Path | None = None) -> None:
        self._dir = models_dir or MODELS_DIR
        self._model: CyberMLP | None = None
        self._scaler = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    def load(self) -> "DeepRiskPredictor":
        """
        Load model weights and scaler from disk.

        Returns self (chainable).

        Raises
        ------
        FileNotFoundError
            If ``dl_model.pt`` or ``dl_scaler.pkl`` is missing.
        """
        model_path  = self._dir / "dl_model.pt"
        scaler_path = self._dir / "dl_scaler.pkl"

        for p in (model_path, scaler_path):
            if not p.is_file():
                raise FileNotFoundError(
                    f"DL artifact not found: {p}. "
                    f"Run `python src/dl/train.py` first."
                )

        self._model = CyberMLP()
        state = torch.load(str(model_path), map_location=self._device, weights_only=True)
        self._model.load_state_dict(state)
        self._model.to(self._device)
        self._model.eval()

        self._scaler = joblib.load(str(scaler_path))
        return self

    # ------------------------------------------------------------------
    def predict(self, features: Dict[str, Any]) -> float:
        """
        Predict the risk probability for a single incident.

        Parameters
        ----------
        features : dict
            Must contain numeric keys ``financial_loss``, ``affected_users``,
            ``response_time``, ``vulnerability_score``, ``attack_enc``,
            ``industry_enc``, ``country_enc``.

        Returns
        -------
        float
            Risk probability in [0.0, 1.0].
        """
        if self._model is None:
            raise DLNotLoadedError("Call load() before predict().")

        row = np.array([[features[k] for k in FEATURE_ORDER]], dtype=np.float32)
        scaled = self._scaler.transform(row)
        tensor = torch.from_numpy(scaled).to(self._device)

        with torch.no_grad():
            prob = self._model(tensor).item()

        return float(prob)

    # ------------------------------------------------------------------
    def is_loaded(self) -> bool:
        """Return True if the model weights are in memory."""
        return self._model is not None
