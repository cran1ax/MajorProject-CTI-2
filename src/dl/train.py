"""
src/dl/train.py
===============
Train the CyberMLP on the same feature set as the RF pipeline and save
the model weights + scaler to ``models/``.

Run from the project root::

    python -m src.dl.train            # defaults
    python -m src.dl.train --epochs 100 --lr 1e-3

Requirements
------------
    pip install torch joblib scikit-learn pandas numpy
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# --- make sure project root is on sys.path ---
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score

from src.dl.model import CyberMLP

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODELS_DIR = _ROOT / "models"
DATA_CSV   = _ROOT / "cybersecurity_threats.csv"
FEATURE_COLUMNS = [
    "financial_loss",
    "affected_users",
    "response_time",
    "vulnerability_score",
    "attack_enc",
    "industry_enc",
    "country_enc",
]
ATTACK_CLASSES   = ["DDoS", "Insider Threat", "Malware", "Phishing", "Ransomware"]
INDUSTRY_CLASSES = ["Education", "Finance", "Government", "Healthcare", "Technology"]
COUNTRY_CLASSES  = ["Germany", "India", "Japan", "UK", "USA"]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def _load_or_generate_data(csv_path: Path) -> pd.DataFrame:
    """Return a DataFrame from CSV, or generate synthetic data if CSV is absent."""
    if csv_path.is_file():
        df = pd.read_csv(str(csv_path))
        # Map column names to the expected schema
        rename = {
            "target_industry": "industry",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        return df

    print("[DL train] CSV not found — generating synthetic data …")
    rng = np.random.default_rng(42)
    n = 2000

    attack_types  = rng.choice(ATTACK_CLASSES,   n)
    industries    = rng.choice(INDUSTRY_CLASSES, n)
    countries     = rng.choice(COUNTRY_CLASSES,  n)
    fin_loss      = rng.exponential(50, n)
    affected      = rng.poisson(10_000, n)
    resp_time     = rng.gamma(2, 2, n)
    vuln_score    = rng.uniform(1, 10, n)
    high_risk     = ((fin_loss > 80) | (affected > 30_000) | (resp_time > 8)).astype(int)

    return pd.DataFrame({
        "attack_type":      attack_types,
        "industry":         industries,
        "country":          countries,
        "financial_loss":   fin_loss,
        "affected_users":   affected,
        "response_time":    resp_time,
        "vulnerability_score": vuln_score,
        "high_risk_incident":  high_risk,
    })


def _build_features(df: pd.DataFrame):
    """Encode categoricals and return (X_arr, y_arr, le_attack, le_industry, le_country)."""
    # Normalise column names
    col_map = {
        "target_industry": "industry",
        "attack_type_raw": "attack_type",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    le_attack   = LabelEncoder()
    le_industry = LabelEncoder()
    le_country  = LabelEncoder()

    df["attack_enc"]   = le_attack.fit_transform(df["attack_type"].str.strip())
    df["industry_enc"] = le_industry.fit_transform(df["industry"].str.strip())
    df["country_enc"]  = le_country.fit_transform(df["country"].str.strip())

    X = df[FEATURE_COLUMNS].values.astype(np.float32)
    y = df["high_risk_incident"].values.astype(np.float32)
    return X, y, le_attack, le_industry, le_country


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(
    epochs: int = 60,
    lr: float = 1e-3,
    batch_size: int = 64,
    val_split: float = 0.2,
    random_state: int = 42,
) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── load data ──────────────────────────────────────────────────────────
    df = _load_or_generate_data(DATA_CSV)
    X, y, le_attack, le_industry, le_country = _build_features(df)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)

    # ── tensors ───────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xtr = torch.from_numpy(X_train).to(device)
    ytr = torch.from_numpy(y_train).to(device)
    Xvl = torch.from_numpy(X_val).to(device)
    yvl = torch.from_numpy(y_val).to(device)

    # ── model ─────────────────────────────────────────────────────────────
    model = CyberMLP(input_dim=X.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n = len(Xtr)
    best_auc = 0.0
    best_state = None

    print(f"\n[DL train] Device: {device} | epochs={epochs} | lr={lr} | batch={batch_size}")
    print(f"[DL train] Train={len(X_train)} | Val={len(X_val)} | Features={X.shape[1]}")

    for epoch in range(1, epochs + 1):
        model.train()
        idx = torch.randperm(n, device=device)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            batch_idx = idx[start : start + batch_size]
            xb, yb = Xtr[batch_idx], ytr[batch_idx]
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_idx)
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_probs = model(Xvl).cpu().numpy()
            auc = roc_auc_score(y_val, val_probs)
            acc = accuracy_score(y_val, val_probs >= 0.5)
            print(f"  epoch {epoch:>3}/{epochs}  loss={epoch_loss/n:.4f}  val_AUC={auc:.4f}  val_Acc={acc:.4f}")

            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # ── save ─────────────────────────────────────────────────────────────
    model_path  = MODELS_DIR / "dl_model.pt"
    scaler_path = MODELS_DIR / "dl_scaler.pkl"

    torch.save(best_state, str(model_path))
    joblib.dump(scaler, str(scaler_path))

    print(f"\n✅  DL model saved → {model_path}")
    print(f"✅  DL scaler saved → {scaler_path}")
    print(f"    Best Val AUC: {best_auc:.4f}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CyberMLP and save to models/")
    parser.add_argument("--epochs",     type=int,   default=60,   help="training epochs")
    parser.add_argument("--lr",         type=float, default=1e-3, help="learning rate")
    parser.add_argument("--batch-size", type=int,   default=64,   help="mini-batch size")
    args = parser.parse_args()

    train(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
