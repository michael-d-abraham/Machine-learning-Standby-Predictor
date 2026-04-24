"""
Assignment 12 Part B: Scoped practical deep learning system on capstone proxy task.

Small feedforward net in PyTorch; same data split and preprocessing philosophy as main.py:
artist-based 60/20/20, binary target from train median only, median impute + scaler fit on train.

System-level focus: reproducibility (seeds), explicit device, train/val observability,
batching, wall-clock, JSON artifact with full per-epoch history, and training-curve PNG.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import main as base

SEED = 42
BATCH_SIZE = 256
MAX_EPOCHS = 40
LEARNING_RATE = 1e-3
HIDDEN = (64, 32)
_ROOT = Path(__file__).resolve().parent
OUTPUT_JSON = _ROOT / "assignment12_dl_run.json"
OUTPUT_CURVES_PNG = _ROOT / "assignment12_train_val_curves.png"


def _set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TabularMLP(nn.Module):
    def __init__(self, n_features: int, hidden: tuple[int, ...] = (64, 32)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_features
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _fit_preprocess(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train_t = imputer.fit_transform(X_train)
    X_val_t = imputer.transform(X_val)
    X_test_t = imputer.transform(X_test)
    X_train_t = scaler.fit_transform(X_train_t)
    X_val_t = scaler.transform(X_val_t)
    X_test_t = scaler.transform(X_test_t)
    return X_train_t, X_val_t, X_test_t


def _save_training_curves(history: list[dict], path: Path) -> None:
    """Train/val loss and val ROC-AUC vs epoch (stability, overfitting at a glance)."""
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    fig, (ax_loss, ax_auc) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax_loss.plot(epochs, [h["train_loss"] for h in history], label="train_loss", color="tab:blue")
    ax_loss.plot(epochs, [h["val_loss"] for h in history], label="val_loss", color="tab:orange")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Assignment 12: training curves (proxy energy classifier)")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)
    ax_auc.plot(epochs, [h["val_roc_auc"] for h in history], label="val_roc_auc", color="tab:green")
    ax_auc.set_xlabel("Epoch")
    ax_auc.set_ylabel("Val ROC-AUC")
    ax_auc.legend()
    ax_auc.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def run() -> dict:
    _set_seeds(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df, numeric_features = base.inspect_dataset(base.CSV_PATH)
    X_train, X_val, X_test, y_train, y_val, y_test, _ = base.split_data(
        df,
        numeric_features,
        base.TARGET_COL,
        seed=SEED,
        train_frac=0.60,
        val_frac=0.20,
    )

    X_tr, X_va, X_te = _fit_preprocess(X_train, X_val, X_test)
    n_features = X_tr.shape[1]

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_train.values, dtype=torch.float32, device=device)
    X_va_t = torch.tensor(X_va, dtype=torch.float32, device=device)
    y_va_t = torch.tensor(y_val.values, dtype=torch.float32, device=device)
    X_te_t = torch.tensor(X_te, dtype=torch.float32, device=device)
    y_te_t = torch.tensor(y_test.values, dtype=torch.float32, device=device)

    train_loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )

    model = TabularMLP(n_features, HIDDEN).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()

    history: list[dict] = []
    t0 = time.perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses: list[float] = []
        for xb, yb in train_loader:
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_logits = model(X_va_t)
            val_loss = float(loss_fn(val_logits, y_va_t).cpu())
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
        train_loss_mean = float(np.mean(train_losses))
        val_auc = float(roc_auc_score(y_val, val_probs))
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_mean,
                "val_loss": val_loss,
                "val_roc_auc": val_auc,
            }
        )
        if epoch == 1 or epoch % 10 == 0 or epoch == MAX_EPOCHS:
            print(
                f"epoch={epoch:3d} train_loss={train_loss_mean:.4f} "
                f"val_loss={val_loss:.4f} val_roc_auc={val_auc:.4f}"
            )

    wall_s = time.perf_counter() - t0

    model.eval()
    with torch.no_grad():
        test_logits = model(X_te_t)
        test_probs = torch.sigmoid(test_logits).cpu().numpy()
    test_auc = float(roc_auc_score(y_test, test_probs))

    meta = {
        "framework": "pytorch",
        "torch_version": torch.__version__,
        "device": str(device),
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "hidden_layers": list(HIDDEN),
        "n_features": int(n_features),
        "train_rows": int(len(y_train)),
        "val_rows": int(len(y_val)),
        "test_rows": int(len(y_test)),
        "wall_clock_seconds": round(wall_s, 3),
        "final_val_roc_auc": history[-1]["val_roc_auc"],
        "test_roc_auc": test_auc,
        "first_epoch": history[0],
        "last_epoch": history[-1],
    }

    _save_training_curves(history, OUTPUT_CURVES_PNG)
    payload = {
        "meta": meta,
        "history": history,
        "history_tail": history[-5:],
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wall clock (train+eval loop): {wall_s:.2f}s on {device}")
    print(f"Test ROC-AUC: {test_auc:.4f}")

    return payload


if __name__ == "__main__":
    run()
