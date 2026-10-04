import argparse
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pennylane as qml
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, roc_curve, precision_recall_curve, confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

SEED = 42
@torch.no_grad()
def predict_proba_torch(model, X, batch_size=1024):
    model.eval()
    dtype = next(model.parameters()).dtype
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.tensor(X[i:i + batch_size], dtype=dtype)
        out.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(out)

def make_validation_split(y, val_frac=0.15):
    n = len(y)
    tail = 0
    while tail < n and y[n - 1 - tail] == 1:
        tail += 1

    if tail >= 0.05 * n:
        n_orig = n - tail
        print(f"Validation: first {n_orig} rows treated as original "
              f"(trailing run of {tail} positives looks like appended SMOTE rows; "
              f"val prevalence {y[:n_orig].mean():.1%}).")
        tr_o, va = train_test_split(np.arange(n_orig), test_size=val_frac,
                                    stratify=y[:n_orig], random_state=SEED)
        return np.concatenate([tr_o, np.arange(n_orig, n)]), va

    return train_test_split(np.arange(n), test_size=val_frac, stratify=y, random_state=SEED)

def load_data():
    def _clean(df):
        return df.drop(columns=[c for c in ["SEQN", "Unnamed: 0"] if c in df.columns])

    X_train = _clean(pd.read_csv("X_train_after_smote.csv"))
    X_test = _clean(pd.read_csv("X_test.csv"))
    y_train = pd.read_csv("y_train_after_smote.csv").iloc[:, -1].values.astype(int)
    y_test = pd.read_csv("y_test.csv").iloc[:, -1].values.astype(int)

    feature_names = list(X_train.columns)
    return X_train.values.astype(float), X_test[feature_names].values.astype(float), y_train, y_test, feature_names


N_QUBITS = 8
N_LAYERS = 4
HIDDEN = (64,32)
TRAIN_CFG = dict(epochs=200, patience=20, lr=3e-3, batch_size=128, weight_decay=1e-4)

class FrozenBaselinePlusCorrection(nn.Module):

    def __init__(self, correction_module):
        super().__init__()
        self.correction_module = correction_module

    def forward(self, x_and_logit):
        x, frozen_logit = x_and_logit[:, :-1], x_and_logit[:, -1]
        return frozen_logit + self.correction_module.correction(x)


def with_frozen_logit(X, logit):
    return np.concatenate([X, logit[:, None]], axis=1)


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_multiread_circuit(n_qubits, n_layers):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(x, in_scale, weights):
        for l in range(n_layers):
            qml.AngleEmbedding(x * in_scale[l], wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(
                weights[l:l + 1], wires=range(n_qubits),
                ranges=[(l % (n_qubits - 1)) + 1],
            )
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit

def _encoder(d_in, n_qubits, hidden):
    layers, prev = [], d_in
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.3)]
        prev = h
    layers += [nn.Linear(prev, n_qubits), nn.Tanh()]
    return nn.Sequential(*layers)

class QuantumResidualCorrection(nn.Module):

    def __init__(self, d_in, n_qubits=N_QUBITS, n_layers=N_LAYERS, hidden=HIDDEN):
        super().__init__()
        self.encoder = _encoder(d_in, n_qubits, hidden)
        self.circuit = make_multiread_circuit(n_qubits, n_layers)
        self.in_scale = nn.Parameter(torch.ones(n_layers, n_qubits))
        self.weights = nn.Parameter(0.1 * torch.randn(n_layers, n_qubits, 3))
        self.head = nn.Linear(n_qubits, 1)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def correction(self, x):
        angles = self.encoder(x) * np.pi
        z = torch.stack(list(self.circuit(angles, self.in_scale, self.weights)), dim=1)
        return self.scale * self.head(z).squeeze(-1)

def fit_frozen_lr(X_tr, y_tr):
    lr = LogisticRegression(max_iter=10000, C=1.0).fit(X_tr, y_tr)
    return lr

def with_frozen_logit(X, logit):
    return np.concatenate([X, logit[:, None]], axis=1)

def lr_logits(lr, X):
    return lr.decision_function(X)

OUT_DIR = "quantum_residual_results"
THRESHOLD = 0.5
MODEL_FILES = ["quantum_residual.pt", "frozen_lr.joblib", "model_config.json"]


def saved_model_exists(out_dir=OUT_DIR):
    return all(os.path.exists(os.path.join(out_dir, f)) for f in MODEL_FILES)


def load_saved_model(d_in, out_dir=OUT_DIR):
    with open(os.path.join(out_dir, "model_config.json")) as f:
        cfg = json.load(f)
    if cfg["d_in"] != d_in:
        raise ValueError(f"Saved model expects {cfg['d_in']} input features, "
                         f"but the loaded data has {d_in}. Delete {out_dir}/ and retrain.")

    frozen_lr = joblib.load(os.path.join(out_dir, "frozen_lr.joblib"))
    corr = QuantumResidualCorrection(cfg["d_in"], n_qubits=cfg["n_qubits"],
                                     n_layers=cfg["n_layers"], hidden=tuple(cfg["hidden"]))
    model = FrozenBaselinePlusCorrection(corr).double()
    model.load_state_dict(torch.load(os.path.join(out_dir, "quantum_residual.pt"),
                                     map_location="cpu"))
    model.eval()
    return frozen_lr, model

def fit_and_log(model, X_tr, y_tr, X_val, y_val, epochs, patience, lr, batch_size, weight_decay):
    dtype = next(model.parameters()).dtype
    loader = DataLoader(
        TensorDataset(torch.tensor(X_tr, dtype=dtype), torch.tensor(y_tr, dtype=dtype)),
        batch_size=batch_size, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5,
                                                       patience=max(3, patience // 4))
    loss_fn = nn.BCEWithLogitsLoss()
    corr_params = list(model.correction_module.parameters())

    Xv = torch.tensor(X_val, dtype=dtype)
    yv = torch.tensor(y_val, dtype=dtype)

    best_auc, best_state, best_epoch, wait = -1.0, None, 0, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses, grad_norms = [], []
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            grad_norms.append(np.mean([p.grad.abs().mean().item()
                                       for p in corr_params if p.grad is not None]))
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), yv).item()
        val_proba = predict_proba_torch(model, X_val)
        val_auc = roc_auc_score(y_val, val_proba)
        sched.step(val_auc)
        scale = model.correction_module.scale.item()

        history.append({"epoch": epoch, "train_loss": np.mean(train_losses),
                        "val_loss": val_loss, "val_auc": val_auc, "scale": scale,
                        "mean_abs_grad": np.mean(grad_norms)})
        print(f"  epoch {epoch:3d} | train_loss {np.mean(train_losses):.4f} | "
             f"val_loss {val_loss:.4f} | val_auc {val_auc:.4f} | scale {scale:+.4f}")

        if val_auc > best_auc:
            best_auc, best_epoch, wait = val_auc, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                print(f"  early stop at epoch {epoch} (best val AUC {best_auc:.4f} "
                     f"at epoch {best_epoch})")
                break

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history), best_epoch


def compute_metrics(y_true, y_proba, threshold=THRESHOLD):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
    }


def plot_training_diagnostics(history: pd.DataFrame, best_epoch: int, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    ax.plot(history["epoch"], history["train_loss"], label="train loss")
    ax.plot(history["epoch"], history["val_loss"], label="val loss")
    ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1, label="best epoch (kept)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("BCE loss"); ax.set_title("Loss curves")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(history["epoch"], history["val_auc"], color="tab:green")
    ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation ROC-AUC"); ax.set_title("Validation AUC")

    ax = axes[1, 0]
    ax.plot(history["epoch"], history["scale"], color="tab:purple")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Epoch"); ax.set_ylabel("scale (alpha)")
    ax.set_title("Correction Scale")

    ax = axes[1, 1]
    ax.plot(history["epoch"], history["mean_abs_grad"], color="tab:red")
    ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Mean |gradient|")
    ax.set_title("Correction term gradient magnitude")
    ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_roc(y_true, y_proba, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"QuantumResidualBoost (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve"); plt.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.close()


def plot_pr(y_true, y_proba, out_path):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    chance = y_true.mean()
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f"QuantumResidualBoost (AP={ap:.3f})")
    plt.axhline(chance, color="k", linestyle="--", linewidth=1, label=f"Chance ({chance:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Curve"); plt.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.close()


def plot_confusion(y_true, y_proba, out_path, threshold=THRESHOLD):
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.xticks([0, 1], ["No CVD", "CVD"]); plt.yticks([0, 1], ["No CVD", "CVD"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.close()


def plot_probability_histogram(y_true, y_proba, out_path, threshold=THRESHOLD):
    plt.figure(figsize=(7, 5))
    plt.hist(y_proba[y_true == 0], bins=30, alpha=0.6, label="No CVD (true)", density=True)
    plt.hist(y_proba[y_true == 1], bins=30, alpha=0.6, label="CVD (true)", density=True)
    plt.axvline(threshold, color="black", linestyle="--", linewidth=1, label=f"threshold={threshold}")
    plt.xlabel("Predicted probability"); plt.ylabel("Density")
    plt.title("Predicted Probability Distribution by True Class")
    plt.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.close()



def main(force_retrain=False):
    os.makedirs(OUT_DIR, exist_ok=True)
    X_train, X_test, y_train, y_test, names = load_data()
    tr, va = make_validation_split(y_train)
    d = X_train.shape[1]
    print(f"Train rows: {len(tr):,} | Val rows: {len(va):,} | Test rows: {len(y_test):,}\n")

    history, best_epoch = None, None

    if saved_model_exists() and not force_retrain:
        print(f"Found a saved model in ./{OUT_DIR}/ — loading it instead of retraining.")
        print("(pass --retrain to ignore this and train from scratch)\n")
        frozen_lr, model = load_saved_model(d)

        history_path = os.path.join(OUT_DIR, "training_history.csv")
        if os.path.exists(history_path):
            history = pd.read_csv(history_path)
            best_epoch = int(history.loc[history["val_auc"].idxmax(), "epoch"])
        else:
            print("No training_history.csv found alongside the saved model — "
                 "skipping training_diagnostics.png this run.\n")
    else:
        frozen_lr = fit_frozen_lr(X_train[tr], y_train[tr])
        logit_tr = lr_logits(frozen_lr, X_train[tr])
        logit_va = lr_logits(frozen_lr, X_train[va])

        Xtr_aug = with_frozen_logit(X_train[tr], logit_tr)
        Xva_aug = with_frozen_logit(X_train[va], logit_va)

        torch.manual_seed(SEED)
        corr = QuantumResidualCorrection(d)
        model = FrozenBaselinePlusCorrection(corr).double()
        print("Training QuantumResidualCorrection...")
        model, history, best_epoch = fit_and_log(
            model, Xtr_aug, y_train[tr], Xva_aug, y_train[va], **TRAIN_CFG)
        history.to_csv(os.path.join(OUT_DIR, "training_history.csv"), index=False)

        joblib.dump(frozen_lr, os.path.join(OUT_DIR, "frozen_lr.joblib"))
        torch.save(model.state_dict(), os.path.join(OUT_DIR, "quantum_residual.pt"))
        with open(os.path.join(OUT_DIR, "model_config.json"), "w") as f:
            json.dump({"d_in": d, "n_qubits": N_QUBITS, "n_layers": N_LAYERS,
                      "hidden": list(HIDDEN)}, f, indent=2)

    logit_te = lr_logits(frozen_lr, X_test)
    Xte_aug = with_frozen_logit(X_test, logit_te)
    test_proba = predict_proba_torch(model, Xte_aug)
    metrics = compute_metrics(y_test, test_proba)
    print("Test metrics:", {k: round(v, 4) for k, v in metrics.items()})
    pd.DataFrame([metrics]).round(4).to_csv(os.path.join(OUT_DIR, "metrics.csv"), index=False)

    if history is not None:
        plot_training_diagnostics(history, best_epoch,
                                  os.path.join(OUT_DIR, "training_diagnostics.png"))
    plot_roc(y_test, test_proba, os.path.join(OUT_DIR, "roc_curve.png"))
    plot_pr(y_test, test_proba, os.path.join(OUT_DIR, "pr_curve.png"))
    plot_confusion(y_test, test_proba, os.path.join(OUT_DIR, "confusion_matrix.png"))
    plot_probability_histogram(y_test, test_proba, os.path.join(OUT_DIR, "probability_histogram.png"))

    print(f"\nAll graphs, metrics, and the saved model are in ./{OUT_DIR}/")
    print("Use load_quantum_residual.py to reload and predict without retraining.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true",
                   help="Force retraining even if a saved model already exists.")
    args = ap.parse_args()
    main(force_retrain=args.retrain)