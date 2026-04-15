import torch
import torch.nn as nn
import optuna
import yaml
from torch.amp import autocast, GradScaler
from pathlib import Path

from dataset import get_dataloader, train_transform
from efficientnet_b4.efficientb4 import get_model, unfreeze_last_n_blocks, DEVICE
from eval import evaluate

import wandb
import warnings
warnings.filterwarnings("ignore")


MODEL_DIR = Path(r"F:\Naman\NeuroLens\models\efficientnet_models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# =====================
# Dataset paths
# =====================
train_set = Path(r"F:\Naman\NeuroLens\data\augmented\train")
val_set = Path(r"F:\Naman\NeuroLens\data\augmented\val")

# =====================
# DataLoaders
# =====================
train_loader = get_dataloader(train_set, transforms=train_transform, shuffle=True)
val_loader = get_dataloader(val_set, transforms=train_transform, shuffle=False)


# =====================
# Focal Loss
# =====================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(
            logits, targets, reduction="none", weight=self.alpha
        )
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma) * ce_loss
        return loss.mean()


# =====================
# Global best tracking
# =====================
BEST_F1 = 0


# =====================
# Optuna Objective
# =====================
def objective(trial):
    global BEST_F1

    # ── Hyperparameters ──────────────────────────────────────────
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    unfreeze_n = trial.suggest_int("unfreeze_n", 1, 6)
    loss_type = trial.suggest_categorical("loss_type", ["ce", "focal"])

    gamma = None
    if loss_type == "focal":
        gamma = trial.suggest_float("gamma", 1.5, 3.0)

    # ── W&B init ─────────────────────────────────────────────────
    wandb.init(
        project="neurolens-efficientnet",
        name=f"trial_{trial.number}",
        config={
            "lr": lr,
            "dropout": dropout,
            "unfreeze_n": unfreeze_n,
            "loss_type": loss_type,
            "gamma": gamma,
        },
        reinit=True
    )

    # ── Model ────────────────────────────────────────────────────
    model = get_model(dropout)
    unfreeze_last_n_blocks(model, unfreeze_n)

    class_weights = torch.tensor(
        [0.7032, 1.1267, 1.5025, 0.9756],
        dtype=torch.float32
    ).to(DEVICE)

    # ── Loss ─────────────────────────────────────────────────────
    if loss_type == "ce":
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = FocalLoss(alpha=class_weights, gamma=gamma)

    # ── Optimizer ────────────────────────────────────────────────
    optimizer = torch.optim.Adam([
        {"params": model.classifier.parameters(), "lr": lr},
        {"params": model.features.parameters(), "lr": lr / 10}
    ])

    scaler = GradScaler(device=DEVICE)

    best_val_f1 = 0

    # ── Training loop ────────────────────────────────────────────
    for epoch in range(5):
        model.train()

        # Keep BatchNorm frozen
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        # Train
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            with autocast(device_type=DEVICE):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # ── Validation ───────────────────────────────────────────
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, DEVICE)

        print(
            f"Trial {trial.number} | Epoch {epoch+1}/10 | "
            f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
        )

        wandb.log({
            "epoch": epoch + 1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1
        })

        # ── Optuna pruning ───────────────────────────────────────
        trial.report(val_f1, epoch)
        if trial.should_prune():
            wandb.finish()
            raise optuna.exceptions.TrialPruned()

        # ── Track best within trial ──────────────────────────────
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1

        # ── Track global best & save model ──────────────────────
        if val_f1 > BEST_F1:
            BEST_F1 = val_f1
            torch.save(model.state_dict(), MODEL_DIR / "best_model.pth")
            print(f"  ✓ New best model saved! F1: {BEST_F1:.4f}")

    wandb.finish()
    return best_val_f1


# =====================
# Main
# =====================
if __name__ == "__main__":
    # Pruner kills bad trials early — saves time
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=3)

    study = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        study_name="neurolens-efficientnet"
    )

    study.optimize(objective, n_trials=5)

    # ── Results ──────────────────────────────────────────────────
    print("\n" + "="*50)
    print("OPTUNA SEARCH COMPLETE")
    print("="*50)
    print(f"Best Trial : {study.best_trial.number}")
    print(f"Best F1    : {BEST_F1:.4f}")
    print(f"Best Params: {study.best_params}")

    # ── Save best params → paste into params.yaml next ───────────
    best = study.best_params.copy()
    best["epochs"] = 10  # lock in epochs too

    # gamma is None for ce loss — remove it if not used
    if best.get("loss_type") == "ce":
        best.pop("gamma", None)

    with open("best_params.yaml", "w") as f:
        yaml.dump({"train": best}, f, default_flow_style=False)

    print("\n✓ Best params saved to best_params.yaml")
    print("  Copy its contents into params.yaml, then run: dvc repro")