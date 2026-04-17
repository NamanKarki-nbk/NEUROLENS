import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import wandb
import optuna
import yaml
import warnings

from pathlib import Path
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler

from vit_transformer.vit import get_model, unfreeze_last_n_blocks, DEVICE
from dataset import get_dataloader, train_transform
from vit_eval import evaluate

warnings.filterwarnings("ignore")

MODEL_DIR = Path(r"F:\Naman\NeuroLens\models\vit_models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 10
BEST_F1 = 0

# Dataset
train_set = Path(r"F:\Naman\NeuroLens\data\augmented\train")
val_set = Path(r"F:\Naman\NeuroLens\data\augmented\val")

train_loader = get_dataloader(train_set, transforms=train_transform)
val_loader = get_dataloader(val_set, shuffle=False, transforms=train_transform)


def objective(trial):
    global BEST_F1

    # Hyperparams
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    unfreeze_n = trial.suggest_int("unfreeze_n", 0, 6)

    wandb.init(
        project="neurolens-vit",
        name=f"trial_{trial.number}",
        config={
            "lr": lr,
            "dropout": dropout,
            "unfreeze_n": unfreeze_n
        },
        reinit=True
    )

    # Model
    model = get_model(dropout)
    unfreeze_last_n_blocks(model, unfreeze_n)

    # Loss
    class_weights = torch.tensor(
        [0.7032, 1.1267, 1.5025, 0.9756],
        dtype=torch.float32
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer (FIXED unfreeze_n=0 bug)
    params = [{"params": model.heads.parameters(), "lr": lr}]

    if unfreeze_n > 0:
        params.append({
            "params": model.encoder.layers[-unfreeze_n:].parameters(),
            "lr": lr / 10
        })

    optimizer = torch.optim.AdamW(params)

    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0)
    scaler = GradScaler()

    best_val_f1 = 0

    # 🔥 Training loop
    for epoch in range(EPOCHS):
        model.train()

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            with autocast(device_type=DEVICE):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # 🔥 Validation INSIDE epoch loop
        avg_loss, accuracy, f1 = evaluate(model, val_loader, criterion, DEVICE)

        print(
            f"Epoch {epoch+1} | Val Loss: {avg_loss:.4f} | "
            f"Acc: {accuracy:.4f} | F1: {f1:.4f}"
        )

        wandb.log({
            "epoch": epoch + 1,
            "val_loss": avg_loss,
            "val_accuracy": accuracy,
            "val_f1": f1
        })

        # Optuna pruning
        trial.report(f1, epoch)
        if trial.should_prune():
            wandb.finish()
            raise optuna.exceptions.TrialPruned()

        # Save best (trial)
        if f1 > best_val_f1:
            best_val_f1 = f1

        # Save global best
        if f1 > BEST_F1:
            BEST_F1 = f1
            torch.save(model.state_dict(), MODEL_DIR / "vit_best_model.pth")
            print(f"✓ New best model saved! F1: {BEST_F1:.4f}")

        scheduler.step()

    wandb.finish()
    return best_val_f1


if __name__ == "__main__":
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=3,
        n_warmup_steps=3
    )

    study = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        study_name="neurolens_vit"
    )

    study.optimize(objective, n_trials=5)

    print("\nOPTUNA SEARCH COMPLETE")
    print(f"Best Trial : {study.best_trial.number}")
    print(f"Best F1    : {BEST_F1:.4f}")
    print(f"Best Params: {study.best_params}")

    best = study.best_params.copy()
    best["epochs"] = EPOCHS

    with open("VIT_best_params.yaml", "w") as f:
        yaml.dump({"train": best}, f)

    print("\n✓ Best params saved")