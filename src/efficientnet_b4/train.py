import torch
import torch.nn as nn
import optuna
from torch.amp import autocast, GradScaler
from pathlib import Path

from dataset import get_dataloader, train_transform
from efficientnet_b4.efficientb4 import get_model, unfreeze_last_n_blocks, DEVICE
from eval import evaluate

MODEL_DIR = Path("models/efficientnet_models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# Dataset paths
train_set = Path(r"F:\Naman\NeuroLens\data\augmented\train")
val_set = Path(r"F:\Naman\NeuroLens\data\augmented\val")
test_set = Path(r"F:\Naman\NeuroLens\data\augmented\test")


#  DataLoaders
train_loader = get_dataloader(train_set, transforms=train_transform, shuffle=True)
val_loader = get_dataloader(val_set, transforms=train_transform, shuffle=False)


# Focal Loss
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


#  Global best tracking (F1)
BEST_F1 = 0


# Optuna Objective
def objective(trial):
    global BEST_F1

    # Hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    unfreeze_n = trial.suggest_int("unfreeze_n", 1, 6)
    loss_type = trial.suggest_categorical("loss_type", ["ce", "focal"])

    # Model
    model = get_model(dropout)
    unfreeze_last_n_blocks(model, unfreeze_n)

    #  Class weights
    class_weights = torch.tensor(
        [0.7032, 1.1267, 1.5025, 0.9756],
        dtype=torch.float32
    ).to(DEVICE)

    # Loss function
    if loss_type == "ce":
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        gamma = trial.suggest_float("gamma", 1.5, 3.0)
        criterion = FocalLoss(alpha=class_weights, gamma=gamma)

    # Optimizer
    optimizer = torch.optim.Adam([
        {"params": model.classifier.parameters(), "lr": lr},
        {"params": model.features.parameters(), "lr": lr / 10}
    ])

    #  AMP scaler (NEW API requires device)
    scaler = GradScaler(device=DEVICE.type)

    best_val_f1 = 0

    # Training loop
    for epoch in range(10):
        model.train()

        # Keep BatchNorm stable
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        # Training phase
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            #  Mixed Precision (correct usage)
            with autocast(device_type=DEVICE.type):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Validation AFTER epoch (correct placement)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, DEVICE)

        print(
            f"Trial {trial.number} | Epoch {epoch+1} | "
            f"Acc: {val_acc:.4f} | F1: {val_f1:.4f}"
        )

        #  Best in this trial
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1

        #  Best across ALL trials
        if val_f1 > BEST_F1:
            BEST_F1 = val_f1
            torch.save(model.state_dict(), MODEL_DIR / "best_model.pth")
            print(f"New BEST model saved! F1: {BEST_F1:.4f}")

    return best_val_f1


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    print("\nBest Params:")
    print(study.best_params)

    print(f"\nBest F1 Score: {BEST_F1:.4f}")