import torch
import torch.nn as nn
import yaml
from torch.amp import autocast, GradScaler
from pathlib import Path
from dataset import get_dataloader, train_transform
from efficientnet_b4.efficientb4 import get_model, unfreeze_last_n_blocks, DEVICE
from eval import evaluate
import wandb
import warnings
import json

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Paths (RELATIVE - DVC SAFE)
# ─────────────────────────────────────────────
MODEL_DIR = Path("models/efficientnet_models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

train_set = Path("data/augmented/train")
val_set = Path("data/augmented/val")

# DataLoaders
train_loader = get_dataloader(train_set, transforms=train_transform, shuffle=True)
val_loader = get_dataloader(val_set, transforms=train_transform, shuffle=False)

# ─────────────────────────────────────────────
# Focal Loss
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# Training Function
# ─────────────────────────────────────────────
def train():
    # Load params (FROM ROOT)
    with open("params.yaml") as f:
        p = yaml.safe_load(f)["train"]

    lr = p["lr"]
    dropout = p["dropout"]
    gamma = p["gamma"]
    loss_type = p["loss_type"]
    unfreeze_n = p["unfreeze_n"]
    epochs = p["epochs"]

    # ── WandB Init ────────────────────────────
    wandb.init(
        project="neurolens-efficientnet",
        name="final_run",
        config=p
    )

    # ── Model ────────────────────────────────
    model = get_model(dropout=dropout)
    model.to(DEVICE)

    unfreeze_last_n_blocks(model, unfreeze_n)

    # Freeze BatchNorm layers
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    # ── Class Weights ────────────────────────
    class_weights = torch.tensor(
        [0.7032, 1.1267, 1.5025, 0.9756],
        dtype=torch.float32
    ).to(DEVICE)

    # ── Loss ────────────────────────────────
    if loss_type == "ce":
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = FocalLoss(alpha=class_weights, gamma=gamma)

    # ── Optimizer ───────────────────────────
    optimizer = torch.optim.Adam(
        [
            {"params": model.classifier.parameters(), "lr": lr},
            {"params": model.features.parameters(), "lr": lr / 10}
        ]
    )

    # ── Mixed Precision ─────────────────────
    scaler = GradScaler()

    best_val_f1 = 0.0

    # ───────────────────────────────────────
    # Training Loop
    # ───────────────────────────────────────
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            with autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        # Validation
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, DEVICE)
        train_loss /= len(train_loader)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Acc: {val_acc:.4f} | "
            f"F1: {val_f1:.4f}"
        )

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1
        })

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                model.state_dict(),
                MODEL_DIR / "best_model_efficientnet.pth"
            )
            print(f"✅ New best model saved with F1: {best_val_f1:.4f}")

    # ── Save metrics for DVC ───────────────
    metrics = {
        "best_val_f1": float(best_val_f1),
        "final_val_loss": float(val_loss),
        "final_val_acc": float(val_acc)
    }

    Path("src").mkdir(exist_ok=True)

    with open("src/train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\n✓ Training complete! Best Val F1: {best_val_f1:.4f}")

    wandb.finish()


if __name__ == "__main__":
    train()