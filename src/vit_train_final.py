import torch
import torch.nn as nn
import yaml
from torch.amp import autocast, GradScaler 
from dataset import get_dataloader, train_transform
from vit_transformer.vit import get_model, unfreeze_last_n_blocks, DEVICE
from torch.optim.lr_scheduler import CosineAnnealingLR
from vit_eval import evaluate
from pathlib import Path
import json
import wandb

# ✅ Use relative path (important for DVC)
MODEL_DIR = Path("models/vit_models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

train_set = Path("data/augmented/train")
val_set = Path("data/augmented/val")

train_loader = get_dataloader(train_set, transforms=train_transform)
val_loader = get_dataloader(val_set, transforms=train_transform, shuffle=False)


def train():

    # ✅ Correct path (matches DVC deps)
    with open("VIT_best_params.yaml") as f:
        p = yaml.safe_load(f)["train"]

    lr = p["lr"]
    dropout = p["dropout"]
    unfreeze_n = p["unfreeze_n"]
    epochs = p["epochs"]

    wandb.init(
        project="neurolens-vit",
        name="vit_final_train",
        config=p
    )

    model = get_model(dropout)
    unfreeze_last_n_blocks(model, unfreeze_n)

    class_weights = torch.tensor(
        [0.7032, 1.1267, 1.5025, 0.9756],
        dtype=torch.float32
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    params = [{"params": model.heads.parameters(), "lr": lr}]

    if unfreeze_n > 0:
        params.append({
            "params": model.encoder.layers[-unfreeze_n:].parameters(),
            "lr": lr / 10
        })

    optimizer = torch.optim.AdamW(params)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    scaler = GradScaler()

    # ✅ FIX: ensure model is always saved at least once
    best_val_f1 = -float("inf")
    best_val_acc = 0.0
    best_val_loss = float("inf")

    # ---------------- TRAIN LOOP ----------------
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            with autocast(device_type=DEVICE):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        val_loss, val_accuracy, val_f1 = evaluate(model, val_loader, criterion, DEVICE)

        print(
            f"Epoch {epoch+1} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Acc: {val_accuracy:.4f} | "
            f"F1: {val_f1:.4f}"
        )

        wandb.log({
            "epoch": epoch + 1,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_f1": val_f1
        })

        # ✅ Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_accuracy
            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                MODEL_DIR / "final_best_model_vit.pth"
            )

            print(f"✓ Best model saved (F1: {best_val_f1:.4f})")

        scheduler.step()

    # ✅ FIX: match DVC metrics path EXACTLY
    with open("src/vit_final_train_metrics.json", "w") as f:
        json.dump({
            "best_val_f1": float(best_val_f1),
            "best_val_acc": float(best_val_acc),
            "best_val_loss": float(best_val_loss)
        }, f, indent=4)

    print(f"\n✓ Training complete! Best Val F1: {best_val_f1:.4f}")

    wandb.finish()


if __name__ == "__main__":
    train()