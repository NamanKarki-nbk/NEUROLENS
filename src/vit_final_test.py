import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import json
from pathlib import Path

from dataset import get_dataloader, train_transform  # ✅ IMPORTANT
from vit_transformer.vit import get_model, DEVICE
from vit_eval import evaluate


# ✅ Use RELATIVE paths (DVC-compatible)
test_path = Path("data/augmented/test")
MODEL_PATH = Path("models/vit_models/final_best_model_vit.pth")
METRICS_PATH = Path("src/vit_final_test_metrics.json")


# ✅ Apply SAME transform as training (fixes shape issue)
test_loader = get_dataloader(
    test_path,
    transforms=train_transform,
    shuffle=False
)


# class weights (same as training)
class_weights = torch.tensor(
    [0.7032, 1.1267, 1.5025, 0.9756],
    dtype=torch.float32
).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)


def test():
    # ✅ Model init (must match training)
    model = get_model(dropout=0.3310474675875413)

    # ✅ Safe loading (future-proof)
    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )
    model.load_state_dict(state_dict)

    model = model.to(DEVICE)
    model.eval()

    # ✅ Evaluation
    test_loss, test_acc, test_f1 = evaluate(
        model,
        test_loader,
        criterion,
        DEVICE
    )

    print("\n===== FINAL ViT TEST RESULTS =====")
    print(f"Loss     : {test_loss:.4f}")
    print(f"Accuracy : {test_acc:.4f}")
    print(f"F1 Score : {test_f1:.4f}")

    # ✅ Save metrics (matches DVC exactly)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(METRICS_PATH, "w") as f:
        json.dump({
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
            "test_f1": float(test_f1)
        }, f, indent=4)

    print("✓ Metrics saved to src/vit_final_test_metrics.json")


if __name__ == "__main__":
    test()