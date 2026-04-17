import torch
import torch.nn as nn
import json
from pathlib import Path

from dataset import get_dataloader, train_transform
from vit_transformer.vit import get_model, DEVICE
from vit_eval import evaluate


test_path = Path(r"F:\Naman\NeuroLens\data\augmented\test")
MODEL_PATH = Path(r"F:\Naman\NeuroLens\models\vit_models\final_best_model_vit.pth")
METRICS_PATH = Path("vit_test_metrics.json")

test_loader = get_dataloader(
    test_path,
    transforms=None,
    shuffle=False
)

class_weights = torch.tensor(
    [0.7032, 1.1267, 1.5025, 0.9756],
    dtype=torch.float32
).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)


def test():
    model = get_model(dropout=0.3310474675875413)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    test_loss, test_acc, test_f1 = evaluate(
        model,
        test_loader,
        criterion,
        DEVICE
    )

    print("\nTEST RESULTS")
    print(f"Loss     : {test_loss:.4f}")
    print(f"Accuracy : {test_acc:.4f}")
    print(f"F1 Score : {test_f1:.4f}")

    metrics = {
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "test_f1": float(test_f1)
    }

    Path("src").mkdir(exist_ok=True)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    test()