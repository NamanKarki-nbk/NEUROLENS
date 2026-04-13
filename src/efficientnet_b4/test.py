import torch
import torch.nn as nn
from pathlib import Path

from dataset import get_dataloader, train_transform
from efficientnet_b4.efficientb4 import get_model, DEVICE
from eval import evaluate


#Test dataset path
test_set = Path(r"F:\Naman\NeuroLens\data\augmented\test")

#DataLoader
test_loader = get_dataloader(
    test_set,
    transforms=train_transform,
    shuffle=False
)

# Class weights (same as training)
class_weights = torch.tensor(
    [0.7032, 1.1267, 1.5025, 0.9756],
    dtype=torch.float32
).to(DEVICE)

#  Loss (use CE for evaluation consistency)
criterion = nn.CrossEntropyLoss(weight=class_weights)


def test():
    # Load model
    model = get_model(dropout=0.3)  # use a reasonable default OR best param later
    model.load_state_dict(torch.load("models/efficientnet_models/best_model.pth", map_location=DEVICE))
    model = model.to(DEVICE)

    # Evaluate
    test_loss, test_acc, test_f1 = evaluate(
        model,
        test_loader,
        criterion,
        DEVICE
    )

    print("\n TEST RESULTS")
    print(f"Loss     : {test_loss:.4f}")
    print(f"Accuracy : {test_acc:.4f}")
    print(f"F1 Score : {test_f1:.4f}")


if __name__ == "__main__":
    test()