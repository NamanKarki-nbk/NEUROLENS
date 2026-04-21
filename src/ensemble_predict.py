import torch
import torch.nn.functional as F
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from dataset import get_dataloader, train_transform
from vit_transformer.vit import get_model as get_vit, DEVICE
from efficientnet_b4.efficientb4 import get_model as get_eff


# ----------------------------
# Load models
# ----------------------------
def load_models():

    eff = get_eff(dropout=0.3119)
    eff.load_state_dict(torch.load(
        "models/efficientnet_models/best_model_efficientnet.pth",
        map_location=DEVICE
    ))
    eff = eff.to(DEVICE).eval()

    vit = get_vit(dropout=0.3310)
    vit.load_state_dict(torch.load(
        "models/vit_models/final_best_model_vit.pth",
        map_location=DEVICE
    ))
    vit = vit.to(DEVICE).eval()

    return eff, vit


# ----------------------------
# Ensemble prediction
# ----------------------------
def ensemble_predict(eff, vit, loader, w_eff=0.45, w_vit=0.55):

    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in loader:

            images = images.to(DEVICE)

            out_eff = eff(images)
            out_vit = vit(images)

            prob_eff = F.softmax(out_eff, dim=1)
            prob_vit = F.softmax(out_vit, dim=1)

            probs = w_eff * prob_eff + w_vit * prob_vit

            preds = torch.argmax(probs, dim=1)

            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    return np.array(y_true), np.array(y_pred)


# ----------------------------
# MAIN
# ----------------------------
def main():

    test_loader = get_dataloader(
        "data/augmented/test",
        transforms=train_transform,
        shuffle=False
    )

    eff, vit = load_models()

    y_true, y_pred = ensemble_predict(eff, vit, test_loader)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    print("\n===== ENSEMBLE RESULTS =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()