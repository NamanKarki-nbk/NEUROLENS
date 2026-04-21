import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, f1_score
import torch.nn.functional as F
import cv2

from dataset import get_dataloader, train_transform
from vit_transformer.vit import get_model as get_vit, DEVICE
from efficientnet_b4.efficientb4 import get_model as get_eff


# ----------------------------
# Setup
# ----------------------------
FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

EFF_PATH = Path(r"F:\Naman\NeuroLens\models\efficientnet_models\best_model_efficientnet.pth")
VIT_PATH = Path(r"F:\Naman\NeuroLens\models\vit_models\final_best_model_vit.pth")

TEST_PATH = Path(r"F:\Naman\NeuroLens\data\augmented\test")

class_names = ["glioma", "meningioma", "notumor", "pituitary"]

test_loader = get_dataloader(
    TEST_PATH,
    transforms=train_transform,
    shuffle=False
)

sns.set_style("whitegrid")


# ----------------------------
# Load models
# ----------------------------
def load_models():
    eff = get_eff(dropout=0.3119081840851825)
    eff.load_state_dict(torch.load(EFF_PATH, map_location=DEVICE))
    eff = eff.to(DEVICE).eval()

    vit = get_vit(dropout=0.3310474675875413)
    vit.load_state_dict(torch.load(VIT_PATH, map_location=DEVICE))
    vit = vit.to(DEVICE).eval()

    return eff, vit


# ----------------------------
# Predictions
# ----------------------------
def get_predictions(model):
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    return np.array(y_true), np.array(y_pred)


# ----------------------------
# Confusion Matrix
# ----------------------------
def confusion_mtx(y_true, y_pred, labels, title, name):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )

    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=300)
    plt.close()


# ----------------------------
# Model comparison
# ----------------------------
def model_comparison(metrics):

    models = list(metrics.keys())

    acc = [metrics[m]["acc"] for m in models]
    f1 = [metrics[m]["f1"] for m in models]

    x = np.arange(len(models))

    plt.figure(figsize=(6, 4))
    plt.bar(x - 0.15, acc, width=0.3, label="Accuracy")
    plt.bar(x + 0.15, f1, width=0.3, label="F1")

    plt.xticks(x, models)
    plt.ylim(0.95, 1.0)
    plt.legend()
    plt.title("Model Comparison")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "model_comparison.png", dpi=300)
    plt.close()


# ----------------------------
# Grad-CAM (EfficientNet only)
# ----------------------------
class GradCAM:
    def __init__(self, model, layer):
        self.model = model
        self.grad = None
        self.act = None

        layer.register_forward_hook(self.forward_hook)
        layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, m, i, o):
        self.act = o

    def backward_hook(self, m, gi, go):
        self.grad = go[0]

    def generate(self, img):
        self.model.zero_grad()

        out = self.model(img)

        cls = out.argmax(dim=1).item()
        out[0, cls].backward()

        weights = self.grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.act).sum(1)
        cam = F.relu(cam)

        cam = cam.squeeze().detach().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() + 1e-8)

        return cam


# ----------------------------
# MAIN PIPELINE
# ----------------------------
def main():

    eff, vit = load_models()

    print("Running inference...")

    y_true_eff, y_pred_eff = get_predictions(eff)
    y_true_vit, y_pred_vit = get_predictions(vit)

    # Confusion matrices
    confusion_mtx(
        y_true_eff, y_pred_eff,
        class_names,
        "EfficientNet Confusion Matrix",
        "cm_effnet.png"
    )

    confusion_mtx(
        y_true_vit, y_pred_vit,
        class_names,
        "ViT Confusion Matrix",
        "cm_vit.png"
    )

    # Metrics
    metrics = {
        "EfficientNet": {
            "acc": (y_true_eff == y_pred_eff).mean(),
            "f1": f1_score(y_true_eff, y_pred_eff, average="macro")
        },
        "ViT": {
            "acc": (y_true_vit == y_pred_vit).mean(),
            "f1": f1_score(y_true_vit, y_pred_vit, average="macro")
        }
    }

    model_comparison(metrics)

    print("\nReports generated at:", FIG_DIR)


if __name__ == "__main__":
    main()