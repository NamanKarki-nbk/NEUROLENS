import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path

DATA_DIR = Path(r"F:\Naman\NeuroLens\data\augmented")
NUM_CLASSES = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_model(dropout):
    model = models.efficientnet_b4(weights= models.EfficientNet_B4_Weights.IMAGENET1K_V1)
    
    #freeze feature extractors
    for param in model.features.parameters():
        param.requires_grad = False
    
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(num_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(512, NUM_CLASSES)
    )
    return model.to(DEVICE)


def unfreeze_last_n_blocks(model: nn.Module, n: int):
    
    # Freeze all
    for param in model.features.parameters():
        param.requires_grad = False
        
    blocks = list(model.features.children())
    n = min(n, len(blocks))
    
    # Unfreeze last n blocks
    for block in blocks[-n:]:
        for param in block.parameters():
            param.requires_grad = True
    
    # Keep BatchNorm stable
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    
    # Count params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")
            