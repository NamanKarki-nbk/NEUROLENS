import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path



DATA_DIR = Path(R"F:\Naman\NeuroLens\data\augmented")
NUM_CLASSES = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_model(dropout):
    model = models.vit_b_32(weights = models.ViT_B_32_Weights.IMAGENET1K_V1)
    
    #freeze all the layers
    for params in model.parameters():
        params.requires_grad = False
        
    
    
    in_features = model.heads.head.in_features
    model.heads.head = nn.Sequential(
        nn.LayerNorm(in_features),
        nn.Dropout(dropout),
        nn.Linear(in_features, 512),
        nn.GELU(),  
        nn.Dropout(dropout/2),
        nn.Linear(512, NUM_CLASSES)
    )
    
    #unfreezing the classfication head
    for params in model.heads.parameters():
        params.requires_grad = True
        
    
    return model.to(DEVICE)



def unfreeze_last_n_blocks(model: nn.Module, n: int):

    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze head
    for param in model.heads.parameters():
        param.requires_grad = True

    # Get encoder blocks
    blocks = list(model.encoder.layers)
    n = min(n, len(blocks))

    # Unfreeze last N blocks
    for block in blocks[-n:]:
        for param in block.parameters():
            param.requires_grad = True

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")

    return model

