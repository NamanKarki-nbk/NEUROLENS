import sys
sys.path.append(r"F:\Naman\NeuroLens")
import cv2
import numpy as np 
import torch 
from albumentations.pytorch import ToTensorV2
import albumentations as A
from src.vit_transformer.vit import DEVICE

transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    ToTensorV2()
])


def preprocess(file, transform = transform, device=DEVICE):
    contents = file.file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Invalid image file")
    
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    augmented = transform(image=image)
    image = augmented["image"]
    
    image = image.unsqueeze(0).to(device)
    return image