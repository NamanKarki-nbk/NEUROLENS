import sys
sys.path.append(r"F:\Naman\NeuroLens")
from src.vit_transformer.vit import DEVICE, get_model as get_vit
from src.efficientnet_b4.efficientb4 import get_model as get_eff
from pathlib import Path
import torch


EFF_PATH = Path(r"F:\Naman\NeuroLens\models\efficientnet_models\best_model_efficientnet.pth")
VIT_PATH = Path(r"F:\Naman\NeuroLens\models\vit_models\final_best_model_vit.pth")

MODEL_REGISTRY={}

def load_model():
    
    #LOADING EFFICIENT NET
    eff_model = get_eff(dropout=0.3119)
    eff_model.load_state_dict(torch.load(EFF_PATH, map_location=DEVICE))
    eff_model = eff_model.to(DEVICE).eval()
    
    #Loading VIT 
    vit_model = get_vit(dropout=0.3310)
    vit_model.load_state_dict(torch.load(VIT_PATH, map_location=DEVICE))
    vit_model = vit_model.to(DEVICE).eval()
    
    return eff_model, vit_model 


def init_models():
    eff_model, vit_model = load_model()
    
    MODEL_REGISTRY["Efficient_net"] = eff_model
    MODEL_REGISTRY["VIT"]= vit_model
    print("Models loaded and registered successfully")