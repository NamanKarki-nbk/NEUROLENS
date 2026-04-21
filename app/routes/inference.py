from fastapi import APIRouter, File, UploadFile, HTTPException
import torch
import torch.nn.functional as F
from utils.preprocess import preprocess
from model_loader import MODEL_REGISTRY




router = APIRouter()

@router.post("/analyze-image/{model}")
async def analyze_image(model:str, file: UploadFile = File(...)):
    model_obj = MODEL_REGISTRY.get(model)
    
    if model_obj is None:
        raise HTTPException(status_code=400, detail="Invalid model name")
    
    image = preprocess(file)
    
    with torch.no_grad():
        output = model_obj(image)
        prediction = torch.argmax(output, dim = 1).item()
    
    return {
        "model": model,
        "prediction": prediction
    }
    
    
@router.post("/ensemble-voting")
async def ensemble(file: UploadFile= File(...), w_eff:float=0.45, w_vit:float=0.55):
    
    eff_model = MODEL_REGISTRY["Efficient_net"]
    vit_model = MODEL_REGISTRY["VIT"]
    
    if eff_model is None or  vit_model is None:
        raise HTTPException(status_code=400, detail="Models are not initialized")
    
    image = preprocess(file)
    
    with torch.no_grad():
        out_eff = eff_model(image)
        out_vit = vit_model(image)
        
        logits = w_eff * out_eff + w_vit * out_vit
        pred = torch.argmax(torch.softmax(logits, dim=1), dim=1).item()
        return{
            "model": "ensemble",
            "prediction": pred
        }
        