from fastapi import APIRouter, File, UploadFile, HTTPException
import torch
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
        