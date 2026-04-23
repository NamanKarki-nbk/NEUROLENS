import sys
sys.path.append(r"F:\Naman\NeuroLens")
from fastapi import APIRouter, File, UploadFile, HTTPException
import torch
import torch.nn.functional as F
from utils.preprocess import preprocess
from model_loader import MODEL_REGISTRY
from src.rag.rag_pipeline import run_rag
from src.efficientnet_b4.efficientb4 import DEVICE
router = APIRouter()

CLASS_NAMES = {
    0: "Glioma",
    1: "Meningioma",
    2: "No Tumor",
    3: "Pituitary Tumor"
}


# =====================
# Single Model Prediction
# =====================
@router.post("/analyze-image/{model}")
async def analyze_image(model: str, file: UploadFile = File(...)):
    model_obj = MODEL_REGISTRY.get(model)

    if model_obj is None:
        raise HTTPException(status_code=400, detail=f"Invalid model name: {model}. Choose 'efficientnet' or 'vit'")

    image = preprocess(file)

    with torch.no_grad():
        output = model_obj(image)
        prediction_idx = torch.argmax(output, dim=1).item()
        probabilities = torch.softmax(output, dim=1).squeeze().tolist()

    return {
        "model": model,
        "prediction_idx": prediction_idx,
        "prediction": CLASS_NAMES[prediction_idx],
        "confidence": round(probabilities[prediction_idx] * 100, 2),
        "probabilities": {
            CLASS_NAMES[i]: round(p * 100, 2)
            for i, p in enumerate(probabilities)
        }
    }


# =====================
# Single Model + RAG Explanation
# =====================
@router.post("/analyze-and-explain/{model}")
async def analyze_and_explain(model: str, file: UploadFile = File(...)):
    model_obj = MODEL_REGISTRY.get(model)

    if model_obj is None:
        raise HTTPException(status_code=400, detail=f"Invalid model name: {model}")

    image = preprocess(file)

    with torch.no_grad():
        output = model_obj(image)
        prediction_idx = torch.argmax(output, dim=1).item()
        probabilities = torch.softmax(output, dim=1).squeeze().tolist()

    # ── RAG explanation ───────────────────────────────────────────
    try:
        explanation = run_rag(prediction_idx)
    except Exception as e:
        explanation = f"Explanation unavailable: {str(e)}"

    return {
        "model": model,
        "prediction_idx": prediction_idx,
        "prediction": CLASS_NAMES[prediction_idx],
        "confidence": round(probabilities[prediction_idx] * 100, 2),
        "probabilities": {
            CLASS_NAMES[i]: round(p * 100, 2)
            for i, p in enumerate(probabilities)
        },
        "explanation": explanation
    }


# =====================
# Ensemble Voting
# =====================
@router.post("/ensemble-voting")
async def ensemble(
    file: UploadFile = File(...),
    w_eff: float = 0.45,
    w_vit: float = 0.55
):
    eff_model = MODEL_REGISTRY["Efficient_net"]
    vit_model = MODEL_REGISTRY["VIT"]

    if eff_model is None or vit_model is None:
        raise HTTPException(status_code=400, detail="Models are not initialized")

    image = preprocess(file)

    with torch.no_grad():
        out_eff = eff_model(image)
        out_vit = vit_model(image)

        logits = w_eff * out_eff + w_vit * out_vit
        prediction_idx = torch.argmax(torch.softmax(logits, dim=1), dim=1).item()
        probabilities = torch.softmax(logits, dim=1).squeeze().tolist()

    return {
        "model": "ensemble",
        "prediction_idx": prediction_idx,
        "prediction": CLASS_NAMES[prediction_idx],
        "confidence": round(probabilities[prediction_idx] * 100, 2),
        "probabilities": {
            CLASS_NAMES[i]: round(p * 100, 2)
            for i, p in enumerate(probabilities)
        }
    }


# =====================
# Ensemble + RAG Explanation
# =====================
@router.post("/ensemble-and-explain")
async def ensemble_and_explain(
    file: UploadFile = File(...),
    w_eff: float = 0.45,
    w_vit: float = 0.55
):
    eff_model = MODEL_REGISTRY["Efficient_net"]
    vit_model = MODEL_REGISTRY["VIT"]

    if eff_model is None or vit_model is None:
        raise HTTPException(status_code=400, detail="Models are not initialized")

    image = preprocess(file)

    with torch.no_grad():
        out_eff = eff_model(image)
        out_vit = vit_model(image)

        logits = w_eff * out_eff + w_vit * out_vit
        prediction_idx = torch.argmax(torch.softmax(logits, dim=1), dim=1).item()
        probabilities = torch.softmax(logits, dim=1).squeeze().tolist()

    # ── RAG explanation ───────────────────────────────────────────
    try:
        explanation = run_rag(prediction_idx)
    except Exception as e:
        explanation = f"Explanation unavailable: {str(e)}"

    return {
        "model": "ensemble",
        "prediction_idx": prediction_idx,
        "prediction": CLASS_NAMES[prediction_idx],
        "confidence": round(probabilities[prediction_idx] * 100, 2),
        "probabilities": {
            CLASS_NAMES[i]: round(p * 100, 2)
            for i, p in enumerate(probabilities)
        },
        "explanation": explanation
    }

