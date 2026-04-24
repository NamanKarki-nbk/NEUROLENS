import sys
sys.path.append(r"F:\Naman\NeuroLens")

from fastapi import APIRouter, File, UploadFile, HTTPException
import torch
import torch.nn.functional as F
from utils.preprocess import preprocess
from model_loader import MODEL_REGISTRY
from src.rag.rag_pipeline import run_rag

# ── QnA imports ─────────────────────────────────────
from pydantic import BaseModel
from src.rag.retriever import HybridRetriever
from src.rag.reranker import rerank
from src.rag.query_rewriter import rewrite_query
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

router = APIRouter()

# ── Class Labels ────────────────────────────────────
CLASS_NAMES = {
    0: "Glioma",
    1: "Meningioma",
    2: "No Tumor",
    3: "Pituitary Tumor"
}

# ── QnA Setup ───────────────────────────────────────
_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.4
)

_qna_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a knowledgeable and compassionate medical assistant specializing in brain tumors.

Relevant Medical Information:
{context}

Patient's Question:
{question}

Instructions:
- Answer clearly in simple language
- Keep under 200 words
- Be empathetic
- Do NOT mention context or documents
"""
)

_qna_chain = _qna_prompt | _llm | StrOutputParser()
_retriever = HybridRetriever()


class QnARequest(BaseModel):
    message: str


class QnAResponse(BaseModel):
    reply: str


# ===================================================
# Image Analysis + Explanation
# ===================================================
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

    # ── RAG Explanation (FIXED) ──────────────────────
    try:
        explanation_raw = run_rag(prediction_idx)

        # HARD CLEANING
        if isinstance(explanation_raw, str):
            explanation = explanation_raw.strip()

        elif isinstance(explanation_raw, dict):
            explanation = explanation_raw.get("answer") or explanation_raw.get("result") or str(explanation_raw)

        elif isinstance(explanation_raw, list):
            explanation = explanation_raw[0] if len(explanation_raw) > 0 else "No explanation available"

        else:
            explanation = str(explanation_raw)

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


# ===================================================
# QnA Endpoint
# ===================================================
@router.post("/qna", response_model=QnAResponse)
async def qna(request: QnARequest):
    question = request.message.strip()

    if not question:
        return QnAResponse(reply="Please ask a question about brain tumors.")

    queries = rewrite_query(question)

    all_docs = []
    for q in queries:
        docs = _retriever.retrieve(q, top_k=6)
        all_docs.extend(docs)

    # remove duplicates
    seen = set()
    unique_docs = []
    for doc in all_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)

    final_docs = rerank(question, unique_docs, top_k=4)
    context = "\n\n".join(doc.page_content for doc in final_docs)

    result = _qna_chain.invoke({
        "context": context,
        "question": question
    })

    # ensure string
    if isinstance(result, str):
        reply = result.strip()
    else:
        reply = str(result)

    return QnAResponse(reply=reply)