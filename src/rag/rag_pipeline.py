from src.rag.retriever import HybridRetriever
from src.rag.reranker import rerank
from src.rag.query_rewriter import rewrite_query
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

CLASS_NAMES = {
    0: "Glioma",
    1: "Meningioma",
    2: "No Tumor",
    3: "Pituitary Tumor"
}

# ── Initialize retriever once at startup ─────────────────────────
retriever = HybridRetriever()


def run_rag(prediction_idx: int) -> str:
    prediction = CLASS_NAMES.get(prediction_idx, "Unknown")

    print(f"\n── RAG Pipeline ──────────────────────────")
    print(f"  Prediction: {prediction}")

    # ── No tumor case — no RAG needed ────────────────────────────
    if prediction_idx == 2:
        return (
            "Great news! No tumor was detected in your scan. "
            "Please continue with regular checkups as advised by your doctor."
        )

    # ── Step 1: Query rewriting ───────────────────────────────────
    queries = rewrite_query(prediction)

    # ── Step 2: Hybrid retrieval for each query ───────────────────
    all_docs = []
    for query in queries:
        docs = retriever.retrieve(query, top_k=8)
        all_docs.extend(docs)

    # Remove duplicates
    seen = set()
    unique_docs = []
    for doc in all_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)

    print(f"  Total unique docs retrieved: {len(unique_docs)}")

    # ── Step 3: Re-rank ───────────────────────────────────────────
    base_query = f"What is {prediction} and what should a patient know?"
    final_docs = rerank(base_query, unique_docs, top_k=5)

    # ── Step 4: Build context ─────────────────────────────────────
    context = "\n\n".join([doc.page_content for doc in final_docs])

    # ── Step 5: Generate patient explanation ─────────────────────
    prompt = f"""You are a compassionate medical assistant explaining a brain scan result to a patient in simple, clear language.

Diagnosis: {prediction}

Relevant Medical Information:
{context}

Instructions:
- Explain what {prediction} is in simple terms a patient can understand
- Mention common symptoms or what the patient might experience
- Explain what the next steps typically are
- Be empathetic, calm, and reassuring
- Do NOT use complex medical jargon
- Keep the response under 200 words
- Do NOT say "based on the context" or reference the documents"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )

    explanation = response.choices[0].message.content.strip()
    print(f"  ✓ Explanation generated")
    return explanation


if __name__ == "__main__":
    # Test the pipeline
    for idx in [0, 1, 2, 3]:
        print(f"\n{'='*50}")
        result = run_rag(idx)
        print(f"Response:\n{result}")