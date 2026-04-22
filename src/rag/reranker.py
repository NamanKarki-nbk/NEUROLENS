from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query: str, docs: list, top_k: int = 5) -> list:
    if not docs:
        return []

    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)

    scored_docs = sorted(
        zip(scores, docs),
        key=lambda x: x[0],
        reverse=True
    )

    print(f"  Top reranked score: {scored_docs[0][0]:.4f}")
    return [doc for _, doc in scored_docs[:top_k]]