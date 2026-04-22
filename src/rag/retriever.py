from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
PERSIST_DIR = BASE_DIR / "data" / "chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class HybridRetriever:
    def __init__(self):
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            encode_kwargs={"normalize_embeddings": True}
        )

        # ── Dense retriever (ChromaDB) ────────────────────────────
        self.vectorstore = Chroma(
            collection_name="neurolens_docs",
            embedding_function=embeddings,
            persist_directory=str(PERSIST_DIR)
        )

        # ── Load all docs for BM25 ────────────────────────────────
        all_docs = self.vectorstore.get()
        self.all_chunks = all_docs["documents"]  # list of strings
        tokenized = [doc.split() for doc in self.all_chunks]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int = 10):
        # ── Dense search ──────────────────────────────────────────
        dense_results = self.vectorstore.similarity_search(query, k=top_k)

        # ── Sparse search (BM25) ──────────────────────────────────
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_indices = bm25_scores.argsort()[-top_k:][::-1]
        sparse_texts = [self.all_chunks[i] for i in top_indices]

        # ── Merge & deduplicate ───────────────────────────────────
        seen = set()
        merged = []

        for doc in dense_results:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                merged.append(doc)

        for text in sparse_texts:
            if text not in seen:
                seen.add(text)
                from langchain_core.documents import Document
                merged.append(Document(page_content=text))

        return merged[:top_k]