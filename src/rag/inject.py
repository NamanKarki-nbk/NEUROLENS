from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]
PERSIST_DIR = BASE_DIR / "data" / "chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def ingest_documents(pdf_paths: list):
    docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separators=["\n\n", "\n", ".", " "]
    )
    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True}
    )

    PERSIST_DIR.mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma(
        collection_name="neurolens_docs",
        embedding_function=embeddings,
        persist_directory=str(PERSIST_DIR)
    )

    vectorstore.add_documents(split_docs)
    print(f"✓ Ingested {len(split_docs)} chunks into ChromaDB")

if __name__ == "__main__":
    print("Ingesting......\n")
    ingest_documents([
        r"F:\Naman\NeuroLens\data\knowledge_base\brain_tumor_primer.pdf",
        r"F:\Naman\NeuroLens\data\knowledge_base\clinical_guidelines.pdf",
        r"F:\Naman\NeuroLens\data\knowledge_base\glioma.pdf",
        r"F:\Naman\NeuroLens\data\knowledge_base\meningioma.pdf",
        r"F:\Naman\NeuroLens\data\knowledge_base\newly_diagnosed.pdf"
    ])
    print("Ingestion Completed")