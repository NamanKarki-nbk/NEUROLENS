from dataclasses import dataclass, field
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.rag.retriever import HybridRetriever
from src.rag.reranker import rerank
from src.rag.query_rewriter import rewrite_query
from dotenv import load_dotenv
import os

load_dotenv()

# ── Label Mapping ────────────────────────────────────
CLASS_NAMES = {
    0: "Glioma",
    1: "Meningioma",
    2: "No Tumor",
    3: "Pituitary Tumor"
}

# ── LLM ─────────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.4
)

# ── Chat Prompt ──────────────────────────────────────
chat_prompt = PromptTemplate(
    input_variables=["diagnosis", "context", "history", "question"],
    template="""You are a compassionate medical assistant helping a patient understand their brain scan result.

Patient's diagnosis: {diagnosis}

Relevant Medical Information:
{context}

Conversation so far:
{history}

Patient's question: {question}

Instructions:
- Answer the patient's specific question using the medical information above
- Keep answers under 150 words and avoid medical jargon
- Be empathetic and reassuring
- If asked something unrelated to their diagnosis, gently redirect
- Do NOT say "based on the context" or reference any documents
- Do NOT repeat the full diagnosis explanation if it was already given"""
)

# ── Chain ────────────────────────────────────────────
chat_chain = chat_prompt | llm | StrOutputParser()

# ── Retriever (initialized once, reused) ─────────────
retriever = HybridRetriever()


# ── Data Classes ─────────────────────────────────────
@dataclass
class ChatMessage:
    role: str        # "user" or "assistant"
    content: str


@dataclass
class ChatSession:
    prediction_idx: int
    prediction_label: str
    history: list = field(default_factory=list)  # list of ChatMessage

    def add_message(self, role: str, content: str):
        self.history.append(ChatMessage(role=role, content=content))

    def format_history(self, max_turns: int = 6) -> str:
        """Return last N turns formatted as a readable string."""
        recent = self.history[-(max_turns * 2):]
        lines = []
        for msg in recent:
            prefix = "Patient" if msg.role == "user" else "Assistant"
            lines.append(f"{prefix}: {msg.content}")
        return "\n".join(lines) if lines else "No previous conversation."


# ── Main Chat Manager Class ───────────────────────────
class ChatManager:
    def __init__(self, prediction_idx: int, initial_explanation: str = None):
        """
        Args:
            prediction_idx: The class index from the model (0-3)
            initial_explanation: The first RAG explanation already shown to the patient.
                                 Pass this so the LLM knows what was already said.
        """
        label = CLASS_NAMES.get(prediction_idx, "Unknown")
        self.session = ChatSession(
            prediction_idx=prediction_idx,
            prediction_label=label
        )

        # Seed the conversation with the initial explanation if provided
        if initial_explanation:
            self.session.add_message("assistant", initial_explanation)

    def _retrieve_context(self, user_question: str) -> str:
        """Run hybrid RAG retrieval scoped to diagnosis + user question."""
        combined_query = f"{self.session.prediction_label}: {user_question}"
        queries = rewrite_query(combined_query)

        all_docs = []
        for query in queries:
            docs = retriever.retrieve(query, top_k=6)
            all_docs.extend(docs)

        # Deduplicate
        seen = set()
        unique_docs = []
        for doc in all_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_docs.append(doc)

        base_query = f"{self.session.prediction_label} {user_question}"
        final_docs = rerank(base_query, unique_docs, top_k=4)

        return "\n\n".join(doc.page_content for doc in final_docs)

    def chat(self, user_message: str) -> str:
        """
        Send a user message and get a RAG-grounded response.

        Args:
            user_message: The patient's question as a string

        Returns:
            The assistant's reply as a string
        """
        print(f"\n── Chat Turn ──────────────────────────────")
        print(f"  Diagnosis: {self.session.prediction_label}")
        print(f"  User: {user_message}")

        # No-tumor shortcut — no RAG needed
        if self.session.prediction_idx == 2:
            reply = (
                "Your scan showed no tumor, which is great news! "
                "Feel free to ask me anything about what to expect "
                "or what regular checkups involve."
            )
            self.session.add_message("user", user_message)
            self.session.add_message("assistant", reply)
            return reply

        # Step 1: Retrieve relevant context for this specific question
        context = self._retrieve_context(user_message)

        # Step 2: Format conversation history
        history_str = self.session.format_history()

        # Step 3: Generate response
        reply = chat_chain.invoke({
            "diagnosis": self.session.prediction_label,
            "context": context,
            "history": history_str,
            "question": user_message,
        })

        # Step 4: Save to history
        self.session.add_message("user", user_message)
        self.session.add_message("assistant", reply)

        print(f"  ✓ Reply generated")
        return reply

    def get_history(self) -> list:
        """Return the full conversation history as a list of ChatMessage objects."""
        return self.session.history

    def reset(self):
        """Clear conversation history (keeps the diagnosis)."""
        label = self.session.prediction_label
        idx = self.session.prediction_idx
        self.session = ChatSession(prediction_idx=idx, prediction_label=label)
        print(f"  ✓ Chat history cleared for {label}")