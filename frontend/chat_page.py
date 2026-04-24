import streamlit as st
import requests

BASE_URL = "http://127.0.0.1:8000"


def send_question(message: str) -> str:
    """Call POST /qna and return the assistant reply."""
    try:
        res = requests.post(
            f"{BASE_URL}/qna",
            json={"message": message},
            timeout=60
        )
        if res.status_code == 200:
            return res.json()["reply"]
        else:
            return f"❌ Error {res.status_code}: {res.json().get('detail', 'Unknown error')}"
    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to the backend. Make sure FastAPI is running."
    except Exception as e:
        return f"❌ Unexpected error: {str(e)}"


def show():
    st.title("Brain Tumor Q&A Assistant")
    st.caption("Ask anything about brain tumors — types, symptoms, treatments, and more.")
    st.markdown("---")

    # Initialise chat history
    if "qna_messages" not in st.session_state:
        st.session_state.qna_messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! I'm your brain tumor information assistant. "
                    "You can ask me anything about different types of brain tumors — "
                    "what they are, their symptoms, treatments, or what to expect. "
                    "How can I help you today?"
                )
            }
        ]

    # ── Layout ───────────────────────────────────────
    left, right = st.columns([1, 2], gap="large")

    with left:
        st.subheader("Suggested Questions")
        suggestions = [
            "What is a Glioma?",
            "What are the symptoms of Meningioma?",
            "How is a Pituitary Tumor treated?",
            "What is the difference between benign and malignant tumors?",
            "What causes brain tumors?",
            "Are brain tumors hereditary?",
            "What is the survival rate for Glioblastoma?",
            "What tests are used to diagnose brain tumors?",
        ]
        for q in suggestions:
            if st.button(q, use_container_width=True, key=f"suggest_{q}"):
                st.session_state.qna_messages.append({"role": "user", "content": q})
                with st.spinner("Thinking..."):
                    reply = send_question(q)
                st.session_state.qna_messages.append({"role": "assistant", "content": reply})
                st.rerun()

        st.markdown("---")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.qna_messages = [
                {
                    "role": "assistant",
                    "content": (
                        "Hello! I'm your brain tumor information assistant. "
                        "You can ask me anything about different types of brain tumors — "
                        "what they are, their symptoms, treatments, or what to expect. "
                        "How can I help you today?"
                    )
                }
            ]
            st.rerun()

    with right:
        st.subheader("Conversation")

        # Chat display
        chat_container = st.container(height=520)
        with chat_container:
            for msg in st.session_state.qna_messages:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

        # Input
        user_input = st.chat_input("Ask a question about brain tumors...")
        if user_input:
            st.session_state.qna_messages.append({"role": "user", "content": user_input})
            with st.spinner("Thinking..."):
                reply = send_question(user_input)
            st.session_state.qna_messages.append({"role": "assistant", "content": reply})
            st.rerun()