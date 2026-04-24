import streamlit as st

st.set_page_config(
    page_title="NEURO LENS",
    layout="wide",
)

# ---------------- SIDEBAR ----------------
st.sidebar.image("assets/brain.png", width=100)
st.sidebar.title("NEURO LENS")
st.sidebar.caption("Brain Tumor Decision Support System")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<h3 style='text-align:center; font-weight:bold;'>Navigation</h3>",
    unsafe_allow_html=True
)

page = st.sidebar.selectbox(
    "Navigation",
    [
        "Home",
        "VIT ANALYSIS",
        "EFFICIENT NET ANALYSIS",
        "ENSEMBLE ANALYSIS",
        "CONFUSION MATRIX AND MODEL COMPARISON",
        "CHAT"                          # ← NEW
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

# Show active session info in sidebar if a chat session exists
if "session_id" in st.session_state and st.session_state.session_id:
    st.sidebar.success("💬 Chat session active")
    st.sidebar.caption(f"Diagnosis: **{st.session_state.get('diagnosis', 'Unknown')}**")
    if st.sidebar.button("Go to Chat"):
        page = "CHAT"
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Built with FastAPI, Streamlit, EfficientNet, ViT, RAG")
st.sidebar.image("assets/deep.png", width=300)

# ---------------- ROUTING ----------------
if page == "Home":
    from home_page import show
    show()

elif page == "VIT ANALYSIS":
    from vit_page import show
    show()

elif page == "EFFICIENT NET ANALYSIS":
    from effiecient_net import show
    show()

elif page == "ENSEMBLE ANALYSIS":
    from ensemble import show
    show()

elif page == "CONFUSION MATRIX AND MODEL COMPARISON":
    from eda import show
    show()

elif page == "CHAT":                    # ← NEW
    from chat_page import show
    show()