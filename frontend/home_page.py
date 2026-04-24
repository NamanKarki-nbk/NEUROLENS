import streamlit as st


def show():
    st.title("NEURO LENS")
    st.image("assets/brains.png", width=200)
    st.subheader("Brain Tumor Decision Support System")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4, gap="large")

    with col1:
        st.markdown("### ViT + RAG")
        st.write(
            "Upload brain MRI images to detect tumors using Vision Transformer (ViT) "
            "and receive explanations via RAG."
        )
        if st.button("Open ViT Analysis", use_container_width=True):
            st.session_state.page = "VIT ANALYSIS"
            st.rerun()

    with col2:
        st.markdown("###  EfficientNet + RAG + GradCAM")
        st.write(
            "Upload brain MRI images to detect tumors using EfficientNet, "
            "generate explanations via RAG, and visualize Grad-CAM heatmaps."
        )
        if st.button("Open EfficientNet Analysis", use_container_width=True):
            st.session_state.page = "EFFICIENT NET ANALYSIS"
            st.rerun()

    with col3:
        st.markdown("### Ensemble + RAG")
        st.write(
            "Combine ViT and EfficientNet predictions using ensemble voting "
            "and generate explanations via RAG."
        )
        if st.button("Open Ensemble Analysis", use_container_width=True):
            st.session_state.page = "ENSEMBLE ANALYSIS"
            st.rerun()

    with col4:
        st.markdown("### AI Chat Assistant")
        st.write(
            "Ask follow-up questions about your diagnosis. "
            "The assistant uses your scan results and medical knowledge to answer."
        )
        if st.button("Open Chat", use_container_width=True):
            st.session_state.page = "CHAT"
            st.rerun()

    st.markdown("---")