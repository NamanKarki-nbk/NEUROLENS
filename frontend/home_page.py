import streamlit as st

def show():
    st.title("NEURO LENS")
    st.image("assets/brains.png", width=200)
    st.subheader("Brain Tumor Decision Support System")
    st.markdown("---")

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown("VIT ANALYSIS + RAG")
        st.write(
            "Upload brain MRI images to detect tumors using Vision Transformer (ViT) "
            "and receive explanations via RAG."
        )
        
        if st.button("Open VIT Analysis"):
            st.session_state.page = "VIT ANALYSIS"
            st.rerun()

    with col2:
        st.markdown("EfficientNet + RAG + GradCAM")
        st.write(
            "Upload brain MRI images to detect tumors using EfficientNet, "
            "generate explanations via RAG, and visualize Grad-CAM heatmaps."
        )
        if st.button("Open EfficientNet Analysis"):
            st.session_state.page = "EFFECIENT NET ANALYSIS"
            st.rerun()

    with col3:
        st.markdown("Ensemble + RAG")
        st.write(
            "Combine ViT and EfficientNet predictions using ensemble voting "
            "and generate explanations via RAG."
        )
        if st.button("Open Ensemble Analysis"):
            st.session_state.page = "ENSEMBLE ANALYSIS"
            st.rerun()
            
    st.markdown("---")