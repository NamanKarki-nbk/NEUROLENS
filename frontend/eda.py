import streamlit as st 

def show():
    st.title("Performance Metrices")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3,gap="large")
    
    with col1:
        st.header("Confusion matrics of Vision Transformer")
        st.image("assets/cm_vit.png")
        m1,m2,m3 = st.columns(3)
        m1.metric("Accuracy", "98.37%")
        m2.metric("F1 Score", "98.36%")
        m3.metric("Loss", "0.1297")
        
    with col2:
        st.header("Confusion matrics of Efficient Net")
        st.image("assets/cm_effnet.png")
        m1,m2,m3 = st.columns(3)
        m1.metric("Accuracy", "98.36%")
        m2.metric("F1 Score", "98.31%")
        m3.metric("Loss", "0.047")
    
    with col3:
        st.header("Model Comparision")
        st.image("assets/model_comparison.png")