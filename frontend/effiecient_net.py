import streamlit as st
from PIL import Image
import requests

API_URL = "http://127.0.0.1:8000/analyze-and-explain/Efficient_net"


def clean_explanation(exp):
    if isinstance(exp, str):
        return exp.strip()
    if isinstance(exp, dict):
        return str(exp.get("answer") or exp.get("result") or exp)
    if isinstance(exp, list):
        return str(exp[0]) if exp else "No explanation available"
    return str(exp)


def show():
    st.title("EfficientNet Analysis")
    st.markdown("---")

    col1, col2 = st.columns([1, 1], gap="large")

    # LEFT
    with col1:
        st.subheader("Upload Image")

        uploaded = st.file_uploader(
            "Select a brain scan",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded:
            image = Image.open(uploaded)
            st.image(image, caption="Uploaded Image", use_container_width=True)

    # RIGHT
    with col2:
        st.subheader("Result")

        if uploaded and st.button("Analyze", type="primary"):

            with st.spinner("Analyzing..."):

                try:
                    files = {
                        "file": (
                            uploaded.name,
                            uploaded.getvalue(),
                            uploaded.type
                        )
                    }

                    res = requests.post(API_URL, files=files, timeout=60)

                    if res.status_code != 200:
                        st.error(f"API Error: {res.status_code}")
                        return

                    data = res.json()

                    label = data.get("prediction", "N/A")
                    conf = data.get("confidence", 0)
                    prob = data.get("probabilities", {})
                    exp = clean_explanation(data.get("explanation"))

                    st.success("Analysis Complete")

                    c1, c2 = st.columns(2)
                    c1.metric("Prediction", label)
                    c2.metric("Confidence", f"{conf}%")

                    st.markdown("### Explanation")
                    st.write(exp)

                    with st.expander("Probabilities"):
                        st.json(prob)

                except Exception as e:
                    st.error(str(e))