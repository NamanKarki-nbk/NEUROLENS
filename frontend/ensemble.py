import streamlit as st
from PIL import Image
import requests

API_URL = "http://127.0.0.1:8000//ensemble-and-explain"


def show():
    st.title("ENSEMBLE VOTING ANALYSIS")
    st.markdown("---")

    st.subheader(
        "Analyze the image using ENSEMBLE voting and get explanation through RAG system"
    )
    st.markdown("---")

    col1, col2 = st.columns([1, 1], gap="large")

    
    with col1:
        st.subheader("Upload X-Ray")

        uploaded = st.file_uploader(
            "Select a brain X-ray image",
            type=["jpg", "jpeg", "png"],
            help="Upload image in JPG or PNG format"
        )

        if uploaded:
            image = Image.open(uploaded)
            st.image(image, caption="Uploaded Image", use_container_width=True)

  
    with col2:
        st.subheader("Analysis Result")

        if uploaded:
            run = st.button("Analyze X-Ray", type="primary")

            if run:
                with st.spinner("Analyzing Image..."):

                    try:
                        files = {
                            "file": (
                                uploaded.name,
                                uploaded.getvalue(),
                                uploaded.type
                            )
                        }

                        response = requests.post(
                            API_URL,
                            files=files,
                            timeout=60
                        )

                        if response.status_code == 200:
                            result = response.json()

                            label = result.get("prediction")
                            conf = result.get("confidence")
                            prob = result.get("probabilities")
                            exp = result.get("explanation")

                            st.success("Analysis Complete")

                            st.markdown("### Prediction")
                            st.write(label)

                            st.markdown("### Confidence")
                            st.write(f"{conf}%")

                            st.markdown("### Explanation")
                            st.write(exp)

                            with st.expander("Probabilities"):
                                st.json(prob)

                        else:
                            st.error(f"API Error: {response.status_code}")

                    except requests.exceptions.ConnectionError:
                        st.error(
                            "Cannot connect to backend.\n\n"
                            "Run FastAPI with:\n"
                            "`uvicorn api.main:app --reload`"
                        )

                    except Exception as e:
                        st.error(f"Unexpected error: {str(e)}")