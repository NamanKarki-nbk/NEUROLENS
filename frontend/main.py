import streamlit as st

st.set_page_config(
    page_title="NEURO LENS",
    layout="wide",
    
)

st.sidebar.image("assets/brain.png", width=100)
st.sidebar.title("NEURO LENS")
st.sidebar.caption("Brain Tumor Decision Support System")
st.sidebar.markdown("---")


st.sidebar.markdown(
    "<h3 style='text-align:center; font-weight:bold;'>Navigation</h3>",
    unsafe_allow_html=True
)

page = st.sidebar.selectbox(
    "",
    [
        "Home",
        "VIT ANALYSIS",
        "EFFECIENT NET ANALYSIS",
        "ENSEMBLE ANALYSIS",
        "CONFUSION MATRIX AND MODEL COMPARISION"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Built with FastAPI, Streamlit, Efficient-NET, VIT, RAG")
st.sidebar.image("assets/deep.png", width=300)

if page == "Home":
    from home_page import show
    show()
    
elif page == "VIT ANALYSIS":
    from vit_page import show
    show()

elif page == "EFFECIENT NET ANALYSIS":
    from effiecient_net import show
    show()
    
elif page == "ENSEMBLE ANALYSIS":
    from ensemble import show
    show()

elif page == "CONFUSION MATRIX AND MODEL COMPARISION":
    from eda import show
    show()

