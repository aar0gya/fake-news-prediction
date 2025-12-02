#############################
#  Fake News Detection App
#  HuggingFace API Version (Cloud Safe)
#############################

import os
import requests
import streamlit as st

API_URL = "https://api-inference.huggingface.co/models/jy46604790/Fake-News-Bert-Detect"
HF_TOKEN = os.environ.get("HF_TOKEN")

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# =========================
# Page config & UI styling
# =========================

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide",
)

st.markdown(
    """
    <style>
        .stApp {
            background: radial-gradient(circle at top, #0f172a 0, #020617 50%, #020617 100%);
            color: #e5e7eb;
        }
        .card {
            background: rgba(15,23,42,0.9);
            border-radius: 18px;
            padding: 1.5rem 1.75rem;
            border: 1px solid rgba(148,163,184,0.3);
            box-shadow: 0 18px 45px rgba(15,23,42,0.75);
        }
        .prediction-card {
            background: linear-gradient(135deg, #0f172a, #020617);
            border-radius: 18px;
            padding: 1.5rem 1.75rem;
            border: 1px solid rgba(148,163,184,0.4);
        }
        .headline {
            font-size: 2.4rem;
            font-weight: 700;
            background: linear-gradient(90deg, #f97316, #e5e7eb);
            -webkit-background-clip: text;
            color: transparent;
        }
        .fake-label { color: #fecaca; }
        .real-label { color: #bbf7d0; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="headline">Fake News Detection Dashboard</div>', unsafe_allow_html=True)
st.write("Enter a news headline or short article to analyze whether it appears real or fake.")

left, right = st.columns([1.3, 1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    input_text = st.text_area("Paste news content", height=180)

    if st.button("Analyze News"):
        with st.spinner("Analyzing using HuggingFace model..."):
            output = query({"inputs": input_text})

            try:
                label = output[0][0]["label"]
                score = output[0][0]["score"]
            except:
                st.error("Error: Model did not return a valid response.")
                st.stop()

            pred = "Fake" if label == "LABEL_0" else "Real"
            conf = round(score * 100, 1)

            css_class = "fake-label" if pred == "Fake" else "real-label"

            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.markdown(f"<h3 class='{css_class}'>{pred} News</h3>", unsafe_allow_html=True)
            st.write(f"**Confidence:** {conf}%")
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Model Info")
    st.write("""
    • Model: RoBERTa Fake News Classifier  
    • Hosted on HuggingFace  
    • Inference via API (Cloud-compatible)  
    • No PyTorch / TensorFlow required  
    """)
    st.markdown("</div>", unsafe_allow_html=True)
