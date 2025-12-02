##############################
#   FAKE NEWS DETECTION APP
##############################

import os
import requests
import streamlit as st
from textblob import TextBlob  # For sentiment insights (Cloud safe)

# ======================================
# HuggingFace API Configuration
# ======================================
API_URL = "https://api-inference.huggingface.co/models/jy46604790/Fake-News-Bert-Detect"
HF_TOKEN = os.environ.get("HF_TOKEN")

headers = {"Authorization": f"Bearer {HF_TOKEN}"}


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload, timeout=40)
    return response.json()


# ======================================
# Page Config & Global Styles
# ======================================
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide",
)

st.markdown(
    """
<style>
/* Background */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    color: #e2e8f0;
    font-family: 'Inter', sans-serif;
}

/* Headline Gradient */
.app-title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #fb923c, #facc15, #fef9c3);
    -webkit-background-clip: text;
    color: transparent;
}

/* Subtitle */
.subtitle {
    font-size: 1.2rem;
    color: #cbd5e1;
}

/* Card Containers */
.card {
    background: rgba(15, 23, 42, 0.85);
    border-radius: 20px;
    padding: 1.6rem 1.8rem;
    border: 1px solid rgba(148,163,184,0.25);
    box-shadow: 0 12px 35px rgba(0,0,0,0.45);
}

/* Prediction card */
.result-card {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    border-radius: 18px;
    padding: 1.5rem;
    border: 1px solid rgba(148,163,184,0.35);
}

/* Fake / Real Labels */
.fake-label { color: #fca5a5; font-weight: 700; font-size: 1.7rem; }
.real-label { color: #86efac; font-weight: 700; font-size: 1.7rem; }

/* Example Buttons */
.example-btn {
    border-radius: 30px;
    padding: 0.4rem 1rem;
    margin-right: 0.4rem;
    background-color: #1e293b;
    color: #e2e8f0;
    border: 1px solid #334155;
    cursor: pointer;
    transition: 0.2s;
}
.example-btn:hover {
    background-color: #475569;
    border-color: #94a3b8;
}

/* Additional Insights */
.insight-label {
    color: #fbbf24;
    font-weight: 700;
    font-size: 1.15rem;
}

</style>
""",
    unsafe_allow_html=True,
)

# ======================================
# Header
# ======================================
st.markdown("<div class='app-title'>Fake News Detection Dashboard</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Analyze news and get AI-driven insights, sentiment breakdown, keywords, and risk analysis.</div>",
    unsafe_allow_html=True)
st.write("")

# ======================================
# Quick Example Headlines
# ======================================
st.subheader("Quick Examples")

examples = [
    "Government secretly plans nationwide curfew starting tomorrow.",
    "Scientists announce breakthrough treatment that reverses aging.",
    "Stock markets soar as central bank signals new interest rate cuts.",
    "International peace deal reached after overnight negotiations.",
    "Aliens have contacted world leaders demanding a global meeting."
]

cols = st.columns(len(examples))
example_selected = ""

for i, text in enumerate(examples):
    if cols[i].button(text[:28] + "..."):
        example_selected = text

# ======================================
# Main Layout
# ======================================
left, right = st.columns([1.5, 1])

# ======================================
# LEFT PANEL (Prediction + Interpretability)
# ======================================
with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Enter News Content")

    input_text = st.text_area(
        "",
        value=example_selected,
        placeholder="Paste or type a news headline/article...",
        height=180
    )

    analyze = st.button("🔍 Analyze News", type="primary")
    result_area = st.empty()

    if analyze and input_text.strip():

        with st.spinner("Analyzing with AI model..."):
            output = query({"inputs": input_text})

        try:
            label = output[0][0]["label"]
            score = float(output[0][0]["score"])
        except Exception:
            result_area.error("API Error — Unable to analyze this text.")
            st.stop()

        prediction = "Fake" if label == "LABEL_0" else "Real"
        confidence_pct = round(score * 100, 2)
        label_class = "fake-label" if prediction == "Fake" else "real-label"

        # ---------------- Sentiment Analysis ----------------
        blob = TextBlob(input_text)
        polarity = blob.sentiment.polarity
        sentiment = "Negative" if polarity < -0.1 else "Positive" if polarity > 0.1 else "Neutral"

        # ---------------- Keyword Extraction ----------------
        keywords = list({w.lower() for w in input_text.split() if len(w) > 6})[:6]

        # ---------------- Credibility Score ----------------
        risk_label = (
            "High Risk" if prediction == "Fake" and confidence_pct > 70 else
            "Questionable" if prediction == "Fake" else
            "Likely Credible"
        )

        with result_area.container():
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)

            st.markdown(f"<span class='{label_class}'>{prediction} News</span>", unsafe_allow_html=True)
            st.write(f"**Model Confidence:** {confidence_pct}%")
            st.progress(score)

            st.markdown("---")
            st.markdown("<span class='insight-label'>📝 Sentiment</span>", unsafe_allow_html=True)
            st.write(f"Sentiment detected: **{sentiment}**")

            st.markdown("<span class='insight-label'>🔑 Possible Keywords</span>", unsafe_allow_html=True)
            st.write(", ".join(keywords) if keywords else "No strong keywords detected")

            st.markdown("<span class='insight-label'>📊 Credibility Assessment</span>", unsafe_allow_html=True)
            st.write(f"**{risk_label}**")

            st.caption("Note: These insights assist understanding but do not replace professional fact-checking.")
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ======================================
# RIGHT PANEL (Educational + Visual)
# ======================================
with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🧠 How Fake News Spreads")
    st.write("""
    Fake news thrives due to psychological and digital vulnerabilities:
    - 🔁 **Rapid Amplification** through social platforms  
    - 😱 **Emotional Targeting** driving engagement  
    - 🤖 **Bots & Automation** boosting false claims  
    - 🎯 **Confirmation Bias** reinforcing beliefs  
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🛡️ How AI Detects Fake News")
    st.write("""
    AI models assess:
    - Writing structure  
    - Emotional tone  
    - Fact consistency  
    - Linguistic patterns  
    """)
    st.write("Why AI is helpful:")
    st.write("""
    - Detects subtle misinformation  
    - Scales across large text volumes  
    - Provides instant credibility scores  
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🎯 Why This Project Matters")
    st.write("""
    - Shows **full-stack ML deployment**  
    - Demonstrates **NLP, UI/UX, API integration**  
    - Works **fully on Streamlit Cloud**  
    - Recruiters see: design + engineering + ML knowledge  
    """)
    st.markdown("</div>", unsafe_allow_html=True)
