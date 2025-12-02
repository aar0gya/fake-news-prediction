import os
import requests
import streamlit as st

# ======================================
# HuggingFace API Configuration
# ======================================
API_URL = "https://api-inference.huggingface.co/models/jy46604790/Fake-News-Bert-Detect"
HF_TOKEN = os.environ.get("HF_TOKEN")

headers = {"Authorization": f"Bearer {HF_TOKEN}"}


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
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
    text-align: left;
}

/* Subheading */
.subtitle {
    font-size: 1.2rem;
    color: #cbd5e1;
    margin-bottom: 1rem;
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

/* Example buttons */
.example-btn {
    border-radius: 30px;
    padding: 0.4rem 1rem;
    margin-right: 0.5rem;
    background-color: #1e293b;
    color: #e2e8f0;
    border: 1px solid #334155;
    cursor: pointer;
    transition: 0.2s ease-in-out;
}
.example-btn:hover {
    background-color: #475569;
    border-color: #94a3b8;
}

/* Labels */
.fake-label { color: #fca5a5; font-weight: 700; font-size: 1.8rem; }
.real-label { color: #86efac; font-weight: 700; font-size: 1.8rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ======================================
# Page Header
# ======================================
st.markdown("<div class='app-title'>Fake News Detection Dashboard</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Analyze headlines or news snippets using an AI-powered model trained on real and fake news datasets.</div>",
    unsafe_allow_html=True,
)

# ======================================
# Example Headlines (Button Style)
# ======================================
st.subheader("Quick Examples")
examples = [
    "Government secretly plans nationwide curfew starting tomorrow.",
    "Scientists announce breakthrough treatment that reverses aging.",
    "Stock markets soar as central bank signals new interest rate cuts.",
    "International peace deal reached after overnight negotiations.",
    "Aliens have contacted world leaders demanding a global meeting."
]

example_cols = st.columns(len(examples))

example_selected = ""
for idx, text in enumerate(examples):
    if example_cols[idx].button(text[:35] + "...", key=f"ex_{idx}"):
        example_selected = text

# ======================================
# Main Input Area
# ======================================
left, right = st.columns([1.4, 1])

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("Enter News Content")
    input_text = st.text_area(
        "",
        placeholder="Type or paste a news headline/article...",
        height=180,
        value=example_selected
    )

    analyze = st.button("🔍 Analyze News", type="primary")

    result_box = st.empty()

    if analyze and input_text.strip():
        with st.spinner("Analyzing text using HuggingFace model..."):
            output = query({"inputs": input_text})

        try:
            label = output[0][0]["label"]
            score = float(output[0][0]["score"])
        except:
            result_box.error("Model error: Unable to interpret the API response.")
            st.stop()

        prediction = "Fake" if label == "LABEL_0" else "Real"
        conf_pct = round(score * 100, 1)
        label_class = "fake-label" if prediction == "Fake" else "real-label"

        with result_box.container():
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='{label_class}'>{prediction} News</div>", unsafe_allow_html=True)
            st.write(f"**Confidence:** {conf_pct}%")
            st.progress(score)
            st.caption("This is an AI model prediction and should not replace fact-checking from reliable sources.")
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ======================================
#  Right Panel — Information & Visuals
# ======================================

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🧠 How Fake News Spreads")

    st.write("""
    Fake news leverages human psychology and digital platforms.  
    Here are the **most common drivers** behind misinformation:
    """)

    st.markdown("""
    - 🔁 **Rapid Sharing:** Social media boosts sensational or emotional content.  
    - 😱 **Emotional Targeting:** Fear, anger, and surprise increase clicks.  
    - 🤖 **Bots & Troll Farms:** Automated networks amplify false narratives.  
    - 🎯 **Confirmation Bias:** People believe content aligning with their views.  
    """)

    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🛡️ How AI Detects Fake News")

    st.write("This app uses a machine learning model to identify patterns linked to misinformation:")

    st.markdown("""
    #### 🔍 What the AI looks at:
    - Unusual writing patterns  
    - Sensational or exaggerated claims  
    - Inconsistent facts  
    - Biased emotional tone  

    #### 🧪 Why AI is useful:
    - Processes large amounts of information  
    - Detects patterns humans might miss  
    - Reduces misinformation spread  
    - Gives a quick, data-driven credibility check  
    """)

    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🌐 Trusted News Sources")

    st.write("Always compare breaking news with reliable, evidence-based outlets:")

    colA, colB = st.columns(2)

    with colA:
        st.image("https://upload.wikimedia.org/wikipedia/commons/4/40/BBC_News_2022.svg", width=90)
        st.image("https://upload.wikimedia.org/wikipedia/commons/4/4e/Reuters_logo.svg", width=90)
        st.image("https://upload.wikimedia.org/wikipedia/commons/0/09/Aljazeera_eng.svg", width=90)

    with colB:
        st.image("https://upload.wikimedia.org/wikipedia/commons/a/a7/CNN_International_logo.svg", width=90)
        st.image("https://upload.wikimedia.org/wikipedia/commons/6/67/The_New_York_Times_logo.png", width=110)
        st.image("https://upload.wikimedia.org/wikipedia/commons/2/25/The_Guardian_2018.svg", width=110)

    st.caption("Tip: Cross-check at least **two independent** sources before trusting a headline.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🎯 Why This Project Matters")

    st.write("""
    Misinformation can influence elections, cause panic, and harm communities.  
    This project shows skills in:

    - 🧩 **Machine Learning Deployment**  
    - 🎨 **UI/UX Design**  
    - 🧪 **NLP & Text Classification**  
    - ☁️ **Cloud Deployment (Streamlit Cloud)**  
    - 🔗 **API Integration (HuggingFace)**  

    Recruiters can view this as a demonstration of **full-stack ML capability**,  
    combining technical depth with a polished, user-friendly interface.
    """)

    st.markdown("</div>", unsafe_allow_html=True)
