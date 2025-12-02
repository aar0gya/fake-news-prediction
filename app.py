###########################################################
#                FAKE NEWS DETECTION DASHBOARD
#     Enhanced UI · Credibility Insights · Topic Analysis
#     Cloud-Safe (HuggingFace API · No Local ML Needed)
###########################################################

import os
import requests
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sentiment_analyzer = SentimentIntensityAnalyzer()

############################################################
#                     HUGGINGFACE API
############################################################

API_URL = "https://router.huggingface.co/jy46604790/Fake-News-Bert-Detect"
HF_TOKEN = os.environ.get("HF_TOKEN")
headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Accept": "application/json",
    "Content-Type": "application/json"
}


def query(payload):
    """Send request to HuggingFace Router API with safe error handling."""
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=40)
        data = response.json()

        # Handle model loading / async response
        if isinstance(data, dict) and "estimated_time" in data:
            return {"error": "Model is loading. Try again in a few seconds."}

        return data

    except Exception as e:
        return {"error": f"Connection failed: {str(e)}"}


############################################################
#                       PAGE SETTINGS
############################################################

st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="wide")

# Global CSS (premium UI)
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg,#0f172a,#1e293b,#0f172a); color:#e2e8f0; }
.app-title { font-size:3rem; font-weight:900; background:linear-gradient(90deg,#fb923c,#fde047,#fef9c3); -webkit-background-clip:text; color:transparent;}
.card { background:rgba(15,23,42,0.85); padding:1.4rem 1.8rem; border-radius:20px; border:1px solid #334155; }
.result-card { background:rgba(30,41,59,0.85); padding:1.2rem; border-radius:16px; }
.fake-label { font-size:1.8rem; font-weight:700; color:#fca5a5; }
.real-label { font-size:1.8rem; font-weight:700; color:#86efac; }
.example-btn { background:#1e293b; padding:0.4rem 1rem; border-radius:20px; border:1px solid #475569; color:#e2e8f0; }
.example-btn:hover { background:#334155; }
.insight-title { font-weight:700; color:#fbbf24; font-size:1.2rem; }
</style>
""", unsafe_allow_html=True)

############################################################
#                         HEADER
############################################################
st.markdown("<div class='app-title'>Fake News Detection Dashboard</div>", unsafe_allow_html=True)
st.write(
    "A next-gen tool that evaluates credibility, writing tone, sentiment, topic category, and risk assessment for news headlines or short articles.")

############################################################
#                     QUICK EXAMPLES
############################################################
st.subheader("Quick Examples")

examples = [
    "Government secretly plans nationwide curfew starting tomorrow.",
    "Scientists reveal new vaccine breakthrough for global virus.",
    "Stock markets surge after central bank signals interest cuts.",
    "Peace agreement reached after historic diplomatic summit.",
    "Aliens have contacted world leaders demanding a meeting."
]

example_cols = st.columns(len(examples))
chosen_example = ""

for i, ex in enumerate(examples):
    if example_cols[i].button(ex[:25] + "..."):
        chosen_example = ex

############################################################
#                         LAYOUT
############################################################
left, right = st.columns([1.6, 1])

############################################################
#                  LEFT PANEL — ANALYSIS
############################################################
with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Enter News Headline or Article")

    input_text = st.text_area(
        "",
        value=chosen_example,
        placeholder="Paste or type a headline/article...",
        height=180
    )

    analyze = st.button("🔍 Analyze News")
    result_area = st.empty()

    if analyze and input_text.strip():

        # ----------------- API CALL -----------------
        with st.spinner("Analyzing with AI model..."):
            output = query({"inputs": input_text})

        if "error" in output:
            result_area.error("API Error: " + output["error"])
            st.stop()

        # Validate structure
        if not isinstance(output, list) or not isinstance(output[0], list):
            result_area.error("Unexpected response format.")
            st.json(output)
            st.stop()

        try:
            block = output[0][0]
            label = block["label"]
            confidence = float(block["score"])
        except:
            result_area.error("Model returned unexpected format.")
            st.json(output)
            st.stop()

        # ----------------- Prediction -----------------
        prediction = "Fake" if label == "LABEL_0" else "Real"
        conf_pct = round(confidence * 100, 2)
        label_class = "fake-label" if prediction == "Fake" else "real-label"

        # ----------------- Sentiment -----------------
        sent = sentiment_analyzer.polarity_scores(input_text)
        sentiment = "Positive" if sent["compound"] >= 0.35 else "Negative" if sent["compound"] <= -0.35 else "Neutral"

        # ----------------- Topic Guessing -----------------
        topic_keywords = {
            "politics": ["election", "minister", "government", "policy", "senate", "president"],
            "science": ["research", "scientist", "study", "laboratory", "breakthrough", "experiment"],
            "finance": ["market", "stocks", "economy", "inflation", "bank", "investment"],
            "conflict": ["war", "attack", "military", "peace", "negotiation", "ceasefire"]
        }
        detected_topic = "General News"
        for topic, words in topic_keywords.items():
            if any(w.lower() in input_text.lower() for w in words):
                detected_topic = topic.capitalize()

        # ----------------- Suspicious Phrase Detection -----------------
        suspicious_phrases = ["shocking truth", "you won't believe", "secret plan", "hidden agenda", "miracle cure"]
        flagged = [p for p in suspicious_phrases if p in input_text.lower()]

        # ----------------- Writing Complexity -----------------
        word_count = len(input_text.split())
        complexity_label = (
            "Simple" if word_count < 10 else
            "Moderate" if word_count < 25 else
            "Complex"
        )

        # ----------------- Risk Assessment -----------------
        risk = (
            "⚠️ High Risk of Misinformation" if prediction == "Fake" and conf_pct > 70 else
            "❓ Questionable – Needs Verification" if prediction == "Fake" else
            "✔ Likely Credible"
        )

        ############################################################
        #                DISPLAY RESULTS
        ############################################################
        with result_area.container():
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)

            st.markdown(f"<span class='{label_class}'>{prediction} News</span>", unsafe_allow_html=True)
            st.write(f"**Confidence:** {conf_pct}%")
            st.progress(confidence)

            st.markdown("---")

            st.markdown("<p class='insight-title'>🧠 Sentiment</p>", unsafe_allow_html=True)
            st.write(f"Detected sentiment: **{sentiment}**")

            st.markdown("<p class='insight-title'>🧭 Topic Category</p>", unsafe_allow_html=True)
            st.write(f"Likely topic: **{detected_topic}**")

            st.markdown("<p class='insight-title'>🔍 Writing Complexity</p>", unsafe_allow_html=True)
            st.write(f"Style complexity: **{complexity_label}**")

            st.markdown("<p class='insight-title'>⚠️ Suspicious Flags</p>", unsafe_allow_html=True)
            st.write(", ".join(flagged) if flagged else "No obvious red flags detected.")

            st.markdown("<p class='insight-title'>📊 Credibility Assessment</p>", unsafe_allow_html=True)
            st.write(f"**{risk}**")

            st.caption("This tool provides AI-assisted analysis and should not replace professional fact-checking.")

            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

############################################################
#             RIGHT PANEL — EDUCATIONAL INSIGHTS
############################################################
with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🧠 How Fake News Spreads")
    st.write("""
Fake news thrives due to:
- 🔁 Viral amplification  
- 😱 Emotional manipulation  
- 🤖 Bot networks  
- 🎯 Confirmation bias  
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🛡️ How AI Detects Fake News")
    st.write("""
AI models examine:
- Linguistic inconsistencies  
- Emotional tone  
- Claim plausibility  
- Keyword anomalies  
""")
    st.write("AI helps by offering:")
    st.write("""
- Scalable analysis  
- Instant scoring  
- Pattern detection  
""")
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🚀 Why This Project Stands Out")
    st.write("""
This showcase demonstrates:
- Full-stack ML deployment  
- Practical use of HuggingFace APIs  
- UI/UX engineering in Streamlit  
- Explainable AI (XAI) techniques  
""")
    st.markdown("</div>", unsafe_allow_html=True)
