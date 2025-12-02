###########################################################
#          FAKE NEWS DETECTION DASHBOARD                  #
###########################################################

import os
import pandas as pd
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sentiment_analyzer = SentimentIntensityAnalyzer()

# =========================================================
# HuggingFace Router API configuration
# =========================================================

HF_TOKEN = os.environ.get("HF_TOKEN")  # must be set in Streamlit Secrets on cloud
API_URL = "https://router.huggingface.co/jy46604790/Fake-News-Bert-Detect"

if HF_TOKEN:
    HF_HEADERS = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
else:
    HF_HEADERS = None  # indicates no remote model


def hf_query(text: str):
    """Call HuggingFace router API. Returns dict with either
       {'ok': True, 'label': 'Fake'/'Real', 'score': float}
       or {'ok': False, 'error': '...'}.
    """
    if HF_HEADERS is None:
        return {"ok": False, "error": "HF_TOKEN not set; using local model."}

    try:
        resp = requests.post(API_URL, headers=HF_HEADERS, json={"inputs": text}, timeout=40)
        if not resp.text.strip():
            return {"ok": False, "error": "Empty response from HuggingFace."}

        try:
            data = resp.json()
        except ValueError as e:
            return {"ok": False, "error": f"Invalid JSON from HuggingFace: {e}"}

        # Model loading / router async message
        if isinstance(data, dict) and "estimated_time" in data:
            return {"ok": False, "error": "Model is loading on HuggingFace. Try again in a few seconds."}
        if isinstance(data, dict) and "error" in data:
            return {"ok": False, "error": f"HuggingFace error: {data['error']}"}

        # Accept both [ {label,score} ] and [ [ {label,score} ] ] formats
        block = None
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                block = data[0]
            elif len(data) > 0 and isinstance(data[0], list) and len(data[0]) > 0 and isinstance(data[0][0], dict):
                block = data[0][0]

        if block is None:
            return {"ok": False, "error": f"Unexpected response format: {data}"}

        raw_label = str(block.get("label", ""))
        score = float(block.get("score", 0.0))

        # Map labels (model uses LABEL_0 = fake, LABEL_1 = real)
        if raw_label in ["LABEL_0", "0", "FAKE", "fake"]:
            label = "Fake"
        else:
            label = "Real"

        return {"ok": True, "label": label, "score": score}

    except Exception as e:
        return {"ok": False, "error": f"Connection failed: {e}"}


# =========================================================
# Local fallback model (TF-IDF + Logistic Regression)
# =========================================================

@st.cache_resource(show_spinner=True)
def load_local_model():
    """Train a local classifier on fake.csv (title + text).
       Returns (model, vectorizer, test_accuracy).
    """
    df = pd.read_csv("fake.csv")
    df = df.fillna("")

    # Combine title + text (Option C)
    df["content"] = df["title"].astype(str) + " " + df["text"].astype(str)

    X = df["content"].values
    y = df["label"].astype(int).values  # 1 = real, 0 = fake

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english",
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_vec, y_train)

    acc = clf.score(X_test_vec, y_test)

    return clf, vectorizer, acc


def local_predict(text: str):
    """Use local TF-IDF + LogisticRegression model."""
    clf, vec, acc = load_local_model()
    X_vec = vec.transform([text])
    proba = clf.predict_proba(X_vec)[0]  # [p(fake=0), p(real=1)]
    prob_fake = float(proba[0])
    prob_real = float(proba[1])

    if prob_real >= prob_fake:
        label = "Real"
        score = prob_real
    else:
        label = "Fake"
        score = prob_fake

    return {
        "ok": True,
        "label": label,
        "score": score,
        "source": "Local ML model (TF-IDF + Logistic Regression)",
        "local_accuracy": acc,
    }


# =========================================================
# Unified analysis function (Hybrid routing)
# =========================================================

def analyze_text(text: str):
    """Try HuggingFace first; if that fails, fall back to local model."""
    # Attempt remote model if configured
    if HF_HEADERS is not None:
        remote = hf_query(text)
        if remote.get("ok"):
            return {
                "label": remote["label"],
                "score": remote["score"],
                "source": "HuggingFace RoBERTa (router API)",
                "error": None,
            }
        # If remote failed, continue to local fallback
        remote_error = remote.get("error", "Unknown remote error.")
    else:
        remote_error = "HF_TOKEN not configured; using local model."

    # Local fallback
    local = local_predict(text)
    return {
        "label": local["label"],
        "score": local["score"],
        "source": local["source"],
        "local_accuracy": local.get("local_accuracy"),
        "error": remote_error,
    }


# =========================================================
# Page config & CSS
# =========================================================

st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="wide")

st.markdown(
    """
<style>
.stApp { background: linear-gradient(135deg,#0f172a,#1e293b,#0f172a); color:#e2e8f0; }
.app-title { font-size:3rem; font-weight:900;
             background:linear-gradient(90deg,#fb923c,#fde047,#fef9c3);
             -webkit-background-clip:text; color:transparent; }
.card { background:rgba(15,23,42,0.85); padding:1.4rem 1.8rem;
        border-radius:20px; border:1px solid #334155; }
.result-card { background:rgba(30,41,59,0.9); padding:1.2rem;
               border-radius:16px; }
.fake-label { font-size:1.8rem; font-weight:700; color:#fca5a5; }
.real-label { font-size:1.8rem; font-weight:700; color:#86efac; }
.example-btn { background:#1e293b; padding:0.4rem 1rem; border-radius:20px;
               border:1px solid #475569; color:#e2e8f0; }
.example-btn:hover { background:#334155; }
.insight-title { font-weight:700; color:#fbbf24; font-size:1.15rem; }
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# Header
# =========================================================

st.markdown("<div class='app-title'>Fake News Detection Dashboard</div>", unsafe_allow_html=True)
st.write("Hybrid AI that evaluates credibility, sentiment, topic, style, and risk — with both a cloud model and a local backup model.")

# =========================================================
# Quick examples
# =========================================================

st.subheader("Quick Examples")

examples = [
    "Government secretly plans nationwide curfew starting tomorrow.",
    "Scientists reveal new vaccine breakthrough for global virus.",
    "Stock markets surge after central bank signals interest rate cuts.",
    "Peace agreement reached after historic diplomatic summit.",
    "Aliens have contacted world leaders demanding a global meeting.",
]

example_cols = st.columns(len(examples))
chosen_example = ""

for i, ex in enumerate(examples):
    if example_cols[i].button(ex[:25] + "..."):
        chosen_example = ex

# =========================================================
# Layout
# =========================================================

left, right = st.columns([1.6, 1])

# =========================================================
# Left panel: prediction & insights
# =========================================================

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Enter News Headline or Article")

    input_text = st.text_area(
        "",
        value=chosen_example,
        placeholder="Paste or type a news headline/article...",
        height=180,
    )

    analyze = st.button("🔍 Analyze News")
    result_area = st.empty()

    if analyze and input_text.strip():
        with st.spinner("Analyzing..."):
            res = analyze_text(input_text)

        label = res["label"]
        score = float(res["score"])
        source = res["source"]
        remote_error = res.get("error")
        confidence_pct = round(score * 100, 2)
        label_class = "fake-label" if label == "Fake" else "real-label"

        # Sentiment
        sent_scores = sentiment_analyzer.polarity_scores(input_text)
        compound = sent_scores["compound"]
        if compound >= 0.35:
            sentiment = "Positive"
        elif compound <= -0.35:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        # Topic guess
        topic_keywords = {
            "Politics": ["election", "minister", "government", "policy", "senate", "president"],
            "Science": ["research", "scientist", "study", "laboratory", "breakthrough", "experiment", "vaccine"],
            "Finance": ["market", "stocks", "economy", "inflation", "bank", "investment"],
            "Conflict": ["war", "attack", "military", "peace", "negotiation", "ceasefire"],
        }
        topic = "General News"
        lower_text = input_text.lower()
        for t, words in topic_keywords.items():
            if any(w in lower_text for w in words):
                topic = t
                break

        # Suspicious phrases
        suspicious_phrases = [
            "shocking truth",
            "you won't believe",
            "secret plan",
            "hidden agenda",
            "miracle cure",
            "what they don't want you to know",
        ]
        flags = [p for p in suspicious_phrases if p in lower_text]

        # Complexity
        word_count = len(input_text.split())
        complexity = "Simple" if word_count < 10 else "Moderate" if word_count < 25 else "Complex"

        # Risk
        risk = (
            "⚠️ High Risk of Misinformation" if label == "Fake" and confidence_pct > 70
            else "❓ Questionable – Needs Verification" if label == "Fake"
            else "✔ Likely Credible"
        )

        with result_area.container():
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)

            st.markdown(f"<span class='{label_class}'>{label} News</span>", unsafe_allow_html=True)
            st.write(f"**Confidence:** {confidence_pct:.2f}%")
            st.progress(score)

            st.write(f"**Model source:** {source}")
            if remote_error and "HF_TOKEN" not in remote_error:
                st.caption(f"Remote model note: {remote_error}")

            st.markdown("---")
            st.markdown("<p class='insight-title'>🧠 Sentiment</p>", unsafe_allow_html=True)
            st.write(f"Detected sentiment: **{sentiment}**")

            st.markdown("<p class='insight-title'>🧭 Topic Category</p>", unsafe_allow_html=True)
            st.write(f"Likely topic: **{topic}**")

            st.markdown("<p class='insight-title'>🔍 Writing Complexity</p>", unsafe_allow_html=True)
            st.write(f"Style complexity: **{complexity}** ({word_count} words)")

            st.markdown("<p class='insight-title'>⚠️ Suspicious Phrases</p>", unsafe_allow_html=True)
            st.write(", ".join(flags) if flags else "No obvious sensational phrases detected.")

            st.markdown("<p class='insight-title'>📊 Credibility Assessment</p>", unsafe_allow_html=True)
            st.write(f"**{risk}**")

            st.caption("This tool is for educational purposes and should not replace professional fact-checking.")
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# Right panel: educational content
# =========================================================

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🧠 How Fake News Spreads")
    st.write(
        "- 🔁 Viral sharing on social media\n"
        "- 😱 Emotionally charged headlines\n"
        "- 🤖 Bot networks amplifying content\n"
        "- 🎯 Confirmation bias in audiences"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🛡️ How This App Detects It")
    st.write(
        "- Primary: RoBERTa model via HuggingFace Router API\n"
        "- Backup: TF-IDF + Logistic Regression\n"
        "- Extra: Sentiment, topic, and style analysis\n"
        "- Output: Fake/Real prediction with confidence and risk label"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🎯 Why This Project Is Valuable")
    st.write(
        "- Demonstrates hybrid **cloud + local ML** design\n"
        "- Uses **NLP, feature engineering, and model deployment**\n"
        "- Designed with **recruiter-friendly UI/UX**\n"
        "- Resilient: still works when external APIs fail"
    )
    st.markdown("</div>", unsafe_allow_html=True)
