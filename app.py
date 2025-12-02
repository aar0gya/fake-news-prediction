import os
import requests
import streamlit as st
import pandas as pd
import numpy as np

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

sentiment_analyzer = SentimentIntensityAnalyzer()

# =========================================================
# HuggingFace Router API configuration
# =========================================================

HF_TOKEN = os.environ.get("HF_TOKEN")
API_URL = "https://router.huggingface.co/jy46604790/Fake-News-Bert-Detect"

HF_HEADERS = (
    {"Authorization": f"Bearer {HF_TOKEN}", "Accept": "application/json", "Content-Type": "application/json"}
    if HF_TOKEN else None
)

def hf_query(text):
    if HF_HEADERS is None:
        return {"ok": False, "error": "HF_TOKEN not set; using local model."}
    try:
        resp = requests.post(API_URL, headers=HF_HEADERS, json={"inputs": text}, timeout=40)
        if not resp.text.strip():
            return {"ok": False, "error": "Empty response from HuggingFace."}

        try:
            data = resp.json()
        except Exception as e:
            return {"ok": False, "error": f"Invalid JSON: {e}"}

        if isinstance(data, dict) and "estimated_time" in data:
            return {"ok": False, "error": "Model loading on HuggingFace—try again."}
        if isinstance(data, dict) and "error" in data:
            return {"ok": False, "error": data["error"]}

        block = None
        if isinstance(data, list):
            if len(data) and isinstance(data[0], dict):
                block = data[0]
            elif len(data) and isinstance(data[0], list) and len(data[0]) and isinstance(data[0][0], dict):
                block = data[0][0]

        if block is None:
            return {"ok": False, "error": f"Unexpected format: {data}"}

        raw = block.get("label", "LABEL_0")
        score = float(block.get("score", 0.0))

        label = "Fake" if raw in ["LABEL_0", "0"] else "Real"

        return {"ok": True, "label": label, "score": score}

    except Exception as e:
        return {"ok": False, "error": f"Connection failed: {e}"}

# =========================================================
# Local TFIDF model fallback
# =========================================================

@st.cache_resource(show_spinner=True)
def load_local_model():
    df = pd.read_csv("fake.csv").fillna("")
    df["content"] = df["title"].astype(str) + " " + df["text"].astype(str)

    X = df["content"].values
    y = df["label"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    vec = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_vec, y_train)

    acc = clf.score(X_test_vec, y_test)
    return clf, vec, acc

def local_predict(text):
    clf, vec, acc = load_local_model()
    Xv = vec.transform([text])
    proba = clf.predict_proba(Xv)[0]

    if proba[1] >= proba[0]:
        return {"ok": True, "label": "Real", "score": float(proba[1]), "source": "Local ML", "local_accuracy": acc}
    return {"ok": True, "label": "Fake", "score": float(proba[0]), "source": "Local ML", "local_accuracy": acc}

# =========================================================
# Unified analysis function
# =========================================================
def analyze_text(text):
    if HF_HEADERS:
        remote = hf_query(text)
        if remote["ok"]:
            return {"label": remote["label"], "score": remote["score"], "source": "HuggingFace", "error": None}
        remote_error = remote["error"]
    else:
        remote_error = "HF token missing"

    local = local_predict(text)
    return {
        "label": local["label"],
        "score": local["score"],
        "source": local["source"],
        "local_accuracy": local.get("local_accuracy"),
        "error": remote_error,
    }

# =========================================================
# UI Setup
# =========================================================

st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="wide")

st.markdown(
    """
<style>
.stApp { background: linear-gradient(135deg,#0f172a,#1e293b,#0f172a); color:#e2e8f0; }
.app-title { font-size:3rem; font-weight:900; 
             background:linear-gradient(90deg,#fb923c,#fde047,#fef9c3);
             -webkit-background-clip:text; color:transparent; }
.card { background:rgba(15,23,42,.85); padding:1.4rem 1.8rem; border-radius:20px; border:1px solid #334155; }
.result-card { background:rgba(30,41,59,.9); padding:1.2rem; border-radius:16px; }
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
st.write("Analyze credibility, sentiment, topic, writing style, and risk using hybrid AI.")

# =========================================================
# QUICK EXAMPLES — now auto-analyze when clicked
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

# Memory variable to auto-trigger analysis:
if "auto_text" not in st.session_state:
    st.session_state.auto_text = ""
if "auto_run" not in st.session_state:
    st.session_state.auto_run = False

for i, ex in enumerate(examples):
    if example_cols[i].button(ex[:28] + "…"):
        st.session_state.auto_text = ex
        st.session_state.auto_run = True
        st.rerun()   # instantly reload and auto-run analysis

# =========================================================
# MAIN LAYOUT
# =========================================================

left, right = st.columns([1.6, 1])

# LEFT PANEL
with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Enter News Headline or Article")

    input_text = st.text_area(
        "",
        value=st.session_state.auto_text,
        placeholder="Paste or type a news headline/article...",
        height=180,
    )

    analyze_btn = st.button("🔍 Analyze News")

    # Auto-run analysis if example clicked
    trigger = analyze_btn or st.session_state.auto_run

    result_box = st.empty()

    if trigger and input_text.strip():
        st.session_state.auto_run = False  # reset

        with st.spinner("Analyzing..."):
            res = analyze_text(input_text)

        label = res["label"]
        score = res["score"]
        confidence = round(score * 100, 2)
        source = res["source"]
        error_msg = res.get("error")

        label_style = "fake-label" if label == "Fake" else "real-label"

        # Sentiment
        sent = sentiment_analyzer.polarity_scores(input_text)
        compound = sent["compound"]
        sentiment = "Positive" if compound >= 0.35 else "Negative" if compound <= -0.35 else "Neutral"

        # topic guess
        topic_map = {
            "Politics": ["election","government","senate","minister","policy"],
            "Science": ["research","study","scientist","vaccine","laboratory"],
            "Finance": ["market","stocks","bank","inflation","economy"],
            "Conflict": ["war","military","attack","ceasefire","peace"],
        }
        topic = "General News"
        lt = input_text.lower()
        for t, words in topic_map.items():
            if any(w in lt for w in words):
                topic = t
                break

        st.markdown("<div class='result-card'>", unsafe_allow_html=True)

        st.markdown(f"<span class='{label_style}'>{label} News</span>", unsafe_allow_html=True)
        st.write(f"**Confidence:** {confidence}%")
        st.progress(score)

        st.write(f"**Model used:** {source}")
        if error_msg and "HF_TOKEN" not in error_msg:
            st.caption(f"(Remote model note: {error_msg})")

        st.markdown("---")
        st.markdown("<p class='insight-title'>🧠 Sentiment</p>", unsafe_allow_html=True)
        st.write(f"Emotion detected: **{sentiment}**")

        st.markdown("<p class='insight-title'>🧭 Topic Category</p>", unsafe_allow_html=True)
        st.write(f"Likely topic: **{topic}**")

        st.markdown("<p class='insight-title'>📊 Writing Complexity</p>", unsafe_allow_html=True)
        wc = len(input_text.split())
        comp = "Simple" if wc < 10 else "Moderate" if wc < 25 else "Complex"
        st.write(f"Text complexity: **{comp}** ({wc} words)")

        st.markdown("<p class='insight-title'>⚠️ Credibility Risk</p>", unsafe_allow_html=True)
        risk = (
            "⚠️ High Risk of Misinformation" if label == "Fake" and confidence > 70 else
            "❓ Needs Verification" if label == "Fake" else
            "✔ Likely Credible"
        )
        st.write(risk)

        st.caption("For educational use. Not a substitute for real fact-checking.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# RIGHT PANEL (unchanged)
with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🧠 How Fake News Spreads")
    st.write("- Viral emotional headlines\n- Bots promoting misinformation\n- Confirmation bias\n- Fast social sharing")
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🛡️ How This App Detects It")
    st.write("- Hybrid AI (Cloud + Local Model)\n- NLP sentiment\n- Topic detection\n- Writing pattern analysis\n- Risk scoring")
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🎯 Why This Project Stands Out")
    st.write("- End-to-end NLP pipeline\n- Cloud resiliency with fallback\n- Strong UI/UX\n- Demonstrates ML engineering + deployment skills")
    st.markdown("</div>", unsafe_allow_html=True)
