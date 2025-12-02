import os

os.environ["TRANSFORMERS_NO_TF"] = "1"

import streamlit as st
from transformers import pipeline

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

        .tag-pill {
            display: inline-block;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            border: 1px solid rgba(148,163,184,0.5);
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #9ca3af;
        }

        .headline {
            font-size: 2.4rem;
            font-weight: 700;
            background: linear-gradient(90deg, #f97316, #e5e7eb);
            -webkit-background-clip: text;
            color: transparent;
            margin-bottom: 0.25rem;
        }

        .subhead {
            font-size: 0.95rem;
            color: #9ca3af;
            margin-bottom: 1rem;
        }

        .prob-label {
            font-size: 0.8rem;
            color: #9ca3af;
            margin-bottom: 0.25rem;
        }

        .confidence {
            font-size: 1.6rem;
            font-weight: 600;
        }

        .fake-label {
            color: #fecaca;
        }

        .real-label {
            color: #bbf7d0;
        }

        textarea {
            border-radius: 14px !important;
            border: 1px solid rgba(148,163,184,0.4) !important;
        }

        [data-testid="stMetric"] {
            background: rgba(15,23,42,0.95);
            padding: 0.9rem 1rem;
            border-radius: 14px;
            border: 1px solid rgba(148,163,184,0.35);
        }

    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Model loading (Transformers)
# =========================

# Caching decorator depending on Streamlit version
if hasattr(st, "cache_resource"):
    cache_model = st.cache_resource
else:
    cache_model = st.cache


@cache_model(show_spinner=False)
def load_model():
    """
    Load RoBERTa-based fake news detection model.
    """
    model_name = "jy46604790/Fake-News-Bert-Detect"
    clf = pipeline("text-classification", model=model_name, tokenizer=model_name)
    return clf


with st.spinner("Loading transformer model for fake news detection..."):
    clf = load_model()

# ======================
# Header Section
# ======================
st.markdown(
    '<div class="tag-pill">Machine Learning · Transformers · Streamlit</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="headline">Fake News Detection Dashboard</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subhead">Paste a news headline or short article and check whether the model considers it more likely to be fake or real.</div>',
    unsafe_allow_html=True,
)

st.write("")

# ======================
# Layout
# ======================
left_col, right_col = st.columns([1.3, 1])

# ======================
# LEFT: Prediction Area
# ======================
with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Try it out")

    input_text = st.text_area(
        "Paste the news content or headline here",
        height=180,
        placeholder="Example: Government announces new policy to reduce taxes on middle-income families...",
    )

    st.caption("Tip: Works best on English news headlines or articles.")

    analyze_button = st.button("Analyze News", type="primary")
    result_placeholder = st.empty()

    if analyze_button and input_text.strip():
        with st.spinner("Analyzing content with transformer model..."):
            result = clf(
                input_text,
                truncation=True,
                max_length=512,
            )[0]

            raw_label = result["label"]
            score = float(result["score"])

            if raw_label == "LABEL_0":
                label_text = "Fake"
                confidence = score
            else:
                label_text = "Real"
                confidence = score

            confidence_pct = confidence * 100.0

        with result_placeholder.container():
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            label_class = "fake-label" if label_text == "Fake" else "real-label"

            st.markdown(
                f"<p class='prob-label'>Prediction</p>"
                f"<p class='confidence {label_class}'>{label_text} news</p>",
                unsafe_allow_html=True,
            )

            st.markdown(
                "<p class='prob-label'>Model confidence in this prediction</p>",
                unsafe_allow_html=True,
            )
            st.progress(confidence)
            st.write(f"{confidence_pct:.1f}%")

            st.markdown("---")
            st.caption(
                "This tool is experimental and should not be treated as an official fact-checking service."
            )

            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ======================
# RIGHT: Model Info
# ======================
with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Model overview")

    col1, col2 = st.columns(2)
    col1.metric("Base model", "RoBERTa-base")
    col2.metric("Training size", "40k+ articles")

    st.markdown("---")

    st.markdown(
        """
        **Under the hood**

        - Transformer: RoBERTa-base  
        - Dataset: 40k+ fact-checked articles  
        - Task: Fake vs Real classification  
        - Framework: PyTorch-only inference  

        **Interpretation**

        - *Real*: Text matches patterns of real journalism  
        - *Fake*: Text resembles deceptive or fabricated news  

        Always verify information using trusted, independent sources.
        """
    )

    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Example prompts")

    st.markdown(
        """
        - *"Government secretly approves plan to ban cash next week."*  
        - *"Scientists discover pill that reverses aging in 24 hours."*  
        - *"Reuters reports peaceful transition of power after election."*  
        """
    )

    st.markdown("</div>", unsafe_allow_html=True)
