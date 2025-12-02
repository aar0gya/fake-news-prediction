import os

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["KERAS_BACKEND"] = "torch"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

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
# Load ONNX Transformer Model
# =========================

# Cache based on Streamlit version
if hasattr(st, "cache_resource"):
    cache_model = st.cache_resource
else:
    cache_model = st.cache


@cache_model(show_spinner=False)
def load_model():
    """
    Loads a fully ONNX-compatible fake news detection model.
    """
    model_name = "microsoft/xtremedistil-l6-h256-uncased-fakenews"

    # Force ONNX Runtime usage
    clf = pipeline(
        task="text-classification",
        model=model_name,
        tokenizer=model_name,
        revision="onnx",  # ensures ONNX export is used
    )

    return clf


with st.spinner("Loading fake news detection model..."):
    clf = load_model()


# =========================
# Header Section
# =========================
st.markdown(
    '<div class="tag-pill">Machine Learning · Transformers · ONNX · Streamlit</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="headline">Fake News Detection Dashboard</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subhead">Analyze a headline or short article using a lightweight ONNX-based transformer model optimized for Streamlit Cloud.</div>',
    unsafe_allow_html=True,
)

st.write("")


# =========================
# Layout
# =========================
left_col, right_col = st.columns([1.3, 1])


# =========================
# LEFT: Prediction Area
# =========================
with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Try it out")

    input_text = st.text_area(
        "Paste a news headline or article",
        height=180,
        placeholder="Example: Government announces new policy to reduce taxes...",
    )

    st.caption("Tip: Short news text (< 500 words) gives the best results.")

    analyze_button = st.button("Analyze News", type="primary")
    result_area = st.empty()

    if analyze_button and input_text.strip():
        with st.spinner("Analyzing content..."):
            output = clf(
                input_text,
                truncation=True,
                max_length=512,
            )[0]

            label_raw = output["label"]
            confidence = float(output["score"])

            label = "Real" if label_raw == "LABEL_1" else "Fake"
            label_class = "real-label" if label == "Real" else "fake-label"
            conf_pct = confidence * 100

        with result_area.container():
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)

            st.markdown(
                f"<p class='prob-label'>Prediction</p>"
                f"<p class='confidence {label_class}'>{label} news</p>",
                unsafe_allow_html=True,
            )

            st.markdown(
                "<p class='prob-label'>Model confidence</p>",
                unsafe_allow_html=True,
            )
            st.progress(confidence)
            st.write(f"{conf_pct:.1f}%")

            st.markdown("---")
            st.caption("This is an educational tool. Always verify with trusted sources.")

            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# RIGHT: Model Info
# =========================
with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Model Overview")

    col1, col2 = st.columns(2)
    col1.metric("Model Type", "Distilled BERT")
    col2.metric("Runtime", "ONNX Runtime")

    st.markdown("---")

    st.markdown(
        """
        **Under the hood**
        - Lightweight DistilBERT designed for classification
        - Exported to **ONNX for high-speed inference**
        - 6-layer Transformer → optimized for cloud usage
        - No PyTorch / TensorFlow dependencies

        **What the model does**
        - Evaluates linguistic patterns
        - Compares your text against news credibility datasets
        - Predicts: **Real** vs **Fake**

        The model does *not* check external facts or verify claims.
        """
    )

    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Example Prompts")

    st.markdown(
        """
        - *"Government secretly passes new ban on cash transactions."*  
        - *"Study shows breakthrough cure for Alzheimer's disease."*  
        - *"BBC confirms new diplomatic agreement signed today."*  
        """
    )

    st.markdown("</div>", unsafe_allow_html=True)
