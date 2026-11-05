from pathlib import Path
import streamlit as st
import pandas as pd
from PIL import Image

from src.predict import predict_image
from src.utils import format_percent, display_class_name

st.set_page_config(
    page_title="FruitScan AI",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background: #0b1020; }
    .block-container { padding-top: 2rem; max-width: 1150px; }
    .hero {
        padding: 1.5rem 1.7rem; border-radius: 18px;
        background: linear-gradient(120deg, #172554, #164e63);
        border: 1px solid rgba(148,163,184,.25);
        margin-bottom: 1.2rem;
    }
    .hero h1 { color: #f8fafc; margin: 0; }
    .hero p { color: #cbd5e1; margin-bottom: 0; }
    .result {
        padding: 1rem 1.2rem; border-radius: 14px;
        background: rgba(16,185,129,.10);
        border: 1px solid rgba(16,185,129,.35);
    }
    [data-testid="stMetric"] {
        background: rgba(30,41,59,.65);
        padding: 12px; border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.title("🍎 FruitScan AI")
    st.caption("AI-powered fruit & vegetable recognition")
    st.markdown("---")
    st.markdown("**How it works**")
    st.write("1. Upload a fruit or vegetable image.")
    st.write("2. Run the CNN classifier.")
    st.write("3. Review the predicted class and top-3 scores.")
    st.markdown("---")
    st.caption("Educational prototype powered by a CNN trained on Fruits-360.")

st.markdown("""
<div class="hero">
  <h1>FruitScan AI 🍊</h1>
  <p>Upload an image and let a convolutional neural network identify the fruit.</p>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1, 1], gap="large")
with left:
    st.subheader("Upload an image")
    uploaded = st.file_uploader(
        "Choose a JPG, JPEG or PNG image",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded:
        try:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Image ready for analysis", use_container_width=True)
        except Exception:
            st.error("The uploaded file could not be opened as an image.")
            image = None
    else:
        image = None
        st.info("Upload a clear image of one fruit or vegetable.")

    analyze = st.button(
        "🔍 Analyze image",
        type="primary",
        use_container_width=True,
        disabled=image is None
    )

with right:
    st.subheader("Recognition result")
    if analyze and image is not None:
        with st.spinner("The CNN is analyzing your image..."):
            try:
                result = predict_image(image)
                st.session_state.history.insert(0, {
                    "Class": display_class_name(result["predicted_class"]),
                    "Confidence": result["confidence"]
                })
                st.session_state.history = st.session_state.history[:10]
                st.session_state.latest = result
            except FileNotFoundError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

    result = st.session_state.get("latest")
    if result:
        st.markdown(
            f'<div class="result"><h2>🍓 {display_class_name(result["predicted_class"])}</h2>'
            f'<p>Top prediction confidence: <b>{format_percent(result["confidence"])}</b></p></div>',
            unsafe_allow_html=True
        )
        st.progress(float(result["confidence"]))
        st.markdown("**Top 3 predictions**")
        df = pd.DataFrame(result["top_3"])
        df["class"] = df["class"].map(display_class_name)
        df["confidence (%)"] = (df["confidence"] * 100).round(2)
        st.bar_chart(df.set_index("class")["confidence (%)"])
        st.dataframe(
            df[["class", "confidence (%)"]].rename(columns={"class": "Class"}),
            hide_index=True,
            use_container_width=True
        )
    else:
        st.write("Your prediction will appear here.")

st.markdown("---")
st.subheader("Recent predictions")
if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
    history_df["Confidence"] = (history_df["Confidence"] * 100).round(1).astype(str) + "%"
    st.dataframe(history_df, hide_index=True, use_container_width=True)
else:
    st.caption("No predictions yet. Your latest 10 predictions will appear here.")

st.caption(
    "Disclaimer: this model recognizes only the classes included in its training data. "
    "Softmax confidence is not a guarantee of correctness. Educational use only."
)
