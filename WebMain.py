import streamlit as st

# ====================== PAGE CONFIG (MUST BE FIRST) ======================
st.set_page_config(page_title="Instagram Fake Detector", layout="wide")

import pandas as pd
import numpy as np
import os
import tempfile
import joblib
from paddleocr import PaddleOCR
import base64
from PIL import Image
from io import BytesIO

# ====================== PIPELINE ======================
try:
    import pipeline
except ImportError as e:
    raise ImportError("CRITICAL ERROR: 'pipeline.py' not found.") from e


def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


# ====================== LOGO ======================
logo_base64 = get_base64_image("assets/logomercu.png")

st.markdown(f"""
<style>
.logo-container {{
    display: flex;
    justify-content: center;
    margin-top: -70px;
}}
.logo-container img {{
    max-width: 120px;
}}
</style>
<div class="logo-container">
    <img src='data:image/png;base64,{logo_base64}'>
</div>
""", unsafe_allow_html=True)

# ====================== STYLE ======================
st.markdown("""
<style>
html { scroll-behavior: smooth; }
.stAppHeader { display: none; }

.hero {
    background: linear-gradient(90deg,#fcb045,#fd1d1d,#833ab4);
    padding: 80px 50px;
    border-radius: 22px;
    color: white;
    margin-bottom: 70px;
}
.hero h1 { font-size: 44px; font-weight: 700; }

.section { padding-top: 80px; margin-top: -80px; }
.soft-divider { height: 1px; margin: 80px 0; background: rgba(255,255,255,0.2); }

.about {
    background: #0e1117;
    padding: 60px;
    border-radius: 22px;
    border: 1px solid rgba(255,255,255,0.08);
}
.creator-item {
    background: rgba(255,255,255,0.05);
    padding: 8px 12px;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ====================== HERO ======================
st.markdown("""
<div class="hero">
    <h1>Instagram Fake Account Detector</h1>
    <a href="#upload">Get Started</a>
</div>
""", unsafe_allow_html=True)

# ====================== MODEL PATHS ======================
MODEL_DIR = "models"
RF_PATH = os.path.join(MODEL_DIR, "rf_multiclass.pkl")
XGB_PATH = os.path.join(MODEL_DIR, "xgb_multiclass.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_xgb.pkl")
LABELMAP_PATH = os.path.join(MODEL_DIR, "label_mapping.pkl")


@st.cache_resource
def load_models():
    return (
        joblib.load(RF_PATH),
        joblib.load(XGB_PATH),
        joblib.load(SCALER_PATH),
        joblib.load(LABELMAP_PATH),
    )


rf, xgb, scaler, label_map = load_models()

# ====================== OCR ======================
@st.cache_resource
def get_ocr():
    return PaddleOCR(
        text_detection_model_name="PP-OCRv5_server_det",
        text_recognition_model_name="PP-OCRv5_server_rec",
        lang="en"
    )


def convert_to_number(text):
    if not text:
        return 0
    text = str(text).upper().replace(",", "")
    try:
        if "K" in text:
            return int(float(text.replace("K", "")) * 1000)
        if "M" in text:
            return int(float(text.replace("M", "")) * 1_000_000)
        return int(float(text))
    except:
        return 0


# ====================== UPLOAD ======================
st.markdown('<div id="upload" class="section"></div>', unsafe_allow_html=True)
st.subheader("Upload Instagram Screenshot")

uploaded_file = st.file_uploader("Upload image...", type=["jpg", "jpeg", "png"])

# ====================== MAIN ANALYSIS ======================
if uploaded_file is None:
    st.info("Please upload an Instagram screenshot to begin analysis.")
else:
    with tempfile.TemporaryDirectory() as temp_dir:
        img_path = os.path.join(temp_dir, uploaded_file.name)
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        col1, col2 = st.columns(2)

        image = Image.open(BytesIO(uploaded_file.getvalue()))
        col1.image(image, caption="Uploaded Screenshot", use_column_width=True)

        with st.spinner("Running OCR..."):
            ocr = get_ocr()
            result = ocr.predict(img_path)
            res = result[0]
            res.save_to_json(temp_dir)
            res.save_to_img(temp_dir)

        json_files = [f for f in os.listdir(temp_dir) if f.endswith(".json")]
        json_path = os.path.join(temp_dir, json_files[0])

        img_out_files = [
            f for f in os.listdir(temp_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
            and f != uploaded_file.name
        ]

        if img_out_files:
            col2.image(
                os.path.join(temp_dir, img_out_files[0]),
                caption="OCR Result",
                use_column_width=True
            )

        st.subheader("Extracted Fields")
        texts = pipeline.load_texts(json_path)
        fields = pipeline.extract_fields_v2(texts)
        st.dataframe(pd.DataFrame([fields]))

        feature_row = pd.DataFrame([{
            "usernameDigitCount": sum(c.isdigit() for c in fields.get("Username", "")),
            "userMediaCount": convert_to_number(fields.get("Posts", "0")),
            "userFollowerCount": convert_to_number(fields.get("Followers", "0")),
            "userFollowingCount": convert_to_number(fields.get("Following", "0")),
            "userBiographyLength": len(fields.get("Bio", ""))
        }])

        st.subheader("Engineered Features")
        st.dataframe(feature_row)

        rf_prob = rf.predict_proba(feature_row)[0]
        xgb_prob = xgb.predict_proba(scaler.transform(feature_row))[0]
        avg_prob = (rf_prob + xgb_prob) / 2

        final_class = np.argmax(avg_prob)
        final_label = label_map[final_class]

        st.success(f"Final Classification: {final_label} ({avg_prob[final_class]*100:.2f}%)")

# ====================== FOOTER ======================
st.markdown("""
<div style="text-align:center; color:#9ca3af; margin-top:40px;">
    Built with Streamlit | © 2025 Instagram Fake Account Detector
</div>
""", unsafe_allow_html=True)
