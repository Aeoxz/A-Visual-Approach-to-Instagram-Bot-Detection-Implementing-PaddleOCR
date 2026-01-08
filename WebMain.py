import streamlit as st
# ====================== PAGE CONFIG ======================
st.set_page_config(page_title="Instagram Fake Detector", layout="wide")

import pandas as pd
import numpy as np
import os
import tempfile
import joblib
from paddleocr import PaddleOCR
import base64

# ====================== PIPELINE ======================
try:
    import pipeline
except ImportError:
    st.error("CRITICAL ERROR: 'pipeline.py' not found.")
    st.stop()

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Convert logo ke base64
logo_base64 = get_base64_image("assets/logomercu.png")

st.markdown(f"""
<style>
.logo-container {{
    display: flex;
    justify-content: center;
    align-items: flex-start;
    margin-top: -70px;
}}

.logo-container img {{
    max-width: 120px;
}}
</style>

<div class="logo-container">
    <img src='data:image/png;base64,{logo_base64}' alt="Logo Mercu">
</div>
""", unsafe_allow_html=True)
# ====================== STYLE ======================
st.markdown("""
<style>
html { scroll-behavior: smooth; }

.stAppHeader {
    display: none;
}

.st-emotion-cache-zy6yx3 {
    width: 100%;
    padding: 6rem 1rem 3rem;
    max-width: initial;
    min-width: auto;
}

.hero {
    background: linear-gradient(
        135deg,
        #0f172a 0%,
        #111827 45%,
        #1e1b4b 100%
    );
    padding: 90px 70px;
    border-radius: 24px;
    color: #f9fafb;
    max-width: 1300px;   /* ⬅️ UBAH LEBAR DI SINI */
    margin: 0 auto 80px auto;
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.06);
}
                  
.hero:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 40px rgba(139,92,246,0.6);
    border-color: rgba(139,92,246,0.6);
}

/* subtle light accent */
.hero::before {
    content: "";
    position: absolute;
    inset: 0;
    background: radial-gradient(
        circle at 20% 20%,
        rgba(139,92,246,0.15),
        transparent 20%
    );
    pointer-events: none;
}

.hero h1 {
    font-size: 46px;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-bottom: 12px;
}

.hero p {
    font-size: 16px;
    color: #d1d5db;
    max-width: 520px;
    margin-bottom: 36px;
}

.hero-buttons {
    display: flex;
    gap: 14px;
}

.hero-buttons a {
    text-decoration: none;
    padding: 12px 30px;
    border-radius: 10px;
    font-weight: 600;
    font-size: 14px;
    letter-spacing: 0.3px;
    transition: all 0.25s ease;
}

/* Primary button */
.btn-primary {
    background: #111827;
    color: #ffffff !important;
    border: 1px solid rgba(255,255,255,0.15);
}

.btn-primary:hover {
    background: #1e1b4b;
    border-color: rgba(139,92,246,0.6);
}

/* Secondary button */
.btn-secondary {
    background: transparent;
    color: #e5e7eb !important;
    border: 1px solid rgba(255,255,255,0.15);
}

.btn-secondary:hover {
    background: rgba(139,92,246,0.15);
    border-color: rgba(139,92,246,0.6);
}
            

.section {
    padding-top: 80px;
    margin-top: -80px;
}

.soft-divider {
    height: 1px;
    margin: 90px 0 70px 0;
    background: linear-gradient(
        90deg,
        transparent,
        rgba(255,255,255,0.25),
        transparent
    );
}

.about {
    position: relative;
    background: radial-gradient(
        circle at 20% 20%,
        rgba(139,92,246,0.15),
        transparent 20%
    );
    padding: 60px 60px 40px;
    border-radius: 22px;
    margin-top: 20px;
    border: 1px solid rgba(255,255,255,0.08);
    transition: all 0.4s ease;
}

.about:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 40px rgba(139,92,246,0.6);
    border-color: rgba(139,92,246,0.6);
}

.about::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    border-radius: 22px 22px 0 0;
    background: linear-gradient(
        90deg,
        #fcb045 0%,
        #fd1d1d 50%,
        #833ab4 100%
    );
    background-size: 200% 100%;
    animation: gradientShift 3s ease infinite;
}

@keyframes gradientShift {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

.about h2 {
    color: #ffffff;
    font-size: 26px;
    font-weight: 700;
    margin-bottom: 18px;
    text-align: center;
    animation: fadeInDown 0.8s ease;
}

@keyframes fadeInDown {
    from {
        opacity: 0;
        transform: translateY(-20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.about p {
    color: #d1d5db;
    font-size: 15px;
    line-height: 1.8;
    max-width: 900px;
    margin: 0 auto;
    text-align: justify;
    text-align-last: center;
    animation: fadeIn 1s ease 0.2s both;
}

@keyframes fadeIn {
    from {
        opacity: 0;
    }
    to {
        opacity: 1;
    }
}
            
.creators {
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid rgba(255,255,255,0.15);
}

.creators h3 {
    color: #ffffff;
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 12px;
    text-align: center;
    animation: fadeIn 1s ease 0.4s both;
}

.creator-list {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 10px;
    color: #d1d5db;
    font-size: 14px;
    max-width: 900px;
    margin: 0 auto;
}

.creator-item {
    padding: 8px 12px;
    background: rgba(255,255,255,0.05);
    border-radius: 8px;
    text-align: center;
    transition: all 0.3s ease;
    animation: fadeInUp 0.6s ease both;
}

.creator-item:nth-child(1) { animation-delay: 0.5s; }
.creator-item:nth-child(2) { animation-delay: 0.6s; }
.creator-item:nth-child(3) { animation-delay: 0.7s; }

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.creator-item:hover {
    background: rgba(139,92,246,0.16);
    transform: translateY(-3px);
    box-shadow: 0 5px 15px rgba(139,92,246,0.6);
}

.supervisor-section {
    margin-top: 35px;
}

.supervisor-item {
    background: rgba(255,255,255,0.05);
    animation-delay: 0.8s;
}

.supervisor-item:hover {
    background: rgba(131, 58, 180, 0.15);
    box-shadow: 0 5px 15px rgba(139,92,246,0.6);
}

.copyright {
    margin-top: 30px;
    font-size: 13px;
    color: #a0aec0;
    text-align: center;
}
    
.footer {
    margin-top: 35px;
    padding-top: 20px;
    text-align: center;
}

.footer img {
    height: 55px;
    margin-bottom: 10px;
}

.footer p {
    font-size: 13px;
    color: #9ca3af;
}

</style>
""", unsafe_allow_html=True)

# ====================== HERO ======================
st.markdown("""
<div class="hero">
    <h1>Instagram Fake Account Detector</h1>
    <p>
        Curious about the authenticity of an Instagram account? <br>
        With just a single screenshot, our system can analyze profile information <br>
and provide an evidence-based classification in seconds.
            </p>
    <div class="hero-buttons">
        <a href="#upload" class="btn-primary">Get Started Now</a>
        <a href="#about" class="btn-secondary">About Us</a>
    </div>
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
    rf_model = joblib.load(RF_PATH)
    xgb_model = joblib.load(XGB_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_map = joblib.load(LABELMAP_PATH)
    return rf_model, xgb_model, scaler, label_map

rf, xgb, scaler, label_map = load_models()

# ====================== OCR ======================
@st.cache_resource
def get_ocr():
    return PaddleOCR(
text_detection_model_name="PP-OCRv5",  
    text_recognition_model_name="PP-OCRv5",  
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

uploaded_file = st.file_uploader(
    "Upload image...",
    type=["jpg", "jpeg", "png"]
)

# ====================== MAIN ANALYSIS ======================
if uploaded_file is None:
    st.info("Please upload an Instagram screenshot to begin analysis.")
else:
    with tempfile.TemporaryDirectory() as temp_dir:
        img_path = os.path.join(temp_dir, uploaded_file.name)
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        col1, col2 = st.columns(2)
        col1.image(uploaded_file, caption="Uploaded Screenshot", use_column_width=True)

        with st.spinner("Running OCR..."):
            ocr = get_ocr()
            result = ocr.predict(img_path)
            res = result[0]
            res.save_to_json(temp_dir)
            res.save_to_img(temp_dir)

        json_files = [f for f in os.listdir(temp_dir) if f.endswith(".json")]
        json_path = os.path.join(temp_dir, json_files[0])

        img_out_files = [f for f in os.listdir(temp_dir) if f.endswith((".jpg", ".jpeg", ".png")) and f != uploaded_file.name]
        if img_out_files:
            col2.image(os.path.join(temp_dir, img_out_files[0]), caption="OCR Result", use_column_width=True)

        st.subheader("Extracted Fields")
        texts = pipeline.load_texts(json_path)
        fields = pipeline.extract_fields_v2(texts)
        username = fields.get("Username", "")
        username_digit_count = sum(c.isdigit() for c in username)
        st.dataframe(pd.DataFrame([fields]))

        posts = convert_to_number(fields.get("Posts", "0"))
        followers = convert_to_number(fields.get("Followers", "0"))
        following = convert_to_number(fields.get("Following", "0"))
        bio_len = len(fields.get("Bio", ""))

        feature_row = pd.DataFrame([{
            "usernameDigitCount": username_digit_count,
            "userMediaCount": posts,
            "userFollowerCount": followers,
            "userFollowingCount": following,
            "userBiographyLength": bio_len
        }])

        st.subheader("Engineered Features")
        st.dataframe(feature_row)

        st.subheader("Prediction Result")

        rf_prob = rf.predict_proba(feature_row)[0]
        scaled = scaler.transform(feature_row)
        xgb_prob = xgb.predict_proba(scaled)[0]

        avg_prob = (rf_prob + xgb_prob) / 2
        final_class = np.argmax(avg_prob)
        final_label = label_map[final_class]

        df_prob = pd.DataFrame({
            "Class": [label_map[i] for i in range(len(avg_prob))],
            "RF Probability": rf_prob,
            "XGB Probability": xgb_prob,
            "Average": avg_prob
        })

        st.dataframe(
            df_prob.style.format({
                "RF Probability": "{:.4f}",
                "XGB Probability": "{:.4f}",
                "Average": "{:.4f}"
            })
        )

        st.markdown("### Final Decision")
        st.success(f"Final Classification: {final_label} ({avg_prob[final_class]*100:.2f}%)")

        st.bar_chart(pd.DataFrame({
            "Class": [label_map[i] for i in range(len(avg_prob))],
            "Probability": avg_prob
        }).set_index("Class"))

        st.success("Prediction Completed.")

# ====================== DIVIDER ======================
st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

# ====================== ABOUT ======================
st.markdown('<div id="about" class="section"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="about">
    <h2>About This Project</h2>
    <p>
       Instagram Fake Account Detector is a smart system designed to help identify fake Instagram accounts by analyzing profile information. It extracts key details from profile screenshots using OCR technology and evaluates them with machine learning models to provide reliable and transparent classification results. This project was developed for academic and research purposes.
    </p>
     <div class="creators">
        <h3>Project Creators</h3>
        <div class="creator-list">
            <div class="creator-item">Muhammad Ikhsanudin</div>
            <div class="creator-item">Azka Faiq Suharyanto</div>
            <div class="creator-item">Annas Wicaksono</div>
        </div>
    </div>
    <div class="creators supervisor-section">
        <h3>Supervisor</h3>
        <div class="creator-list">
            <div class="creator-item supervisor-item">Afiyati, Dr., S.Si, MT</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    <p>© 2025 Instagram Fake Account Detector</p>
</div>
""", unsafe_allow_html=True)





