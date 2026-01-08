import streamlit as st
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
    background: linear-gradient(90deg, #fcb045 0%, #fd1d1d 50%, #833ab4 100%);
    padding: 80px 50px;
    border-radius: 22px;
    color: white;
    margin-bottom: 70px;
}

.hero h1 {
    font-size: 44px;
    font-weight: 700;
    margin-bottom: 8px;
}

.hero p {
    font-size: 17px;
    opacity: 0.95;
    margin-bottom: 32px;
}

.hero-buttons {
    display: flex;
    gap: 16px;
}

.hero-buttons a {
    text-decoration: none;
    padding: 12px 28px;
    border-radius: 10px;
    font-weight: 600;
    cursor: pointer;
    display: inline-block;
    transition: all 0.25s ease;
}

.btn-primary {
    background: #0f0f0f;
    color: #ffffff !important;
}

.btn-secondary {
    background: rgba(255,255,255,0.95);
    color: #020202 !important;
}

.btn-primary:hover {
    color: #ffffff;
    background: #fd1d1d;
    transition: all 0.25s ease;
}

.btn-secondary:hover {
    color: white;
    background: #fd1d1d;
    transition: all 0.25s ease;
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
    background: linear-gradient(
        180deg,
        rgba(14,17,23,0.95) 0%,
        rgba(14,17,23,1) 100%
    );
    padding: 60px 60px 40px;
    border-radius: 22px;
    margin-top: 20px;
    border: 1px solid rgba(255,255,255,0.08);
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
}

.about h2 {
    color: #ffffff;
    font-size: 26px;
    font-weight: 700;
    margin-bottom: 18px;
}

.about p {
    color: #d1d5db;
    font-size: 15px;
    line-height: 1.8;
    max-width: 900px;
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
}

.creator-list {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 10px;
    color: #d1d5db;
    font-size: 14px;
}

.creator-item {
    padding: 8px 12px;
    background: rgba(255,255,255,0.05);
    border-radius: 8px;
}

.copyright {
    margin-top: 25px;
    font-size: 13px;
    color: #9ca3af;
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

# col1, col2, col3 = st.columns([1, 2, 1])
# with col2:
#     st.image("assets/logomercu.png", width=100)

# ====================== HERO ======================
st.markdown("""
<div class="hero">
    <h1>Instagram Fake Account Detector</h1>
    <p></p>
    <div class="hero-buttons">
        <a href="#upload" class="btn-primary">Get Started</a>
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

uploaded_file = st.file_uploader(
    "Upload image...",
    type=["jpg", "jpeg", "png"]
)

# ====================== MAIN ANALYSIS (IDENTIK KODE AWAL) ======================
if uploaded_file is None:
    st.info("Please upload an Instagram screenshot to begin analysis.")
else:
    with tempfile.TemporaryDirectory() as temp_dir:
        img_path = os.path.join(temp_dir, uploaded_file.name)
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        col1, col2 = st.columns(2)
        col1.image(uploaded_file, caption="Uploaded Screenshot", use_container_width=True)

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
            col2.image(os.path.join(temp_dir, img_out_files[0]), caption="OCR Result", use_container_width=True)

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
        Instagram Fake Account Detector merupakan sistem analisis berbasis
        machine learning yang dirancang untuk mengidentifikasi akun Instagram
        palsu melalui pemanfaatan metadata profil.
    </p>
    <p>
        Sistem ini mengintegrasikan teknologi Optical Character Recognition
        (PP-OCRv5) untuk mengekstraksi informasi dari tangkapan layar profil,
        yang kemudian dianalisis menggunakan algoritma Random Forest dan
        XGBoost dalam skema ensemble.
    </p>
    <p>
        Proyek ini dikembangkan untuk kebutuhan akademik dan penelitian,
        dengan penekanan pada keterjelasan fitur, konsistensi prediksi,
        serta transparansi proses klasifikasi.
    </p>
     <div class="creators">
        <h3>Project Creators</h3>
        <div class="creator-list">
            <div class="creator-item">👤 Muhammad Ikhsanudin</div>
            <div class="creator-item">👤 Azka Faiq</div>
            <div class="creator-item">👤 Annas Wicaksono</div>
        </div>
    </div>
    <div class="creators">
        <h3>supervisor</h3>
        <div class="creator-list">
            <div class="creator-item">👩‍🏫 Ibu Afiyati</div>
        </div>
</div>   
""", unsafe_allow_html=True)


# st.markdown('<footer id="footer" class="footer"></footer>', unsafe_allow_html=True)
st.markdown("""
<div class="footer">
    <p>Built with Streamlit | © 2025 Instagram Fake Account Detector</p>
</div>

""", unsafe_allow_html=True)
