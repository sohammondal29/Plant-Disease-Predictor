import os
import json
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from fpdf import FPDF
import gdown

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    # Optional local fallback if you run it on a machine with TensorFlow installed
    from tensorflow.lite.python.interpreter import Interpreter


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Plant Disease Predictor",
    page_icon="🌿",
    layout="wide"
)

# ---------------- THEME ----------------

st.markdown("""
<style>

/* LIGHT MODE */
.stApp {
    background: linear-gradient(135deg,#e8f5e9,#c8e6c9);
    color:#0b3d2e;
}

/* DARK MODE */
@media (prefers-color-scheme: dark) {
    .stApp {
        background: linear-gradient(135deg,#06281c,#0f5132,#198754);
        color:#c9ffd5;
    }
}

/* TITLE */
h1 {
    font-size:64px;
    font-weight:800;
    letter-spacing:-2px;
    text-align:center;
}

/* TAGLINE */
.tagline {
    font-size:20px;
    text-align:center;
}

/* ACCENT */
.accent { color:#1b5e20; }

@media (prefers-color-scheme: dark) {
    .accent { color:#6fff8c; }
}

/* SECTION */
.section-title {
    font-size:28px;
    margin-top:40px;
}

/* IMAGE STYLE */
img {
    border-radius:12px;
    transition: transform 0.3s ease;
}

img:hover {
    transform: scale(1.05);
}

/* DOWNLOAD BUTTON */
.stDownloadButton button {
    background-color:#0d6efd !important;
    color:white !important;
    border-radius:8px !important;
    font-weight:bold !important;
}

/* FOOTER */
.footer {
    margin-top:60px;
    padding-top:20px;
    border-top:1px solid #c8e6c9;
    font-size:14px;
    text-align:center;
}

@media (prefers-color-scheme: dark) {
    .footer {
        border-top:1px solid #1b4332;
    }
}

</style>
""", unsafe_allow_html=True)


# ---------------- PATHS ----------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "plant_model.tflite")
CLASS_FILE = os.path.join(BASE_DIR, "class_indices.json")
DB_FILE = os.path.join(BASE_DIR, "history.db")
EXAMPLE_FOLDER = os.path.join(BASE_DIR, "Examples")

FILE_ID = "1Y8dRQTEE_16c8UEjRFdoppBMi-cZrzJ7"


# ---------------- DOWNLOAD MODEL ----------------

def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading AI model..."):
            gdown.download(id=FILE_ID, output=MODEL_PATH, quiet=False)


download_model()


# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_disease_model():
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]["shape"]
    input_dtype = input_details[0]["dtype"]

    # Expected shape: [1, height, width, channels]
    height = int(input_shape[1])
    width = int(input_shape[2])

    return interpreter, input_details, output_details, (height, width), input_dtype


try:
    interpreter, input_details, output_details, INPUT_SIZE, INPUT_DTYPE = load_disease_model()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()


# ---------------- CLASS LABELS ----------------

try:
    with open(CLASS_FILE, "r") as f:
        class_indices = json.load(f)
except Exception as e:
    st.error(f"Could not load class labels: {e}")
    st.stop()


# ---------------- DATABASE ----------------

conn = sqlite3.connect(DB_FILE, check_same_thread=False)
cur = conn.cursor()

cur.execute(
    "CREATE TABLE IF NOT EXISTS history(time TEXT, plant TEXT, disease TEXT, confidence REAL)"
)
conn.commit()


# ---------------- PDF REPORT ----------------

def create_report(plant, disease, confidence, severity_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)

    pdf.cell(200, 10, "Plant Disease Diagnosis Report", ln=True, align="C")
    pdf.ln(10)

    pdf.cell(200, 10, f"Plant: {plant}", ln=True)
    pdf.cell(200, 10, f"Disease: {disease}", ln=True)
    pdf.cell(200, 10, f"Confidence: {confidence:.2f}%", ln=True)
    pdf.cell(200, 10, f"Severity: {severity_text}", ln=True)

    file_path = os.path.join(BASE_DIR, "report.pdf")
    pdf.output(file_path)
    return file_path


# ---------------- IMAGE PROCESS ----------------

def preprocess_image(image, size=(224, 224), dtype=np.float32):
    image = image.resize(size).convert("RGB")
    arr = np.array(image)

    if dtype == np.float32:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(dtype)

    arr = np.expand_dims(arr, axis=0)
    return arr


# ---------------- PREDICT ----------------

def predict_disease(image):
    img = preprocess_image(image, size=INPUT_SIZE, dtype=INPUT_DTYPE)

    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()

    pred = interpreter.get_tensor(output_details[0]["index"])

    # Dequantize if needed
    out_dtype = output_details[0]["dtype"]
    quant_params = output_details[0].get("quantization", (0.0, 0))
    scale, zero_point = quant_params

    if out_dtype != np.float32 and scale and scale > 0:
        pred = scale * (pred.astype(np.float32) - zero_point)

    index = int(np.argmax(pred))
    conf = float(np.max(pred)) * 100

    label = class_indices.get(str(index), f"Class_{index}")
    return label, conf, pred


# ---------------- LEAF CHECK ----------------

def detect_leaf(image):
    """
    Lightweight heuristic so the app works without TensorFlow.
    It is intentionally lenient.
    """
    img = image.resize((224, 224)).convert("RGB")
    arr = np.array(img).astype(np.float32)

    r_mean = arr[:, :, 0].mean()
    g_mean = arr[:, :, 1].mean()
    b_mean = arr[:, :, 2].mean()

    green_dominance = g_mean - (r_mean + b_mean) / 2

    # Lenient threshold: most leaf images should pass
    return green_dominance > -10


# ---------------- SAVE HISTORY ----------------

def save_history(plant, disease, conf):
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cur.execute(
        "INSERT INTO history VALUES(?,?,?,?)",
        (time, plant, disease, conf)
    )
    conn.commit()


def load_history():
    try:
        return pd.read_sql_query(
            "SELECT * FROM history ORDER BY datetime(time) DESC",
            conn
        )
    except Exception:
        return pd.DataFrame(columns=["time", "plant", "disease", "confidence"])


# ---------------- SEVERITY ----------------

def severity(image):
    img = np.array(image.convert("RGB"))
    gray = np.mean(img, axis=2)

    infected = np.sum(gray < 120)
    ratio = infected / gray.size

    if ratio < 0.1:
        return "Low"
    elif ratio < 0.3:
        return "Moderate"
    else:
        return "Severe"


# ---------------- HERO ----------------

st.markdown("<h1><span class='accent'>🌿 Plant</span> Disease Predictor</h1>", unsafe_allow_html=True)

st.markdown(
    "<div class='tagline'>Scan a leaf. Detect the disease. Protect your plants.</div>",
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)


# ---------------- IMAGE INPUT ----------------

input_choice = st.radio(
    "Select Image Source",
    ["Use Example Image", "Upload Leaf Image", "Camera"]
)

image = None

if input_choice == "Use Example Image":
    if os.path.exists(EXAMPLE_FOLDER):
        files = sorted(
            f for f in os.listdir(EXAMPLE_FOLDER)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if files:
            example = st.selectbox("Choose example", files)
            image = Image.open(os.path.join(EXAMPLE_FOLDER, example))
        else:
            st.warning("No example images found in the Examples folder.")
    else:
        st.warning("Examples folder not found.")

elif input_choice == "Upload Leaf Image":
    uploaded = st.file_uploader("Upload leaf", type=["jpg", "png", "jpeg"])
    if uploaded:
        image = Image.open(uploaded)

elif input_choice == "Camera":
    cam = st.camera_input("Take photo")
    if cam:
        image = Image.open(cam)


# ---------------- PREDICTION ----------------

if image is not None:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(image, width=350)

    with col2:
        if not detect_leaf(image):
            st.warning("This image may not contain a leaf, but you can still try prediction.")
        label, conf, pred = predict_disease(image)

        if "___" in label:
            plant, disease = label.split("___", 1)
            disease = disease.replace("_", " ")
        else:
            plant, disease = "Unknown", label.replace("_", " ")

        sev = severity(image)
        save_history(plant, disease, conf)

        st.success(f"Plant: {plant}")

        if "healthy" in disease.lower():
            st.success(f"Disease: {disease}")
        else:
            st.error(f"Disease: {disease}")

        st.info(f"Confidence: {conf:.2f}%")
        st.warning(f"Severity: {sev}")

        st.markdown("<div class='section-title'>Top Predictions</div>", unsafe_allow_html=True)

        top3 = pred[0].argsort()[-3:][::-1]

        for i in top3:
            name = class_indices.get(str(i), f"Class_{i}")
            if "___" in name:
                p, d = name.split("___", 1)
                d = d.replace("_", " ")
            else:
                p, d = "Unknown", name.replace("_", " ")
            c = pred[0][i] * 100
            st.write(f"{p} — {d}: {c:.2f}%")

        report = create_report(plant, disease, conf, sev)

        with open(report, "rb") as f:
            st.download_button(
                "Download Diagnosis Report",
                data=f,
                file_name="plant_disease_report.pdf"
            )


# ---------------- HISTORY ----------------

st.markdown("<br>", unsafe_allow_html=True)
st.subheader("Prediction History")
st.dataframe(load_history(), use_container_width=True)


# ---------------- FOOTER ----------------

st.markdown(
    """
    <div class="footer">
    Created by <b>Soham Mondal</b><br>
    Contact: <b>sohammondal29@gmail.com</b>
    </div>
    """,
    unsafe_allow_html=True
)
