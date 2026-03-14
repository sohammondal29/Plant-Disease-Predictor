import os
import json
import sqlite3
import gc
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import gdown
from PIL import Image
from fpdf import FPDF

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Use MobileNetV2 utilities for leaf detection
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# ---------------- CONFIG & PATHS ---------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "plant_disease_prediction_model.h5")
CLASS_FILE = os.path.join(BASE_DIR, "class_indices.json")
DB_FILE = os.path.join(BASE_DIR, "history.db")
EXAMPLE_FOLDER = os.path.join(BASE_DIR, "Examples")
FILE_ID = "1akhIIwfWmp3aD-gGl9nGaoY9uRs2iorP"

# ---------------- DOWNLOAD MODEL ---------------- #
def download_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        try:
            with st.spinner("Downloading AI model (this may take a minute)..."):
                gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
        except Exception as e:
            st.error(f"Error downloading model: {e}")

# Call download before loading starts
download_model()

# ---------------- CACHED MODEL LOADING ---------------- #
@st.cache_resource
def load_models():
    """Load both models once and cache them to save RAM."""
    # 1. Load Custom Disease Model
    disease_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    
    # 2. Load Leaf Detector (MobileNetV2)
    leaf_detector = MobileNetV2(weights="imagenet")
    
    return disease_model, leaf_detector

# Initialize models
try:
    disease_model, leaf_detector = load_models()
except Exception as e:
    st.error(f"Failed to load models: {e}")

# ---------------- LOAD CLASS INDICES ---------------- #
@st.cache_data
def load_class_indices():
    with open(CLASS_FILE) as f:
        return json.load(f)

class_indices = load_class_indices()

# ---------------- DATABASE SETUP ---------------- #
def init_db():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE IF NOT EXISTS history(time TEXT, plant TEXT, disease TEXT, confidence REAL)"
    )
    conn.commit()
    return conn

conn = init_db()

# ---------------- HELPER FUNCTIONS ---------------- #

def create_report(plant, disease, confidence, severity_lvl):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, "Plant Disease Diagnosis Report", ln=True, align="C")
    pdf.ln(10)
    pdf.cell(200, 10, f"Plant: {plant}", ln=True)
    pdf.cell(200, 10, f"Disease: {disease}", ln=True)
    pdf.cell(200, 10, f"Confidence: {confidence:.2f}%", ln=True)
    pdf.cell(200, 10, f"Severity: {severity_lvl}", ln=True)
    
    report_file = "report.pdf"
    pdf.output(report_file)
    return report_file

def preprocess_image(image, size=(224, 224)):
    img = image.resize(size).convert("RGB")
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def detect_leaf(image):
    """Checks if the image contains a leaf using MobileNetV2."""
    img = image.resize((224, 224)).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    
    preds = leaf_detector.predict(arr, verbose=0)
    decoded = decode_predictions(preds, top=5)[0]
    
    plant_keywords = ["leaf", "tree", "plant", "flower", "corn", "grape", "apple", "pot", "buckeye"]
    
    is_leaf = any(any(word in label.lower() for word in plant_keywords) for _, label, _ in decoded)
    gc.collect() # Free memory
    return is_leaf

def calculate_severity(image):
    img = np.array(image)
    gray = np.mean(img, axis=2)
    infected = np.sum(gray < 120)
    total = gray.size
    ratio = infected / total
    
    if ratio < 0.1: return "Low"
    elif ratio < 0.3: return "Moderate"
    else: return "Severe"

def save_history(plant, disease, conf):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cur = conn.cursor()
    cur.execute("INSERT INTO history VALUES(?,?,?,?)", (current_time, plant, disease, conf))
    conn.commit()

# ---------------- UI LAYOUT ---------------- #
st.set_page_config(page_title="Plant Disease Predictor", layout="wide")

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg,#0f2027,#203a43,#2c5364); color:white; }
    h1, h2, h3, label { color:white !important; }
</style>
""", unsafe_allow_html=True)

st.title("🌿 AI Plant Disease Detection System")

input_choice = st.sidebar.radio("Select Image Source", ["Upload Leaf Image", "Use Example Image", "Camera"])

image = None

if input_choice == "Use Example Image":
    if os.path.exists(EXAMPLE_FOLDER):
        example_files = sorted(os.listdir(EXAMPLE_FOLDER))
        selected_example = st.selectbox("Select Example", example_files)
        if selected_example:
            image = Image.open(os.path.join(EXAMPLE_FOLDER, selected_example)).convert("RGB")
    else:
        st.error("Example folder not found.")

elif input_choice == "Upload Leaf Image":
    uploaded = st.file_uploader("Browse Leaf Image", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")

elif input_choice == "Camera":
    uploaded = st.camera_input("Take Leaf Photo")
    if uploaded:
        image = Image.open(uploaded).convert("RGB")

# ---------------- PREDICTION LOGIC ---------------- #
if image is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        if not detect_leaf(image):
            st.error("⚠️ The image does not appear to contain a plant leaf. Please try again.")
        else:
            with st.spinner("Analyzing..."):
                processed_img = preprocess_image(image)
                predictions = disease_model.predict(processed_img, verbose=0)
                
                idx = np.argmax(predictions)
                conf = float(np.max(predictions)) * 100
                label = class_indices[str(idx)]
                
                try:
                    plant, disease = label.split("___")
                    disease = disease.replace("_", " ")
                except:
                    plant, disease = "Unknown", label
                
                sev = calculate_severity(image)
                save_history(plant, disease, conf)

                st.success(f"**Plant:** {plant}")
                st.error(f"**Disease:** {disease}")
                st.info(f"**Confidence:** {conf:.2f}%")
                st.warning(f"**Severity Level:** {sev}")

                # Top 3 list
                st.subheader("Top Predictions")
                top3_indices = predictions[0].argsort()[-3:][::-1]
                for i in top3_indices:
                    name = class_indices[str(i)].replace("___", " - ").replace("_", " ")
                    c = predictions[0][i] * 100
                    st.write(f"{name}: {c:.2f}%")

                # PDF Report
                report_path = create_report(plant, disease, conf, sev)
                with open(report_path, "rb") as f:
                    st.download_button("Download Diagnosis Report", f, "report.pdf", "application/pdf")

# ---------------- HISTORY TABLE ---------------- #
st.markdown("---")
st.subheader("Recent Prediction History")
try:
    history_df = pd.read_sql_query("SELECT * FROM history ORDER BY time DESC LIMIT 10", conn)
    st.dataframe(history_df, use_container_width=True)
except:
    st.write("No history available yet.")

st.markdown("<div style='text-align:center;'>Created by <b>Soham Mondal</b></div>", unsafe_allow_html=True)
