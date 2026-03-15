import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import numpy as np
import tensorflow as tf
import streamlit as st
import pandas as pd
import sqlite3
from PIL import Image
from datetime import datetime
from fpdf import FPDF
import gdown

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions


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

/* BLUE DOWNLOAD BUTTON */

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

MODEL_PATH = os.path.join(BASE_DIR,"plant_model.tflite")
CLASS_FILE = os.path.join(BASE_DIR,"class_indices.json")
DB_FILE = os.path.join(BASE_DIR,"history.db")
EXAMPLE_FOLDER = os.path.join(BASE_DIR,"Examples")

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

    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return interpreter,input_details,output_details


interpreter,input_details,output_details = load_disease_model()


# ---------------- LEAF DETECTOR ----------------

@st.cache_resource
def load_leaf_detector():
    return MobileNetV2(weights="imagenet")


# ---------------- CLASS LABELS ----------------

with open(CLASS_FILE) as f:
    class_indices = json.load(f)


# ---------------- DATABASE ----------------

conn = sqlite3.connect(DB_FILE,check_same_thread=False)
cur = conn.cursor()

cur.execute(
"CREATE TABLE IF NOT EXISTS history(time TEXT,plant TEXT,disease TEXT,confidence REAL)"
)

conn.commit()


# ---------------- PDF REPORT ----------------

def create_report(plant,disease,confidence,severity):

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial",size=14)

    pdf.cell(200,10,"Plant Disease Diagnosis Report",ln=True,align="C")
    pdf.ln(10)

    pdf.cell(200,10,f"Plant: {plant}",ln=True)
    pdf.cell(200,10,f"Disease: {disease}",ln=True)
    pdf.cell(200,10,f"Confidence: {confidence:.2f}%",ln=True)
    pdf.cell(200,10,f"Severity: {severity}",ln=True)

    file="report.pdf"
    pdf.output(file)

    return file


# ---------------- IMAGE PROCESS ----------------

def preprocess_image(image,size=(224,224)):

    image=image.resize(size).convert("RGB")

    img=np.array(image,dtype=np.float32)/255.0
    img=np.expand_dims(img,axis=0)

    return img


# ---------------- PREDICT ----------------

def predict_disease(image):

    img=preprocess_image(image)

    interpreter.set_tensor(input_details[0]['index'],img)
    interpreter.invoke()

    pred=interpreter.get_tensor(output_details[0]['index'])

    index=np.argmax(pred)
    conf=float(np.max(pred))*100

    label=class_indices[str(index)]

    return label,conf,pred


# ---------------- LEAF CHECK ----------------

def detect_leaf(image):

    leaf_model = load_leaf_detector()

    img=image.resize((224,224)).convert("RGB")

    arr=np.array(img,dtype=np.float32)
    arr=np.expand_dims(arr,axis=0)
    arr=preprocess_input(arr)

    preds=leaf_model.predict(arr,verbose=0)

    decoded=decode_predictions(preds,top=5)[0]

    keywords=["leaf","plant","tree","flower","corn","grape","apple"]

    for _,label,_ in decoded:
        for word in keywords:
            if word in label.lower():
                return True

    return False


# ---------------- SAVE HISTORY ----------------

def save_history(plant,disease,conf):

    time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cur.execute(
        "INSERT INTO history VALUES(?,?,?,?)",
        (time,plant,disease,conf)
    )

    conn.commit()


def load_history():

    return pd.read_sql_query(
        "SELECT * FROM history ORDER BY datetime(time) DESC",
        conn
    )


# ---------------- SEVERITY ----------------

def severity(image):

    img=np.array(image)

    gray=np.mean(img,axis=2)

    infected=np.sum(gray<120)

    ratio=infected/gray.size

    if ratio<0.1:
        return "Low"
    elif ratio<0.3:
        return "Moderate"
    else:
        return "Severe"


# ---------------- HERO ----------------

st.markdown("<h1><span class='accent'>Plant</span> Disease Predictor</h1>",unsafe_allow_html=True)

st.markdown(
"<div class='tagline'>Scan a leaf. Detect the disease. Protect your plants.</div>",
unsafe_allow_html=True
)

st.markdown("<br>",unsafe_allow_html=True)


# ---------------- IMAGE INPUT ----------------

input_choice=st.radio(
"Select Image Source",
["Use Example Image","Upload Leaf Image","Camera"]
)

image=None

if input_choice=="Use Example Image":

    files=sorted(os.listdir(EXAMPLE_FOLDER))

    example=st.selectbox("Choose example",files)

    image=Image.open(os.path.join(EXAMPLE_FOLDER,example))

elif input_choice=="Upload Leaf Image":

    uploaded=st.file_uploader("Upload leaf",type=["jpg","png","jpeg"])

    if uploaded:
        image=Image.open(uploaded)

elif input_choice=="Camera":

    cam=st.camera_input("Take photo")

    if cam:
        image=Image.open(cam)


# ---------------- PREDICTION ----------------

if image is not None:

    col1,col2=st.columns([1,2])

    with col1:
        st.image(image,width=350)

    with col2:

        if not detect_leaf(image):

            st.error("Image does not appear to contain a leaf.")

        else:

            label,conf,pred=predict_disease(image)

            plant,disease=label.split("___")
            disease=disease.replace("_"," ")

            sev=severity(image)

            save_history(plant,disease,conf)

            st.success(f"Plant: {plant}")

            if "healthy" in disease.lower():
                st.success(f"Disease: {disease}")
            else:
                st.error(f"Disease: {disease}")

            st.info(f"Confidence: {conf:.2f}%")
            st.warning(f"Severity: {sev}")

            st.markdown("<div class='section-title'>Top Predictions</div>",unsafe_allow_html=True)

            top3 = pred[0].argsort()[-3:][::-1]

            for i in top3:

                name = class_indices[str(i)]

                p,d = name.split("___")
                d = d.replace("_"," ")

                c = pred[0][i]*100

                st.write(f"{p} — {d}: {c:.2f}%")

            report=create_report(plant,disease,conf,sev)

            with open(report,"rb") as f:

                st.download_button(
                    "Download Diagnosis Report",
                    data=f,
                    file_name="plant_disease_report.pdf"
                )


# ---------------- HISTORY ----------------

st.markdown("<br>",unsafe_allow_html=True)
st.subheader("Prediction History")

st.dataframe(load_history(),use_container_width=True)


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
