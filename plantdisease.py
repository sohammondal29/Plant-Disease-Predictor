import os
import json
import numpy as np
import streamlit as st
import pandas as pd
import sqlite3
from PIL import Image
from datetime import datetime
from fpdf import FPDF
import gdown
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR,"plant_model.tflite")
CLASS_FILE = os.path.join(BASE_DIR,"class_indices.json")
DB_FILE = os.path.join(BASE_DIR,"history.db")
EXAMPLE_FOLDER = os.path.join(BASE_DIR,"Examples")

FILE_ID = "YOUR_TFLITE_MODEL_DRIVE_ID"


# ---------------- DOWNLOAD MODEL ---------------- #

@st.cache_resource
def download_model():

    if not os.path.exists(MODEL_PATH):

        url = f"https://drive.google.com/uc?id={FILE_ID}"

        with st.spinner("Downloading AI model..."):
            gdown.download(url, MODEL_PATH, quiet=False)

download_model()


# ---------------- LOAD MODEL ---------------- #

@st.cache_resource
def load_model():

    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    return interpreter

interpreter = load_model()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# ---------------- LOAD CLASS INDICES ---------------- #

with open(CLASS_FILE) as f:
    class_indices = json.load(f)


# ---------------- DATABASE ---------------- #

conn = sqlite3.connect(DB_FILE,check_same_thread=False)
cur = conn.cursor()

cur.execute(
"CREATE TABLE IF NOT EXISTS history(time TEXT,plant TEXT,disease TEXT,confidence REAL)"
)

conn.commit()


# ---------------- PDF REPORT ---------------- #

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


# ---------------- IMAGE PROCESSING ---------------- #

def preprocess_image(image):

    image=image.resize((224,224)).convert("RGB")

    img=np.array(image,dtype=np.float32)

    img=img/255.0
    img=np.expand_dims(img,axis=0)

    return img


def predict_disease(image):

    img=preprocess_image(image)

    interpreter.set_tensor(input_details[0]['index'], img)

    interpreter.invoke()

    pred = interpreter.get_tensor(output_details[0]['index'])

    index=np.argmax(pred)
    conf=float(np.max(pred))*100

    label=class_indices[str(index)]

    return label,conf,pred


# ---------------- LEAF DETECTION ---------------- #

def detect_leaf(image):

    img=image.resize((224,224)).convert("RGB")

    img_np=np.array(img)

    R=img_np[:,:,0]
    G=img_np[:,:,1]
    B=img_np[:,:,2]

    green_pixels=(G>R) & (G>B) & (G>100)

    green_ratio=np.sum(green_pixels)/green_pixels.size

    return green_ratio > 0.20


# ---------------- DATABASE FUNCTIONS ---------------- #

def save_history(plant,disease,conf):

    current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cur.execute(
        "INSERT INTO history VALUES(?,?,?,?)",
        (current_time,plant,disease,conf)
    )

    conn.commit()


def load_history():

    return pd.read_sql_query(
        "SELECT * FROM history ORDER BY datetime(time) DESC",
        conn
    )


# ---------------- SEVERITY ---------------- #

def severity(image):

    img=np.array(image)

    gray=np.mean(img,axis=2)

    infected=np.sum(gray<120)
    total=gray.size

    ratio=infected/total

    if ratio<0.1:
        return "Low"
    elif ratio<0.3:
        return "Moderate"
    else:
        return "Severe"


# ---------------- UI ---------------- #

st.set_page_config(page_title="Plant Disease Predictor",layout="wide")

st.title("🌿 AI Plant Disease Detection System")


if "image" not in st.session_state:
    st.session_state.image=None


input_choice=st.radio(
"Select Image Source",
["Upload Leaf Image","Camera"]
)


if input_choice=="Upload Leaf Image":

    uploaded=st.file_uploader("Browse Leaf Image",type=["jpg","jpeg","png"])

    if uploaded:
        st.session_state.image=Image.open(uploaded).convert("RGB")


elif input_choice=="Camera":

    uploaded=st.camera_input("Take Leaf Photo")

    if uploaded:
        st.session_state.image=Image.open(uploaded).convert("RGB")


image=st.session_state.image


if image is not None:

    col1,col2=st.columns(2)

    with col1:
        st.image(image,width=400)

    with col2:

        if not detect_leaf(image):

            st.error("The uploaded image does not appear to contain a plant leaf.")

        else:

            label,conf,pred=predict_disease(image)

            plant,disease=label.split("___")
            disease=disease.replace("_"," ")

            sev=severity(image)

            save_history(plant,disease,conf)

            st.success(f"Plant: {plant}")
            st.error(f"Disease: {disease}")
            st.info(f"Confidence: {conf:.2f}%")
            st.warning(f"Severity Level: {sev}")


st.subheader("Prediction History")

hist=load_history()

st.dataframe(hist,use_container_width=True)
