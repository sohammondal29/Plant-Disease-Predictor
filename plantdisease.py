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
import gc

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR,"plant_model.tflite")
CLASS_FILE = os.path.join(BASE_DIR,"class_indices.json")
DB_FILE = os.path.join(BASE_DIR,"history.db")
EXAMPLE_FOLDER = os.path.join(BASE_DIR,"Examples")

FILE_ID = "1Y8dRQTEE_16c8UEjRFdoppBMi-cZrzJ7"


# ---------------- DOWNLOAD MODEL ---------------- #

def download_model():

    if not os.path.exists(MODEL_PATH):

        with st.spinner("Downloading AI model (first run only)..."):
            gdown.download(id=FILE_ID, output=MODEL_PATH, quiet=False)

download_model()


# ---------------- LOAD TFLITE MODEL ---------------- #

@st.cache_resource
def load_disease_model():

    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return interpreter,input_details,output_details


interpreter,input_details,output_details = load_disease_model()


# ---------------- LOAD LEAF DETECTOR ---------------- #

@st.cache_resource
def load_leaf_detector():
    return MobileNetV2(weights="imagenet")


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

def preprocess_image(image,size=(224,224)):

    image=image.resize(size).convert("RGB")

    img=np.array(image,dtype=np.float32)

    img=img/255.0
    img=np.expand_dims(img,axis=0)

    return img


# ---------------- PREDICTION USING TFLITE ---------------- #

def predict_disease(image):

    img=preprocess_image(image)

    interpreter.set_tensor(input_details[0]['index'],img)

    interpreter.invoke()

    pred=interpreter.get_tensor(output_details[0]['index'])

    index=np.argmax(pred)
    conf=float(np.max(pred))*100

    label=class_indices[str(index)]

    return label,conf,pred


# ---------------- LEAF DETECTION ---------------- #

def detect_leaf(image):

    leaf_model = load_leaf_detector()

    img=image.resize((224,224)).convert("RGB")

    arr=np.array(img,dtype=np.float32)

    arr=np.expand_dims(arr,axis=0)

    arr=preprocess_input(arr)

    preds=leaf_model.predict(arr,verbose=0)

    decoded=decode_predictions(preds,top=5)[0]

    plant_keywords=["leaf","tree","plant","flower","corn","grape","apple"]

    for _,label,_ in decoded:
        for word in plant_keywords:
            if word in label.lower():

                gc.collect()

                return True

    gc.collect()

    return False


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


# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(page_title="Plant Disease Predictor",layout="wide")


# ---------------- DARK UI ---------------- #

st.markdown("""
<style>

.stApp {
background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
color:white;
}

h1,h2,h3 {
color:white;
}

label {
color:white !important;
}

</style>
""",unsafe_allow_html=True)


st.title("🌿 AI Plant Disease Detection System")


# ---------------- IMAGE INPUT ---------------- #

if "image" not in st.session_state:
    st.session_state.image=None


input_choice=st.radio(
"Select Image Source",
["Use Example Image","Upload Leaf Image","Camera"]
)


if input_choice=="Use Example Image":

    example_files=sorted(os.listdir(EXAMPLE_FOLDER))

    selected_example=st.selectbox("Select Example",example_files)

    if selected_example:

        example_path=os.path.join(EXAMPLE_FOLDER,selected_example)

        st.session_state.image=Image.open(example_path).convert("RGB")


elif input_choice=="Upload Leaf Image":

    uploaded=st.file_uploader("Browse Leaf Image",type=["jpg","jpeg","png"])

    if uploaded:
        st.session_state.image=Image.open(uploaded).convert("RGB")


elif input_choice=="Camera":

    uploaded=st.camera_input("Take Leaf Photo")

    if uploaded:
        st.session_state.image=Image.open(uploaded).convert("RGB")


image=st.session_state.image


# ---------------- PREDICTION ---------------- #

if image is not None:

    col1,col2=st.columns(2)

    with col1:
        st.image(image,width=500)

    with col2:

        if not detect_leaf(image):

            st.error("The uploaded image does not appear to contain a plant leaf.")

        else:

            with st.spinner("Analyzing leaf disease..."):

                label,conf,pred=predict_disease(image)

            plant,disease=label.split("___")
            disease=disease.replace("_"," ")

            sev=severity(image)

            save_history(plant,disease,conf)

            st.success(f"Plant: {plant}")
            st.error(f"Disease: {disease}")
            st.info(f"Confidence: {conf:.2f}%")
            st.warning(f"Severity Level: {sev}")


            st.subheader("Top Predictions")

            top3 = pred[0].argsort()[-3:][::-1]

            for i in top3:

                name = class_indices[str(i)]

                p,d = name.split("___")
                d = d.replace("_"," ")

                c = pred[0][i]*100

                st.write(f"{p} — {d} : {c:.2f}%")


            st.subheader("Download Report")

            report=create_report(plant,disease,conf,sev)

            with open(report,"rb") as file:

                st.download_button(
                    label="Download Diagnosis Report",
                    data=file,
                    file_name="plant_disease_report.pdf",
                    mime="application/pdf"
                )


# ---------------- HISTORY ---------------- #

st.subheader("Prediction History")

hist=load_history()

st.dataframe(hist,use_container_width=True)


st.markdown("---")

st.markdown(
"""
<div style='text-align:center;font-size:16px;margin-top:20px;'>
Created by <b>Soham Mondal</b><br>
For any query contact <b>sohammondal29@gmail.com</b>
</div>
""",
unsafe_allow_html=True
)
