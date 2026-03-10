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

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR,"plant_disease_prediction_model.h5")
CLASS_FILE = os.path.join(BASE_DIR,"class_indices.json")
DB_FILE = os.path.join(BASE_DIR,"history.db")
EXAMPLE_FOLDER = os.path.join(BASE_DIR,"Examples")

@st.cache_resource
def load_disease_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_resource
def load_leaf_detector():
    return MobileNetV2(weights="imagenet")

disease_model = load_disease_model()
leaf_model = load_leaf_detector()

with open(CLASS_FILE) as f:
    class_indices = json.load(f)

conn = sqlite3.connect(DB_FILE,check_same_thread=False)
cur = conn.cursor()

cur.execute("CREATE TABLE IF NOT EXISTS history(time TEXT,plant TEXT,disease TEXT,confidence REAL)")
conn.commit()

# Disease descriptions
disease_info = {
"Tomato Yellow Leaf Curl Virus":
"A viral disease causing yellowing and curling of tomato leaves. It spreads through whiteflies and can significantly reduce crop yield.",

"Potato Late Blight":
"A fungal disease caused by Phytophthora infestans that creates dark lesions on leaves and spreads rapidly in humid environments.",

"Apple Scab":
"A fungal disease that causes dark spots on apple leaves and fruits. It thrives in cool and wet climates."
}

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

    pdf.ln(10)

    if disease in disease_info:
        pdf.multi_cell(0,10,f"About the Disease:\n{disease_info[disease]}")

    file="report.pdf"
    pdf.output(file)

    return file


def preprocess_image(image,size=(224,224)):
    image=image.resize(size)
    img=np.array(image)
    img=img.astype("float32")/255.0
    img=np.expand_dims(img,axis=0)
    return img


def predict_disease(image):
    img=preprocess_image(image)
    pred=disease_model.predict(img,verbose=0)
    index=np.argmax(pred)
    conf=float(np.max(pred))*100
    label=class_indices[str(index)]
    return label,conf,pred


def detect_leaf(image):

    img=image.resize((224,224))
    arr=np.array(img)
    arr=np.expand_dims(arr,axis=0)
    arr=preprocess_input(arr)

    preds=leaf_model.predict(arr,verbose=0)
    decoded=decode_predictions(preds,top=5)[0]

    plant_keywords=["leaf","tree","plant","flower","corn","grape","apple"]

    for _,label,_ in decoded:
        for word in plant_keywords:
            if word in label.lower():
                return True

    return False


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


st.set_page_config(page_title="Plant Disease Predictor",layout="wide")

st.markdown("""
<style>
.stApp {
background: linear-gradient(135deg,#e6f9f0,#f0f9ff);
}

h1 {
text-align:center;
font-weight:700;
}

.block-container {
padding-top:2rem;
}

@media (prefers-color-scheme: dark) {

.stApp {
background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
}

h1 {
color:white;
}

}
</style>
""",unsafe_allow_html=True)

st.title("🌿 Plant Disease Predictor")

if "image" not in st.session_state:
    st.session_state.image=None

input_choice=st.radio(
"Select Image Source",
["Use Example Image","Upload Leaf Image","Camera"]
)

if "last_input" not in st.session_state:
    st.session_state.last_input=input_choice

if st.session_state.last_input!=input_choice:
    st.session_state.image=None
    st.session_state.last_input=input_choice


# Example images
if input_choice=="Use Example Image":

    example_files=sorted(os.listdir(EXAMPLE_FOLDER))
    selected_example=st.selectbox("Select Example",example_files)

    if selected_example:
        example_path=os.path.join(EXAMPLE_FOLDER,selected_example)
        st.session_state.image=Image.open(example_path).convert("RGB")


# Upload image
elif input_choice=="Upload Leaf Image":

    uploaded=st.file_uploader("Browse Leaf Image",type=["jpg","jpeg","png"])

    if uploaded:
        st.session_state.image=Image.open(uploaded).convert("RGB")


# Camera
elif input_choice=="Camera":

    uploaded=st.camera_input("Take Leaf Photo")

    if uploaded:
        st.session_state.image=Image.open(uploaded).convert("RGB")


image=st.session_state.image

if image is not None:

    col1,col2=st.columns(2)

    with col1:
        st.image(image,width=620)

    with col2:

        if not detect_leaf(image):

            st.error("The uploaded image does not appear to contain a plant leaf.")

        else:

            with st.spinner("🔍 Analyzing leaf disease..."):

                label,conf,pred=predict_disease(image)

            plant,disease=label.split("___")
            disease=disease.replace("_"," ")

            sev=severity(image)

            save_history(plant,disease,conf)

            st.success(f"🌱 Plant : {plant}")
            st.error(f"🦠 Disease : {disease}")

            st.info(f"Confidence : {conf:.2f}%")
            st.warning(f"Severity Level : {sev}")

            st.subheader("About the Disease")

            if disease in disease_info:
                st.write(disease_info[disease])

            st.subheader("Download Report")

            report=create_report(plant,disease,conf,sev)

            with open(report,"rb") as file:

                st.download_button(
                    label="📄 Download Diagnosis Report",
                    data=file,
                    file_name="plant_disease_report.pdf",
                    mime="application/pdf"
                )

            st.subheader("Top Predictions")

            top3=pred[0].argsort()[-3:][::-1]

            for i in top3:

                name=class_indices[str(i)]
                p,d=name.split("___")
                d=d.replace("_"," ")

                c=pred[0][i]*100

                st.write(f"{p} — {d} : {c:.2f}%")

st.subheader("Prediction History")

hist=load_history()

st.dataframe(hist,use_container_width=True)

st.markdown("---")

st.markdown(
"""
<div style='text-align: center; font-size:16px; margin-top:20px;'>
Created by <b>Soham Mondal</b><br>
For any query contact <b>sohammondal29@gmail.com</b>
</div>
""",
unsafe_allow_html=True
)