import streamlit as st
import cv2
import numpy as np
from model import predict_emotion
from report_generator import generate_report

st.title("Facial Emotion Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)

    st.image(img, caption="Uploaded Image")

    result = predict_emotion(img)
    report = generate_report(result)

    st.subheader("Prediction Result")
    st.write("Emotion:", report["predicted_emotion"])
    st.write("Confidence:", report["confidence"])

    st.subheader("Full Report")
    st.json(report)