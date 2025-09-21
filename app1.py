import streamlit as st
import requests

st.set_page_config(page_title="Deepfake Detector", layout="centered")

st.title("🎭 Deepfake Detector")
st.write("Upload an image or video to check if it's real or fake.")

# File upload
uploaded_file = st.file_uploader("Upload Image/Video", type=["jpg", "jpeg", "png", "mp4", "avi"])

if uploaded_file is not None:
    st.info("File uploaded successfully! ✅")

    # Button to send to backend
    if st.button("Detect Deepfake"):
        st.write("⏳ Sending to backend for detection...")

        # Call backend API (placeholder for now)
        files = {"file": uploaded_file.getvalue()}
        response = requests.post("http://127.0.0.1:5000/detect", files=files)

        if response.status_code == 200:
            result = response.json()
            st.success(f"Result: {result['result']} (Fake Score: {result['fake_score']:.2f})")
        else:
            st.error("Error connecting to backend")
