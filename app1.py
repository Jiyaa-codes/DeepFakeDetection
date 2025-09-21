import streamlit as st
import requests
import io

st.set_page_config(page_title="Deepfake Detector", layout="centered")
st.title("🎭 Deepfake Detector")
st.write("Upload an image or video to check if it's real or fake.")




# File upload
uploaded_file = st.file_uploader("Upload Image/Video", type=["jpg", "jpeg", "png", "mp4", "avi"])

if uploaded_file is not None:
    st.info("File uploaded successfully! ✅")

    if st.button("Detect Deepfake"):
        st.write("⏳ Sending to backend for detection...")

    # Backend settings
    BACKEND_IP = "172.18.237.50"    # <-- replace with backend device IP
    BACKEND_PORT = 8000
    BASE_URL = f"http://{BACKEND_IP}:{BACKEND_PORT}"


        # Choose endpoint based on file type
    if uploaded_file.type.startswith("image"):
            url = f"{BASE_URL}/upload/image"
    else:
            url = f"{BASE_URL}/upload/video"

        # Prepare file for POST request
    files = {"file": (uploaded_file.name, io.BytesIO(uploaded_file.read()))}

    try:
            response = requests.post(url, files=files)

            if response.status_code == 200:
                result = response.json()
                st.success(f"✅ Result: {result}")
            else:
                st.error(f"❌ Backend error: {response.status_code}")
    except requests.exceptions.RequestException as e:
            st.error(f"❌ Connection error: {e}")

            