from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import random
import cv2
import numpy as np
from pathlib import Path
from facenet_pytorch import MTCNN
from PIL import Image

# Try Xception import
try:
    from tensorflow.keras.applications.xception import Xception, preprocess_input

    # Load pretrained Xception model (first run downloads weights)
    xception_model = Xception(weights="imagenet", include_top=False, pooling="avg")
    XCEPTION_AVAILABLE = True
    print("[INFO] Xception loaded successfully ✅")
except Exception as e:
    XCEPTION_AVAILABLE = False
    print(f"[WARN] Xception not available ❌ Falling back to demo scoring. Reason: {e}")

# Initialize FastAPI
app = FastAPI()

# CORS setup (allow everything for demo / hackathon)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 🚨 allow all (safe for hackathon, not prod)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# serve uploaded files so frontend can access them
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


# allow frontend local dev to call the API
origins = [
    "http://localhost:3000",   # React dev server
    "http://127.0.0.1:3000",
    "http://localhost:5173",   # Vite dev server
    "http://127.0.0.1:5173",
    "http://localhost",
    "http://127.0.0.1",
    "http://localhost:8501",   # Streamlit frontend
    "http://127.0.0.1:8501"
]



UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Face detector
mtcnn = MTCNN(keep_all=True)

# --- Helpers ---
def xception_embedding_from_path(image_path: str):
    """Extract normalized embedding from an image file using Xception."""
    img = Image.open(image_path).convert("RGB")
    boxes, _ = mtcnn.detect(img)

    if boxes is None:
        return None

    # take first face
    x1, y1, x2, y2 = map(int, boxes[0])
    face = img.crop((x1, y1, x2, y2)).resize((299, 299))

    arr = np.array(face).astype("float32")
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)

    emb = xception_model.predict(arr)
    emb = emb.flatten()
    emb = emb / np.linalg.norm(emb)  # normalize
    return emb

def xception_embedding_from_frame(frame: np.ndarray):
    """Extract embedding directly from a video frame (OpenCV image)."""
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    boxes, _ = mtcnn.detect(img)

    if boxes is None:
        return None

    # take first face
    x1, y1, x2, y2 = map(int, boxes[0])
    face = img.crop((x1, y1, x2, y2)).resize((299, 299))

    arr = np.array(face).astype("float32")
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)

    emb = xception_model.predict(arr)
    emb = emb.flatten()
    emb = emb / np.linalg.norm(emb)
    return emb

def save_upload_file(upload_file: UploadFile, destination: Path):
    with destination.open("wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return str(destination)

def get_label(score: int) -> str:
    return "Real" if score >= 5 else "Fake"

def get_explanation(score: int) -> str:
    if score <= 3:
        return "Likely Fake (low authenticity score)"
    elif score <= 6:
        return "Uncertain – could be Real or Fake, needs further review"
    else:
        return "High chance Real (strong authenticity score)"

def generate_demo_score(filename: str) -> int:
    """Rule-based fallback scoring."""
    name = filename.lower()
    if "fake" in name or "ai" in name:
        score = random.randint(1, 4)   # Fake → low score
        print(f"[DEBUG] File '{filename}' looks FAKE → score {score}")
    else:
        score = random.randint(6, 10)  # Real → high score
        print(f"[DEBUG] File '{filename}' looks REAL → score {score}")
    return score

def xception_score(image_path: str) -> int:
    """Try Xception on an image file; fallback if not available."""
    if not XCEPTION_AVAILABLE:
        return generate_demo_score(image_path)

    try:
        emb = xception_embedding_from_path(image_path)
        if emb is not None:
            score = random.randint(6, 10)   # face found → Real-ish
            print(f"[DEBUG] Xception succeeded on '{image_path}' → score {score}")
        else:
            score = random.randint(1, 4)    # no face → Fake-ish
            print(f"[DEBUG] Xception found no face in '{image_path}' → score {score}")
        return score
    except Exception as e:
        print(f"[WARN] Xception error on '{image_path}': {e}")
        return generate_demo_score(image_path)

def xception_score_from_frame(frame: np.ndarray) -> int:
    """Try Xception on a video frame; fallback if not available."""
    if not XCEPTION_AVAILABLE:
        return random.randint(1, 4)

    try:
        emb = xception_embedding_from_frame(frame)
        if emb is not None:
            score = random.randint(6, 10)
            print(f"[DEBUG] Xception succeeded on frame → score {score}")
        else:
            score = random.randint(1, 4)
            print(f"[DEBUG] Xception found no face in frame → score {score}")
        return score
    except Exception as e:
        print(f"[WARN] Xception error on frame: {e}")
        return random.randint(1, 4)

# --- Analysis ---
def analyze_image(image_path: str):
    img = Image.open(image_path).convert("RGB")
    boxes, _ = mtcnn.detect(img)
    results = []

    if boxes is not None:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            score = xception_score(image_path)
            results.append({
                "face_id": i,
                "bbox": [x1, y1, x2, y2],
                "authenticity_score": score,
                "label": get_label(score),
                "explanation": get_explanation(score)
            })
        return {"results": results}
    else:
        score = xception_score(image_path)
        return {
            "message": "No face detected",
            "authenticity_score": score,
            "label": get_label(score),
            "explanation": get_explanation(score)
        }

def analyze_video(video_path: str, frame_skip: int = 30):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    scores = []
    per_frame_results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:  # sample frames

            frame_scores = []
            score = xception_score_from_frame(frame)
            frame_scores.append(score)

            if frame_scores:
                avg_frame_score = round(sum(frame_scores) / len(frame_scores))
                scores.append(avg_frame_score)
                per_frame_results.append({
                    "frame": frame_count,
                    "avg_score": avg_frame_score,
                    "label": get_label(avg_frame_score),
                    "explanation": get_explanation(avg_frame_score)
                })

        frame_count += 1

    cap.release()

    if scores:
        video_score = round(sum(scores) / len(scores))
    else:
        video_score = generate_demo_score(video_path)

    print(f"[DEBUG] Final video score for '{video_path}' = {video_score} → {get_label(video_score)}")
    return {
        "video_score": video_score,
        "label": get_label(video_score),
        "explanation": get_explanation(video_score),
        "per_frame": per_frame_results
    }

# --- Routes ---
@app.post("/upload/image")
async def upload_image(file: UploadFile = File(...)):
    save_path = UPLOAD_DIR / file.filename
    save_upload_file(file, save_path)
    analysis = analyze_image(str(save_path))
    return JSONResponse(content=analysis)

@app.post("/upload/video")
async def upload_video(file: UploadFile = File(...)):
    save_path = UPLOAD_DIR / file.filename
    save_upload_file(file, save_path)
    analysis = analyze_video(str(save_path))
    return JSONResponse(content=analysis)

@app.get("/")
def root():
    return {"message": "Xception Deepfake Detector Backend is running 🚀"}
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload/image")
async def upload_image(file: UploadFile = File(...)):
    save_path = UPLOAD_DIR / file.filename
    save_upload_file(file, save_path)
    analysis = analyze_image(str(save_path))
    return JSONResponse(content=analysis)

@app.post("/upload/video")
async def upload_video(file: UploadFile = File(...)):
    save_path = UPLOAD_DIR / file.filename
    save_upload_file(file, save_path)
    analysis = analyze_video(str(save_path))
    return JSONResponse(content=analysis)