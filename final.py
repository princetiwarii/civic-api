# =====================
# IMPORTS
# =====================
from fastapi import FastAPI, UploadFile, File
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import json
import io
import google.generativeai as genai
import warnings
warnings.filterwarnings("ignore")
# =====================
# CONFIG
# =====================
# genai.configure(api_key="AIzaSyDEKSYVltmPNPzb5WT6XE6dOPnJ5o2g-xI")
import os
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
app = FastAPI(title="Civic Issue Detection API")

# =====================
# LOAD MODEL & LABELS
# =====================
model = load_model("final_model.keras")

with open("class_labels.json", "r") as f:
    class_indices = json.load(f)

class_labels = {v: k for k, v in class_indices.items()}

# =====================
# ISSUE MAPPING
# =====================
ISSUE_MAPPING = {
    "garbage": {
        "department": "Sanitation",
        "category": "GARBAGE_BIN",
        "title": "Overflowing Garbage",
        "description": "Garbage accumulation detected causing hygiene and health issues.",
        "severity": "high"
    },
    "streetlight": {
        "department": "Electricity Department",
        "category": "STREETLIGHT_FAULT",
        "title": "Streetlight Issue",
        "description": "Streetlight not working or damaged.",
        "severity": "medium"
    },
    "potholes": {
        "department": "Public Works (PWD)",
        "category": "POTHOLE_DEEP",
        "title": "Pothole Detected",
        "description": "Road damage detected which may cause accidents.",
        "severity": "high"
    },
    "waterlogging": {
        "department": "Public Works (PWD)",
        "category": "WATERLOGGING",
        "title": "Waterlogging Issue",
        "description": "Water accumulation due to poor drainage.",
        "severity": "high"
    },
    "plain": {
        "department": "General",
        "category": "NO_ISSUE",
        "title": "No Issue",
        "description": "No visible civic issue detected.",
        "severity": "none"
    }
}

# =====================
# SEVERITY LOGIC
# =====================
def adjust_severity(base_severity, confidence):
    if base_severity == "none":
        return "none"

    if confidence > 0.8:
        return "critical"
    elif confidence < 0.6:
        return "low"
    else:
        return base_severity

# =====================
# PREDICTION FUNCTION
# =====================
def predict(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(preds)
    confidence = float(np.max(preds))

    label = class_labels[predicted_class]
    return label, confidence

# =====================
# GEMINI REPORT
# =====================
def generate_report(title, description, confidence, severity):
    try:
        model_gemini = genai.GenerativeModel("gemini-2.5-flash")

        prompt = f"""
        Generate a short civic issue report.

        Issue: {title}
        Description: {description}
        Severity: {severity}
        Confidence: {confidence*100:.2f}%

        Keep it under 4 lines.

        Format:
        Problem:
        Impact:
        Action:
        """

        response = model_gemini.generate_content(prompt)
        return response.text.strip()

    except Exception:
        return "Report generation failed."

# =====================
# API ENDPOINT
# =====================
@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    label, confidence = predict(img)

    # Confidence safety
    if confidence < 0.5:
        label = "plain"

    issue_data = ISSUE_MAPPING.get(label.lower(), ISSUE_MAPPING["plain"])

    # Adjust severity
    severity = adjust_severity(issue_data["severity"], confidence)

    report = generate_report(
        issue_data["title"],
        issue_data["description"],
        confidence,
        severity
    )

    return {
        "department": issue_data["department"],
        "category": issue_data["category"],
        "issue": label,
        "severity": severity,
        "confidence": round(confidence * 100, 2),
        "title": issue_data["title"],
        "description": issue_data["description"],
        "ai_report": report
    }

# =====================
# ROOT CHECK
# =====================
@app.get("/")
def home():
    return {"message": "Civic Issue Detection API with Severity is running 🚀"}
