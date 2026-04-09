# 🚀 Civic Issue Detection API

An AI-powered FastAPI backend that detects civic issues like **garbage, streetlight faults, potholes, and waterlogging** from images and generates structured reports.

---

## 📌 Features

* 🧠 Deep Learning model (MobileNetV2)
* 🗑️ Garbage detection
* 💡 Streetlight issue detection
* 🛣️ Pothole detection
* 🌧️ Waterlogging detection
* 📊 Severity scoring (low → critical)
* 📝 Automated issue report generation
* ⚡ FastAPI backend (easy integration with frontend)

---

## 🧠 Model Details

* Architecture: **MobileNetV2 (Transfer Learning)**
* Input Size: `224x224`
* Classes:

  * Garbage
  * Streetlight
  * Potholes
  * Waterlogging
  * Plain (No Issue)

---

## 📂 Project Structure

```
.
├── final.py                # FastAPI application
├── final_model.keras      # Trained DL model
├── class_labels.json      # Class mapping
├── requirements.txt       # Dependencies
├── runtime.txt            # Python version
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/princetiwarii/civic-api.git
cd civic-api
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the API

```bash
uvicorn final:app --reload
```

---

## 🌐 API Endpoints

### 🔹 Health Check

```
GET /
```

### 🔹 Predict Civic Issue

```
POST /predict
```

#### 📤 Input:

* Image file (form-data)

#### 📥 Output:

```json
{
  "department": "Sanitation",
  "category": "GARBAGE_BIN",
  "issue": "garbage",
  "severity": "critical",
  "confidence": 95.2,
  "title": "Overflowing Garbage",
  "description": "Garbage accumulation detected...",
  "ai_report": "Problem: ... Impact: ... Action: ..."
}
```

---

## 🧪 Testing

Open Swagger UI:

```
http://127.0.0.1:8000/docs
```

---

## 🚀 Deployment

This API can be deployed on:

* Render
* Railway
* AWS / GCP

---

## 📦 Tech Stack

* Python
* FastAPI
* TensorFlow / Keras
* NumPy
* Pillow

---

## 💡 Use Case

* Smart City Solutions
* Civic Complaint Automation
* Municipal Issue Monitoring
* AI-based Urban Management

---

## 👨‍💻 Author

**Prince Tiwari**
GitHub: https://github.com/princetiwarii

---

## ⭐ Future Improvements

* 📍 Location tagging
* 📊 Dashboard analytics
* 📱 Mobile app integration
* 🎥 Real-time video detection

---

## 📜 License

This project is for educational and research purposes.
