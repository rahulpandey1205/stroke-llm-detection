import joblib
import pandas as pd
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request

from backend.database import engine, SessionLocal, Base
from backend.models import PatientRecord
from inference.predict import predict_risk
from llm_engine.explain_prediction import explain_with_llm

import pdfplumber
import pytesseract
from PIL import Image
import io
import re

# --------------------------------
# Setup
# --------------------------------

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = FastAPI()

templates = Jinja2Templates(directory="frontend")

# Create database tables
Base.metadata.create_all(bind=engine)


# --------------------------------
# OCR Medical Data Extraction
# --------------------------------

def extract_medical_values(text):
    """Robust extraction of patient metrics from raw text/medical reports."""
    
    # Default values
    patient = {
        "gender": 1,
        "age": 50,
        "hypertension": 0,
        "heart_disease": 0,
        "ever_married": 1,
        "work_type": 2,
        "Residence_type": 1,
        "avg_glucose_level": 100,
        "bmi": 25,
        "smoking_status": 0
    }

    if not text or not isinstance(text, str):
        return patient

    text_lower = text.lower()

    # Improved patterns
    extraction_map = {
        "age": [r'age[:\s\-]*(\d+)', r'patient age[:\s]*(\d+)', r'(\d+)\s*years'],
        "bmi": [r'bmi[:\s]*(\d+\.?\d*)', r'body mass index[:\s]*(\d+\.?\d*)', r'index[:\s]*(\d+\.?\d*)'],
        "avg_glucose_level": [r'glucose[:\s]*(\d+\.?\d*)', r'blood glucose[:\s]*(\d+\.?\d*)', r'fasting glucose[:\s]*(\d+\.?\d*)', r'sugar[:\s]*(\d+\.?\d*)', r'hba1c[:\s]*(\d+\.?\d*)']
    }

    for key, patterns in extraction_map.items():
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                val = float(match.group(1))
                # Special handling for HbA1c to Glucose conversion if HbA1c is found
                if 'hba1c' in pattern:
                    # Formula: (HbA1c * 28.7) - 46.7 = estimated average glucose
                    patient["avg_glucose_level"] = (val * 28.7) - 46.7
                else:
                    patient[key] = val
                break

    # Advanced Extraction for Extra Medical Context
    extra_metrics = {}
    
    # Cholesterol & Lipids
    chol_match = re.search(r'cholesterol[:\s\-]*(\d+)', text_lower)
    if chol_match:
        extra_metrics["cholesterol"] = int(chol_match.group(1))
        if extra_metrics["cholesterol"] > 240:
            patient["heart_disease"] = 1 # Proxy high cholesterol to heart risk

    ldl_match = re.search(r'ldl[:\s\-]*(\d+)', text_lower)
    if ldl_match:
        extra_metrics["ldl"] = int(ldl_match.group(1))
        if extra_metrics["ldl"] > 160:
            patient["heart_disease"] = 1

    # Hypertension Detection
    bp_match = re.search(r'(\d{2,3})\/(\d{2,3})', text_lower)
    if bp_match:
        systolic, diastolic = map(int, bp_match.groups())
        if systolic > 140 or diastolic > 90:
            patient["hypertension"] = 1
            extra_metrics["blood_pressure"] = f"{systolic}/{diastolic}"

    # Smoking Status
    if any(term in text_lower for term in ["smoker", "smoking", "tobacco"]):
        patient["smoking_status"] = 2  # Active smoker
    elif "never smoked" in text_lower:
        patient["smoking_status"] = 0

    # Include extra metrics for the LLM (though model doesn't use them)
    patient["extra_metrics"] = extra_metrics

    return patient


# --------------------------------
# Homepage
# --------------------------------

@app.get("/", response_class=HTMLResponse)
def render_assessment(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# --------------------------------
# Manual Prediction
# --------------------------------

@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    gender: int = Form(...),
    age: float = Form(...),
    hypertension: int = Form(...),
    heart_disease: int = Form(...),
    ever_married: int = Form(...),
    work_type: int = Form(...),
    residence_type: int = Form(...),
    avg_glucose_level: float = Form(...),
    bmi: float = Form(...),
    smoking_status: int = Form(...)
):
    patient_data = {
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": residence_type,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "smoking_status": smoking_status
    }

    # Perform prediction
    prediction_result = predict_risk(patient_data)
    probability = prediction_result["probability"]
    risk = prediction_result["risk"]
    shap_contributors = prediction_result["shap_contributors"]

    # Generate clinical explanation
    clinical_explanation = explain_with_llm(patient_data, probability, risk, shap_contributors)

    # Store in database
    db = SessionLocal()
    record = PatientRecord(
        age=age,
        bmi=bmi,
        avg_glucose_level=avg_glucose_level,
        hypertension=hypertension,
        heart_disease=heart_disease,
        smoking_status=smoking_status,
        prediction=risk,
        probability=probability
    )
    db.add(record)
    db.commit()
    db.close()

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "risk": risk,
            "probability": probability,
            "explanation": clinical_explanation,
            "contributors": shap_contributors,
            "dashboard": {
                "Age": min(age / 80 * 100, 100),
                "Glucose": min(avg_glucose_level / 250 * 100, 100),
                "BMI": min(bmi / 40 * 100, 100),
                "Hypertension": 80 if hypertension == 1 else 10,
                "Heart Disease": 85 if heart_disease == 1 else 10,
                "Smoking": 70 if smoking_status == 2 else 15
            }
        }
    )


# --------------------------------
# Upload Medical Report
# --------------------------------

@app.post("/upload_report", response_class=HTMLResponse)
async def upload_report(request: Request, file: UploadFile = File(...)):
    extracted_text = ""
    print(f"DEBUG: Received file {file.filename}")
    
    try:
        if file.filename.endswith(".pdf"):
            print("DEBUG: Processing PDF with pdfplumber")
            with pdfplumber.open(file.file) as pdf:
                extracted_text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        elif file.filename.endswith((".png", ".jpg", ".jpeg")):
            print("DEBUG: Processing Image with Tesseract")
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            extracted_text = pytesseract.image_to_string(image)
        else:
            print("DEBUG: Unsupported file format")
    except Exception as e:
        print(f"DEBUG OCR Error: {e}")

    print(f"DEBUG: Extracted text length: {len(extracted_text)}")
    
    try:
        patient_data = extract_medical_values(extracted_text)
        print(f"DEBUG: Extracted patient data: {patient_data}")

        # Perform prediction
        print("DEBUG: Starting prediction")
        prediction_result = predict_risk(patient_data)
        probability = prediction_result["probability"]
        risk = prediction_result["risk"]
        shap_contributors = prediction_result["shap_contributors"]
        print(f"DEBUG: Prediction results: {risk}, {probability}%")

        # Generate clinical explanation
        print("DEBUG: Requesting clinical explanation")
        clinical_explanation = explain_with_llm(patient_data, probability, risk, shap_contributors)
        print("DEBUG: Clinical explanation received")

        # Store OCR data
        print("DEBUG: Storing in database")
        db = SessionLocal()
        record = PatientRecord(
            age=patient_data["age"],
            bmi=patient_data["bmi"],
            avg_glucose_level=patient_data["avg_glucose_level"],
            hypertension=patient_data["hypertension"],
            heart_disease=patient_data["heart_disease"],
            smoking_status=patient_data["smoking_status"],
            prediction=risk,
            probability=probability
        )
        db.add(record)
        db.commit()
        db.close()
        print("DEBUG: Database record created")

        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "risk": risk,
                "probability": probability,
                "explanation": clinical_explanation,
                "contributors": shap_contributors,
                "dashboard": {
                    "Age": min(patient_data["age"] / 80 * 100, 100),
                    "Glucose": min(patient_data["avg_glucose_level"] / 250 * 100, 100),
                    "BMI": min(patient_data["bmi"] / 40 * 100, 100),
                    "Hypertension": 80 if patient_data["hypertension"] == 1 else 10,
                    "Heart Disease": 85 if patient_data["heart_disease"] == 1 else 10,
                    "Smoking": 70 if patient_data["smoking_status"] == 2 else 15
                },
                "report_text": extracted_text[:700]
            }
        )
    except Exception as e:
        print(f"DEBUG PROCESSING ERROR: {e}")
        import traceback
        traceback.print_exc()
        return HTMLResponse(content=f"Internal Server Error: {str(e)}", status_code=500)

@app.get("/history", response_class=HTMLResponse)
def history(request: Request):
    db = SessionLocal()
    patients = db.query(PatientRecord).order_by(PatientRecord.id.desc()).all()
    db.close()
    return templates.TemplateResponse("history.html", {"request": request, "patients": patients})

@app.get("/guidelines", response_class=HTMLResponse)
def guidelines(request: Request):
    return templates.TemplateResponse("guidelines.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    db = SessionLocal()
    patients = db.query(PatientRecord).all()
    db.close()

    total = len(patients)
    high = len([p for p in patients if p.prediction == "High Stroke Risk"])
    moderate = len([p for p in patients if p.prediction == "Moderate Stroke Risk"])
    low = len([p for p in patients if p.prediction == "Low Stroke Risk"])

    # Calculate average metrics for charts
    avg_metrics = {
        "glucose": sum(p.avg_glucose_level for p in patients) / total if total > 0 else 0,
        "bmi": sum(p.bmi for p in patients) / total if total > 0 else 0,
        "age": sum(p.age for p in patients) / total if total > 0 else 0
    }

    # Data for Age distribution chart
    age_groups = {"0-20": 0, "21-40": 0, "41-60": 0, "61-80": 0, "80+": 0}
    for p in patients:
        if p.age <= 20: age_groups["0-20"] += 1
        elif p.age <= 40: age_groups["21-40"] += 1
        elif p.age <= 60: age_groups["41-60"] += 1
        elif p.age <= 80: age_groups["61-80"] += 1
        else: age_groups["80+"] += 1

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "patients": patients[::-1],  # Show newest first
            "total": total,
            "high": high,
            "moderate": moderate,
            "low": low,
            "avg_metrics": avg_metrics,
            "age_groups": age_groups
        }
    )