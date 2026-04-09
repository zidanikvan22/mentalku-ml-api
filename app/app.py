from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model # type: ignore
import pickle
import os
from dotenv import load_dotenv

from app.dass21 import process_dass21
from app.emotion import predict_emotion
from app.rekomendasi import get_rekomendasi_async

# Load env variables
load_dotenv(".env")

MODEL_DIR = os.getenv("MODEL_DIR", "app/models")
MODEL_PATH = os.path.join(MODEL_DIR, "mentalku_best_model.keras")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer_lstm.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

app = FastAPI(
    title="MentalKU Mega-API",
    description="API Terpusat untuk Evaluasi DASS-21, Klasifikasi Teks LSTM, dan Rekomendasi Gemini",
    version="2.0.0"
)

# Global Variables buat nampung model ML di memori
model = None
tokenizer = None
label_encoder = None

@app.on_event("startup")
async def load_ml_components():
    """
    Fungsi ini otomatis jalan sekali pas server nyala.
    Nge-load model berat ke RAM biar pas user request, API langsung gas gak perlu load ulang.
    """
    global model, tokenizer, label_encoder
    try:
        model = load_model(MODEL_PATH, compile=False)
        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)
        with open(LABEL_ENCODER_PATH, "rb") as f:
            label_encoder = pickle.load(f)
        print("✅ SUCCESS: Model LSTM, Tokenizer, dan Encoder berhasil dimuat ke RAM!")
    except Exception as e:
        print(f"❌ FATAL ERROR saat memuat model: {e}")

# --- PYDANTIC SCHEMA (JSON Contract dari Laravel) ---
class EvaluationRequest(BaseModel):
    answers: list[int] = Field(..., min_length=21, max_length=21, description="Array 21 jawaban DASS-21 (0-3)")
    vent_text: str = Field(..., description="Teks curhatan user")

@app.get("/", tags=["General"])
def read_root():
    return {"message": "MentalKU Mega-API is running smoothly! 🚀"}

# --- THE MEGA ENDPOINT ---
@app.post("/api/v1/evaluate", tags=["Core Evaluation"])
async def evaluate_mental_health(payload: EvaluationRequest):
    # 0. Safety Check
    if model is None or tokenizer is None or label_encoder is None:
        raise HTTPException(status_code=500, detail="Model ML belum siap.")

    # 1. Hitung DASS-21
    try:
        dass_result = process_dass21(payload.answers)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 2. Prediksi ML & TRANSLASI BAHASA (Anti-Retrain System)
    try:
        emotion_label_eng = predict_emotion(
            text=payload.vent_text, model=model, tokenizer=tokenizer, label_encoder=label_encoder
        )
        # Translator Cepat
        translator = {"depression": "Depresi", "anxiety": "Kecemasan", "stress": "Stres", "normal": "Normal"}
        ml_label_indo = translator.get(emotion_label_eng.lower(), "Normal")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memproses AI Prediksi: {e}")

    # 3. IDENTIFIKASI DOMINASI DASS-21 (Tanpa mengorbankan LSTM)
    severity_map = {"Normal": 0, "Ringan": 1, "Sedang": 2, "Parah": 3, "Sangat Parah": 4}
    dass_levels = dass_result["levels"]
    scores_index = {
            "Depresi": severity_map.get(dass_levels["depression"], 0),
            "Kecemasan": severity_map.get(dass_levels["anxiety"], 0),
            "Stres": severity_map.get(dass_levels["stress"], 0)
    }

    max_index = max(scores_index.values())
    if max_index == 0:
            dass_dominant = "Normal"
    else:
        # Ambil semua yang tertinggi (misal bisa aja Depresi & Stres sama-sama parah)
        highest_cats = [k for k, v in scores_index.items() if v == max_index]
        dass_dominant = " dan ".join(highest_cats)  # Output: "Stres" atau "Depresi dan Kecemasan"

    # 4. Minta Rekomendasi Gemini (Kirim DASS dan LSTM sebagai 2 entitas sejajar)
    try:
        recommendation = await get_rekomendasi_async(
            scores=dass_result["scores"],
            dass_dominant=dass_dominant,  # Kirim temuan utama tes objektif
            ml_label=ml_label_indo,  # Kirim temuan utama teks subjektif (LSTM)
            vent_text=payload.vent_text
        )
    except Exception as e:
        print(f"⚠️ Warning: Gemini gagal ngasih respon -> {e}")
        recommendation = "Terjadi gangguan saat menarik rekomendasi AI. Mohon tetap jaga kesehatan mental Anda dan konsultasi ke profesional."

    # 5. Susun Format JSON Akhir (Tambahin final_label)
    return {
        "status": "success",
        "scores": dass_result["scores"],
        "levels": dass_result["levels"],
        "average_score": dass_result["average_score"],
        "final_level": dass_result["final_level"],
        "dass_dominant": dass_dominant,
        "ml_analysis": {
            "label": ml_label_indo
        },
        "gemini": {
            "recommendation": recommendation
        }
    }