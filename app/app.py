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
MODEL_PATH = os.path.join(MODEL_DIR, "best_lstm_model_tuning.h5")
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
    """
    Endpoint tunggal untuk memproses semua logic evaluasi MentalKU.
    """
    # 0. Safety Check: Pastikan model udah masuk RAM
    if model is None or tokenizer is None or label_encoder is None:
        raise HTTPException(status_code=500, detail="Model ML belum siap. Cek log server Python lu.")

    # 1. Hitung DASS-21 (Synchronous - karena cuma hitungan matematika ringan)
    try:
        dass_result = process_dass21(payload.answers)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 2. Prediksi Klasifikasi LSTM (Synchronous)
    try:
        emotion_label = predict_emotion(
            text=payload.vent_text,
            model=model,
            tokenizer=tokenizer,
            label_encoder=label_encoder
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memproses AI Prediksi: {e}")

    # 3. Minta Rekomendasi Gemini (Asynchronous - MAGIC HAPPENS HERE)
    # Kita panggil pakai 'await' biar server gak nge-freeze pas nunggu balasan Google
    try:
        recommendation = await get_rekomendasi_async(
            scores=dass_result["scores"],
            label_ml=emotion_label
        )
    except Exception as e:
        print(f"⚠️ Warning: Gemini gagal ngasih respon -> {e}")
        recommendation = "Terjadi gangguan saat menarik rekomendasi AI. Mohon tetap jaga kesehatan mental Anda dan konsultasi ke profesional jika dirasa perlu."

    # 4. Susun Format JSON Akhir (Sesuai dengan ekspektasi Laravel Controller)
    return {
        "status": "success",
        "scores": dass_result["scores"],
        "levels": dass_result["levels"],
        "average_score": dass_result["average_score"],
        "final_level": dass_result["final_level"],
        "ml_analysis": {
            "label": emotion_label
        },
        "gemini": {
            "recommendation": recommendation
        }
    }