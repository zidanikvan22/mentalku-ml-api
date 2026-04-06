import os
from google import genai
from dotenv import load_dotenv

# Load env variables
load_dotenv(".env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY tidak ditemukan di environment.")

# Inisialisasi Client di luar fungsi biar gak di-load berulang kali
client = genai.Client(api_key=GOOGLE_API_KEY)

async def get_rekomendasi_async(scores: dict, label_ml: str) -> str:
    """
    Fungsi ASYNC untuk meminta rekomendasi ke Gemini 3 Flash.
    Tidak akan nge-block server saat menunggu respons.
    """
    prompt = (
        f"Kamu adalah seorang profesional kesehatan mental (psikolog/psikiater). "
        f"Berdasarkan skrining DASS-21, pasien memiliki skor:\n"
        f"- Depresi: {scores['depression']}\n"
        f"- Kecemasan: {scores['anxiety']}\n"
        f"- Stres: {scores['stress']}\n"
        f"Dari cerita pasien, model AI mendeteksi kecenderungan dominan: {label_ml}.\n\n"
        f"Berikan rekomendasi psikologis dalam 4 poin utama. "
        f"Gunakan bahasa profesional yang padat, empati, sederhana, dan mudah dimengerti. "
        f"Fokus HANYA pada poin rekomendasi, JANGAN berikan kalimat pendahuluan atau penutup apapun."
    )

    try:
        # MAGIC HAPPENS HERE: Update target model ke gemini-3-flash
        response = await client.aio.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt
        )
        return response.text
    except Exception as e:
        print(f"Error dari Gemini: {e}")
        return "Maaf, kami kesulitan memproses rekomendasi saat ini. Silakan istirahat sejenak dan hubungi profesional jika butuh bantuan."