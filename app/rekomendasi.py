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


async def get_rekomendasi_async(scores: dict, dass_dominant: str, ml_label: str, vent_text: str) -> str:
    """
    Fungsi ASYNC dengan Dual-Modality Prompting (DASS-21 + LSTM).
    """
    prompt = (
        f"Kamu adalah psikolog klinis yang empatik, profesional, dan realistis. "
        f"ATURAN MUTLAK: Hindari 'toxic positivity' (jangan gunakan kata seperti 'semua akan baik-baik saja', 'jangan bersedih', 'kamu pasti kuat'). "
        f"Gunakan pendekatan validasi emosi dan berikan solusi praktis.\n\n"
        f"Sistem kami menggunakan Skrining Validasi Silang (Dual-Modality) dengan hasil berikut:\n"
        f"1. Tes Kognitif Objektif (DASS-21): Dominan pada {dass_dominant} (Skor: Depresi {scores['depression']}, Kecemasan {scores['anxiety']}, Stres {scores['stress']}).\n"
        f"2. Analisis Emosi Teks (Machine Learning LSTM): Mendeteksi indikasi kuat adanya {ml_label}.\n"
        f"3. Curhatan Pasien: \"{vent_text}\"\n\n"
        f"TUGAS:\n"
        f"Berikan 4 poin rekomendasi singkat, menenangkan, dan actionable (bisa langsung diterapkan saat ini juga). "
        f"Sintesiskan rekomendasi berdasarkan temuan tes objektif ({dass_dominant}) DAN indikasi emosi dari teks ({ml_label}). "
        f"Fokus HANYA pada 4 poin rekomendasi tersebut tanpa pendahuluan atau penutup. Jangan ulangi menyebutkan skor mereka di jawabanmu."
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