def process_dass21(answers: list[int]):
    """
    Menerima array 21 jawaban (0-3).
    Mengembalikan dictionary berisi skor dan level untuk masing-masing dimensi.
    """
    if len(answers) != 21:
        raise ValueError("Jumlah jawaban DASS-21 harus tepat 21.")

    # Slicing Array & Hitung Skor (DASS-21 harus dikali 2 untuk menyesuaikan dengan DASS-42)
    # Urutan UI kita: 1-7 (Stress), 8-14 (Kecemasan), 15-21 (Depresi)
    stress_score = sum(answers[0:7]) * 2
    anxiety_score = sum(answers[7:14]) * 2
    depresi_score = sum(answers[14:21]) * 2

    # Logic Penentuan Kategori
    dass21_scale = {
        "Depresi": [(0, 9, "Normal"), (10, 13, "Ringan"), (14, 20, "Sedang"), (21, 27, "Berat"), (28, 100, "Sangat berat")],
        "Kecemasan": [(0, 7, "Normal"), (8, 9, "Ringan"), (10, 14, "Sedang"), (15, 19, "Berat"), (20, 100, "Sangat berat")],
        "Stres": [(0, 14, "Normal"), (15, 18, "Ringan"), (19, 25, "Sedang"), (26, 33, "Berat"), (34, 100, "Sangat berat")]
    }

    def get_level(score, category):
        for min_val, max_val, label in dass21_scale[category]:
            if min_val <= score <= max_val:
                return label
        return "Tidak Diketahui"

    # Hitung Rata-rata & Status Final
    rata_rata = (depresi_score + anxiety_score + stress_score) / 3
    if rata_rata < 10:
        final_level = "Normal"
    elif rata_rata < 14:
        final_level = "Ringan"
    elif rata_rata < 20:
        final_level = "Sedang"
    elif rata_rata < 27:
        final_level = "Berat"
    else:
        final_level = "Sangat Berat"

    return {
        "scores": {
            "depression": depresi_score,
            "anxiety": anxiety_score,
            "stress": stress_score
        },
        "levels": {
            "depression": get_level(depresi_score, "Depresi"),
            "anxiety": get_level(anxiety_score, "Kecemasan"),
            "stress": get_level(stress_score, "Stres")
        },
        "average_score": round(rata_rata, 2),
        "final_level": final_level
    }