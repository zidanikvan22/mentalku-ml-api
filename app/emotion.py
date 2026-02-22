import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore


def predict_emotion(text: str, model, tokenizer, label_encoder, max_sequence_length: int = 777) -> str:
    """
    Menerima teks curhatan dan mengembalikan HANYA label string (misal: 'Stress', 'Depresi').
    """
    # Fallback kalau user gak nulis apa-apa (opsional di UI)
    if not text or not text.strip():
        return "Netral"

    # Preprocessing Text
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_sequence_length)

    # Inference (verbose=0 biar terminal lo gak penuh log sampah dari TF)
    probabilities = model.predict(padded, verbose=0)[0]
    predicted_index = np.argmax(probabilities)

    # Decode Index ke Label String
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]

    return predicted_label