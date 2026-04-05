import re
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore


def clean_text_v1(text: str) -> str:
    """
    Zero-Destruction Preprocessing Protocol.
    IDENTIK 100% dengan fungsi clean_text di Google Colab.
    """
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URL
    text = re.sub(r'@\w+', '', text)  # Remove Mention
    text = re.sub(r'#\w+', '', text)  # Remove Hashtag
    text = re.sub(r'\d+', '', text)  # Remove Digit
    text = re.sub(r'[^\w\s]', '', text)  # Remove Special char
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize Whitespace
    return text


def predict_emotion(text: str, model, tokenizer, label_encoder, max_sequence_length: int = 97) -> str:
    """
    Inference Keras 3 Model.
    MAX_LEN = 97, padding='post', truncating='post'.
    """
    # Fallback kalau input kosong
    if not text or not text.strip():
        return "normal"

    # 1. Cleaning Teks sesuai Pipeline Training
    cleaned_text = clean_text_v1(text)

    # 2. Tokenization ke Sequence
    sequence = tokenizer.texts_to_sequences([cleaned_text])

    # 3. Padding (Wajib 'post' sesuai blueprint arsitektur)
    padded = pad_sequences(
        sequence,
        maxlen=max_sequence_length,
        padding='post',
        truncating='post'
    )

    # 4. Prediksi Keras 3
    probabilities = model.predict(padded, verbose=0)[0]
    predicted_index = np.argmax(probabilities)

    # 5. Decode Index ke Label (contoh: 'depression', 'stress')
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]

    return predicted_label