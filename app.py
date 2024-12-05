import streamlit as st
import numpy as np
import librosa
import soundfile as sf
from tensorflow.keras.models import load_model
import tempfile

model = load_model('model.h5')

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000) 
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  
    mfccs = np.mean(mfccs.T, axis=0)  
    return mfccs

emotion_dict = {0: 'neutral', 1: 'calm', 2: 'happy', 3: 'sad', 4: 'angry', 5: 'fear', 6: 'disgust', 7: 'surprise'}

st.title('Speech Emotion Recognition')

st.write("Upload an audio file to see the detected emotion!")

audio_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if audio_file is not None:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with open(temp_file.name, 'wb') as f:
        f.write(audio_file.getvalue())

    mfccs = extract_features(temp_file.name)
    mfccs = np.reshape(mfccs, (1, 13, 1)) 
    emotion_prediction = model.predict(mfccs)
    predicted_emotion = emotion_dict[np.argmax(emotion_prediction)]
    confidence = np.max(emotion_prediction)

    st.write(f"Predicted Emotion: {predicted_emotion}")
    st.write(f"Confidence: {confidence * 100:.2f}%")
