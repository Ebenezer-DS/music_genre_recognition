import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os
import random
import sounddevice as sd
import scipy.io.wavfile as wav

# Load the pre-trained model
model_path = "best_audio_model.keras"
model = load_model(model_path)

# Function to preprocess audio
def preprocess_audio(file_path):
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=22050)
        
        # Extract 13 MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Compute the mean across time for each MFCC
        mfcc = np.mean(mfcc.T, axis=0)
        
        # Reshape to match model input: (None, 13, 1)
        mfcc = mfcc.reshape(1, 13, 1)
        
        return mfcc
    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
        return None


# Record audio function
def record_audio(duration=5, samplerate=22050):
    try:
        st.write("Recording...")
        audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
        sd.wait()
        file_path = "temp_recording.wav"
        wav.write(file_path, samplerate, audio)
        st.write("Recording complete.")
        return file_path
    except Exception as e:
        st.error(f"Error during recording: {e}")
        return None

# Ensure directories exist
def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

# App layout
st.title("Music Genre Recognition")
st.sidebar.header("Options")

# Genre Labels Mapping
genre_labels = {
    0: "blues",
    1: "classical",
    2: "country",
    3: "disco",
    4: "hiphop",
    5: "jazz",
    6: "metal",
    7: "pop",
    8: "reggae",
    9: "rock"
}

# Random Test Button
if st.sidebar.button("Random Test"):
    dataset_path = './data/genres/rock'  # Modify with your dataset path
    if os.path.exists(dataset_path) and os.listdir(dataset_path):
        random_file = random.choice(os.listdir(dataset_path))
        file_path = os.path.join(dataset_path, random_file)
        st.write(f"Testing on {random_file}")

        # Preprocess and predict
        features = preprocess_audio(file_path)
        if features is not None:
            prediction = model.predict(features)
            genre_index = np.argmax(prediction)
            genre_name = genre_labels.get(genre_index, "Unknown Genre")
            st.write(f"Predicted Genre: {genre_name}")
            st.audio(file_path)

            # Bar chart with genre labels
            predictions_with_labels = {genre_labels[i]: prediction[0][i] for i in range(len(genre_labels))}
            st.bar_chart(predictions_with_labels)

        else:
            st.error("Unable to preprocess the audio file.")
    else:
        st.error("Dataset directory does not exist or is empty.")

# File upload
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
if uploaded_file:
    temp_dir = "./temp/"
    ensure_directory(temp_dir)
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    st.audio(file_path, format="audio/wav")

    # Preprocess and predict
    features = preprocess_audio(file_path)
    if features is not None:
        prediction = model.predict(features)
        genre_index = np.argmax(prediction)
        genre_name = genre_labels.get(genre_index, "Unknown Genre")

        st.write(f"Predicted Genre: {genre_name}")

        # Bar chart with genre labels
        predictions_with_labels = {genre_labels[i]: prediction[0][i] for i in range(len(genre_labels))}
        st.bar_chart(predictions_with_labels)

        # Visualize spectrogram
        try:
            y, sr = librosa.load(file_path, sr=22050)
            fig, ax = plt.subplots()
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)  # Corrected API usage
            S_dB = librosa.power_to_db(S, ref=np.max)
            img = librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel", ax=ax)
            ax.set_title(f"Mel Spectrogram - {genre_name}")
            plt.colorbar(img, ax=ax, format='%+2.0f dB')
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error during visualization: {e}")
    else:
        st.error("Unable to preprocess the uploaded audio file.")
