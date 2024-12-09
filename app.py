import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os
import requests
from tqdm import tqdm
import json

# Load the pre-trained model
model_path = "best_audio_model.keras"
model = load_model(model_path)

# Set Kaggle credentials for GitHub usage
os.environ['KAGGLE_USERNAME'] = os.getenv("KAGGLE_USERNAME")
os.environ['KAGGLE_KEY'] = os.getenv("KAGGLE_KEY")

# Ensure the temp directory exists before saving the file
temp_dir = './temp'
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

# Function to download the dataset
def download_kaggle_dataset(dataset_name, save_path):
    os.makedirs(save_path, exist_ok=True)
    # Kaggle API download URL
    url = f"https://www.kaggle.com/api/v1/datasets/download/{dataset_name}"
    kaggle_api_token_path = os.path.expanduser("~/.kaggle/kaggle.json")
    with open(kaggle_api_token_path, "r") as f:
        kaggle_credentials = json.load(f)
    auth = (kaggle_credentials["username"], kaggle_credentials["key"])
    response = requests.get(url, stream=True, auth=auth)
    total_size = int(response.headers.get("content-length", 0))
    zip_path = os.path.join(save_path, f"{dataset_name.split('/')[-1]}.zip")
    with open(zip_path, "wb") as file, tqdm(
        desc=f"Downloading {dataset_name}",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            file.write(data)
            bar.update(len(data))
    return zip_path

# Function to preprocess audio
def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0).reshape(1, -1)

# App layout
st.title("Music Genre Recognition")
st.sidebar.header("Options")

# Dataset download
if st.sidebar.button("Download Dataset"):
    dataset_path = download_kaggle_dataset(
        "andradaolteanu/gtzan-dataset-music-genre-classification",
        "./data"
    )
    st.sidebar.success(f"Dataset downloaded to {dataset_path}")

# File upload
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
if uploaded_file:
    file_path = f"{temp_dir}/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    st.audio(file_path, format="audio/wav")
    
    # Preprocess and predict
    features = preprocess_audio(file_path)
    prediction = model.predict(features)
    genre = np.argmax(prediction)
    
    st.write(f"Predicted Genre: {genre}")
    st.bar_chart(prediction.flatten())

    # Visualize spectrogram
    y, sr = librosa.load(file_path, sr=22050)
    fig, ax = plt.subplots()
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel", ax=ax)
    ax.set_title("Mel Spectrogram")
    st.pyplot(fig)
