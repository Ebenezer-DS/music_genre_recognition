
# Music Genre Recognition

This project is focused on classifying music genres using audio features extracted from the GTZAN dataset. 
It utilizes machine learning techniques and a trained neural network model to predict the genre of given audio files.

## Features

- **Dynamic Dataset Download**: The GTZAN dataset is dynamically downloaded from Kaggle, ensuring accessibility and reproducibility.
- **Preprocessing**: Extracts audio features such as Mel-Frequency Cepstral Coefficients (MFCC), chroma, and spectral contrast.
- **Model Training**: Employs a Convolutional Neural Network (CNN) for audio classification.
- **Visualization**: Includes exploratory data analysis and model evaluation visuals.

## Prerequisites

Before running the notebook or deploying the app, ensure the following tools and libraries are installed:

- Python 3.7+
- TensorFlow
- Librosa
- Pandas
- NumPy
- Matplotlib
- Streamlit (if deploying as a web app)

## Files

- **music_genre_recognition.ipynb**: Main notebook containing the implementation.
- **best_audio_model.keras**: Pre-trained model file for inference.

## How to Use

1. Clone this repository:
   ```bash
   git clone ttps://github.com/Ebenezer-DS/music_genre_recognition.git
   cd music-genre-recognition
   ```

2. Download the GTZAN dataset from Kaggle dynamically by running the notebook.

3. Run the notebook to preprocess the data, train the model, or use the pre-trained model for predictions.

4. Optionally, deploy the app using Streamlit:
   ```bash
   streamlit run app.py
   ```

## Dataset

The GTZAN dataset is sourced from the Kaggle dataset [`andradaolteanu/gtzan-dataset-music-genre-classification`](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification). It contains 1000 audio tracks categorized into 10 genres.

## Model

The model `best_audio_model.keras` is trained using the following configuration:
- **Architecture**: CNN with dropout and batch normalization.
- **Input Features**: MFCCs, chroma, and spectral contrast.
- **Output**: 10 music genres.

## Results

The model achieves a high accuracy on the test set, demonstrating its effectiveness in recognizing music genres.

## Contributions

Contributions are welcome! Feel free to fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.
