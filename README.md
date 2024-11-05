## README

# Sentiment and Genre Analysis from Audio and Text Data

This project combines natural language processing (NLP) and deep learning techniques to perform **sentiment analysis** and **genre classification** on song data. It processes both **textual lyrics** and **audio signals** to classify songs by genre and detect sentiment, making it a versatile solution for analyzing musical content.

### Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Dataset](#dataset)
4. [Setup and Requirements](#setup-and-requirements)
5. [Project Structure](#project-structure)
6. [Model Training](#model-training)
7. [Evaluation and Metrics](#evaluation-and-metrics)
8. [Results](#results)
9. [Usage](#usage)
10. [Future Enhancements](#future-enhancements)

---

### Project Overview

The project builds and trains a deep learning model that:
- Analyzes song lyrics to classify sentiment.
- Processes audio features to categorize music by genre.
- Integrates the two modalities (text and audio) to improve the classification performance.

### Features

- **Text Processing for Lyrics:** Tokenizes and sequences lyrics for sentiment classification.
- **Audio Feature Extraction:** Uses Mel-frequency cepstral coefficients (MFCCs) from audio files for genre classification.
- **Multi-Modal Neural Network:** Combines audio and text data for improved classification.
- **Visualization and Evaluation:** Includes plotting functions for model accuracy, loss, and confusion matrix.

### Dataset

The project requires:
1. **Lyrics CSV File:** Contains processed lyrics for sentiment analysis.
2. **Audio Data JSON File:** Includes MFCCs and labels for genre classification.
3. **Audio Files:** Used for feature extraction in genre classification (optional but recommended).

### Setup and Requirements

Ensure you have the following libraries installed:

```bash
!pip install transformers
```

**Other Libraries Required:**
- `numpy`
- `pandas`
- `tensorflow`
- `matplotlib`
- `librosa`
- `scikit-learn`
- `seaborn`

Mount Google Drive to access the dataset:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Project Structure

Here's a breakdown of the primary files:

1. **`Create_Dataset.ipynb`:** Creates the dataset by reading audio files, extracting MFCCs, and saving them as JSON.
2. **`Training_the_model.ipynb`:** Defines, compiles, and trains the multi-modal deep learning model.
3. **`Sentiment_analysis.ipynb`:** Analyzes song lyrics using tokenization and padding, followed by training for sentiment classification.
4. **`final_script.ipynb`:** Combines all steps, from loading data to training the multi-modal model and evaluating it.

### Model Training

#### 1. Text Processing for Lyrics

- **Tokenizer Setup:** Tokenizes lyrics using Keras' `Tokenizer` with a vocabulary size of 5000.
- **Sequence Padding:** Pads sequences to a fixed length of 1000 for input uniformity.
- **Sentiment Classification Model:** Uses bidirectional LSTM layers and dense layers with dropout for sentiment classification.

#### 2. Audio Feature Extraction and Preprocessing

- **MFCC Extraction:** Extracts MFCCs for 30-second segments of audio data, setting up for genre classification.
- **Normalization:** Scales extracted MFCC features to ensure uniform input.

#### 3. Model Integration and Training

The multi-modal model uses:
- A pre-trained convolutional model (`model_crnn`) for audio features.
- An embedding layer for lyrics.
- Bidirectional LSTM layers for lyrics, dense layers, and a dropout layer to prevent overfitting.

The model is trained using sparse categorical cross-entropy and the Adam optimizer for accurate multi-class classification.

### Evaluation and Metrics

- **Accuracy and Loss Visualization:** Plots training and validation accuracy/loss over epochs.
- **Confusion Matrix:** Provides a confusion matrix for detailed evaluation of model performance.
- **Function `plot_confusion_matrix`:** Plots a detailed confusion matrix using seaborn's heatmap for visual interpretation.

### Results

The results section will provide insights into:
- Model performance on both sentiment and genre classification.
- Potential areas for improving classification accuracy.

### Usage

To test the model on an audio file, use the following function:

```python
def test_music(audio_file):
    # Run audio file through the model and output predictions
```

### Future Enhancements

- **Add more sentiment labels:** Enhance sentiment analysis to include nuanced categories (e.g., joy, anger).
- **Expand genre classification:** Include more genres by training on a larger, diverse dataset.
- **Real-time prediction:** Build a real-time prediction pipeline for streaming audio data.

