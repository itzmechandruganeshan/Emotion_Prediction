# 🎥 Emotional State Detection Through Video

This project implements an emotional state detection system that analyzes the emotions of a person from a video file. The system uses deep learning models to detect facial expressions from video frames and predict the corresponding emotional state, all while creating a seamless and interactive user experience. 🌟

## Project Structure

```bash
app/
│
├── .dockerignore                # Specifies files to be excluded from Docker builds
├── .gitattributes               # Git settings for handling line endings and other attributes
├── Dockerfile                   # Dockerfile for containerizing the application
├── main.py                      # Main application script that runs the emotion detection pipeline
├── requirements.txt             # Required Python packages for the project
├── sample.py                    # Example script for reference
│
├── extracted_data/
│   └── emotions_data.csv         # Stores the detected emotions from video processing
│
├── initial_code/
│   └── Emotion_detector.py       # Initial code or reference implementation for emotion detection
│
├── output/
│   └── output_video.mp4          # Output video with detected emotions overlaid on faces
│
├── pages/
│   └── Emotion_Prediction.py     # Additional pages for the Streamlit app (if needed)
│
└── utils/
    ├── haarcascade_frontalface_default.xml  # Haar Cascade file for face detection
    ├── Model.h5                            # Pre-trained emotion detection model
    └── temp_video.mp4                      # Temporary storage for uploaded video
```

## 1. Overview

📊 The application leverages a pre-trained Convolutional Neural Network (CNN) model to predict emotions based on facial expressions detected in a video. Using **Streamlit** for a user-friendly interface, and **OpenCV** to process video frames and detect faces, the application predicts emotions, stores the data in a CSV file, and overlays labels on the processed video. 

Additionally, the processed video can be previewed with real-time emotion annotations for each face in the video. This adds an engaging touch by showing emotional transitions as the video plays! 🎬😃

## 2. Features

✨ **Key Highlights:**
- 🎥 **Video Upload:** Users can upload an `.mp4` video for emotion detection.
- 👤 **Face Detection:** Uses OpenCV's Haar Cascade to detect faces in each frame.
- 🤖 **Emotion Prediction:** Predicts emotions using a pre-trained model (`Model.h5`) and labels the dominant emotion for each detected face.
- 📊 **CSV Output:** Extracted emotions are saved in a CSV file for further analysis.
- 📽️ **Video Output:** Generates an output video with detected emotions overlaid on each detected face.
- 🖥️ **Streamlit Interface:** Simple, interactive web interface for video upload, processing, and result preview.

## 3. Emotion Classes

The model is trained to detect the following emotions:
- 😡 Angry
- 😒 Disgust
- 😱 Fear
- 😊 Happy
- 😢 Sad
- 😲 Surprise
- 😐 Neutral

These emotions are visually displayed on the detected faces throughout the video, making it easier to observe how emotions change over time. ⏳

## 4. How It Works

1. 📥 **Model Loading:** The pre-trained CNN model (`Model.h5`) is loaded along with the Haar Cascade classifier for face detection.
2. 🎦 **Video Processing:** The user uploads a video file, and the application processes it frame by frame.
3. 🧠 **Face Detection:** Faces are detected in each frame using the Haar Cascade classifier.
4. 🤔 **Emotion Prediction:** For each detected face, the emotion is predicted by the CNN model, and the dominant emotion is labeled.
5. 🎬 **Output Generation:** The processed video with emotion labels is displayed to the user, and the emotion data is saved in `emotions_data.csv`. The video playback also reflects changes in emotional states throughout the video! 📊🎥

## 5. Installation

To run the project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/itzmechandruganeshan/Emotion_Prediction.git
    ```

2. Navigate to the project directory:
    ```bash
    cd app
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the Streamlit app:
    ```bash
    streamlit run main.py
    ```

## 6. Dependencies

The project requires the following Python packages:

- `opencv-python` 🖼️
- `tensorflow` 🤖
- `keras` 🔬
- `numpy` 🔢
- `pandas` 📊
- `streamlit` 🌐

These can be installed using the `requirements.txt` file.

## 7. Usage

1. 🚀 Launch the application using Streamlit.
2. 📤 Upload a video file in `.mp4` format.
3. 🕰️ Wait for the video processing to complete.
4. 🎥 The detected emotions will be overlaid on the faces in the video.
5. 💾 Download the generated CSV file with emotion data for further analysis.
