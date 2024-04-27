# Importing the packages
import cv2
from keras.utils import img_to_array
import numpy as np
import pandas as pd
from keras.models import model_from_json
import streamlit as st
import os

st.set_page_config("Predicting Forthcoming Emotional State",layout='centered',initial_sidebar_state='collapsed')

emotion_num = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Sad': 4, 'Surprise': 5, 'Neutral': 6}
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

inverse_emotion = {}
for key, value in emotion_num.items():
    inverse_emotion[value] = key
st.header("Detecting Emotional state of the person through video")

def run_detection():
    st.cache_resource()
    def load_model():
        # Load the Emotion Detector json and weights
        json_file = open('Model_json.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        classifier = model_from_json(loaded_model_json)
        classifier.load_weights("Model_Weights.h5")
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        return classifier, face_cascade

    classifier, face_cascade = load_model()

    # Create dataframe
    emotion_df = pd.DataFrame(columns=emotion_labels)

    file = st.file_uploader(label="Upload your Video", type=['mp4'])

    if file is not None:
        # Save the uploaded file to a temporary location
        temp_file_path = f"./temp_video.{file.name.split('.')[-1]}"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file.read())

        save_interval = 30
        frame_count = 0

        cap = cv2.VideoCapture(temp_file_path)

        # Get the frames per second (fps) and frame size
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the codec and create a video writer object
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        out_path = r'C:/Users/pandi/Visual_Studio_code/Mini_Project/output_video.mp4'
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        while True:
            _, frame = cap.read()
            if frame is None:
                break

            labels = []
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1,5)
            emotions_list = []

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                if np.sum([roi_gray]) != 0:
                    roi = roi_gray.astype('float')
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    prediction = classifier.predict(roi)[0]

                    emotions_dict = {}

                    label = emotion_labels[prediction.argmax()]
                    emotions_dict['Dominant_Emotion'] = emotion_num[label]

                    emotions_list.append(emotions_dict)

                    label_position = (x-5, y-5)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            if emotions_list:
                emotion_df = pd.concat([emotion_df, pd.DataFrame(emotions_list)])

            frame_count += 1
            if frame_count % save_interval == 0:
                emotion_df.to_csv('emotions_data.csv', index=False)

            # Write the frame to the video file
            out.write(frame)

        # Release the VideoCapture, VideoWriter, and remove the temporary file
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        st.success("Video processing completed!")

        # Convert the video file to bytes and display it
        video_file = open(out_path, 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)
        
        if st.button("Done"):
            if os.path.isfile(out_path):
                os.remove(out_path)
                os.remove(temp_file_path)
        st.info("Note:Output video will be deleted after you click the Done button")

run_detection()