import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder


def run_emotion_detection():
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    classifier = load_model(r"D:\vscode\Pandas\Streamlit\Model.h5")

    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    save_interval = 30
    frame_count = 0

    time_limit = 150  

    # Get the start time
    start_time = time.time()

    # Emotions Dict
    emotion_num = {'Angry':0, 'Disgust':1, 'Fear':2, 'Happy':3, 'Sad':4, 'Surprise':5, 'Neutral':6}

    # Initialize DataFrame
    columns = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral', 'Dominant_Emotion']
    emotion_df = pd.DataFrame(columns=columns)

    cap = cv2.VideoCapture(0)

    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time > time_limit:
            print("Time limit reached. Exiting...")
            break

        _, frame = cap.read()
        labels = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.1)
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
                for label, prob in zip(emotion_labels, prediction):
                    emotions_dict[label] = prob

                label = emotion_labels[prediction.argmax()]
                emotions_dict['Dominant_Emotion'] = emotion_num[label]

                emotions_list.append(emotions_dict)

                label_position = (x-5, y-5)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        if emotions_list:
            emotion_df = pd.concat([emotion_df, pd.DataFrame(emotions_list)])

        cv2.imshow('Emotion Detector', frame)

        frame_count += 1
        if frame_count % save_interval == 0:
            emotion_df.to_csv('emotions_data.csv', index=False)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            emotion_df.to_csv('emotions_data.csv', index=False)
            break

    cap.release()
    cv2.destroyAllWindows()
    
df = pd.read_csv("emotions_data.csv")

def run_lstm():

    # Assuming you have 'Dominant_Emotion' as a string column
    df['Dominant_Emotion'] = df['Dominant_Emotion']

    # window size
    window_size = 50

    # Define a function to create input-output pairs
    def create_sequences(data, window_size):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i + window_size])
            y.append(data[i + window_size])
        return np.array(X), np.array(y)

    # Check unique values in 'Dominant_Emotion'
    unique_values = df['Dominant_Emotion'].unique()
    print("Unique Values in 'Dominant_Emotion':", unique_values)

    # Handle missing values if any
    df['Dominant_Emotion'].fillna(method='ffill', inplace=True)

    # Label Encoding
    label_encoder = LabelEncoder()
    df['Dominant_Emotion'] = label_encoder.fit_transform(df['Dominant_Emotion'])

    # Save the mapping of encoded labels to original labels
    label_mapping = dict(zip(label_encoder.transform(unique_values), unique_values))

    # Create input-output pairs
    X, y = create_sequences(df['Dominant_Emotion'], window_size)

    # Split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=64, activation='relu', input_shape=(window_size, 1)))
    model.add(Dense(units=np.max(y) + 1, activation='softmax'))  # Use the number of unique classes in your dataset

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Use Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=1, min_lr=0.0001)

    # Train the model with callbacks
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping, reduce_lr])

    # Reshape X_test for prediction
    X_test_reshaped = X_test.reshape(X_test.shape[0], window_size, 1)

    # Predict using the trained model
    y_pred = model.predict(X_test_reshaped)

    # Reverse label encoding to get the original labels
    y_test_original = label_encoder.inverse_transform(y_test)
    y_pred_labels_original = label_encoder.inverse_transform(np.argmax(y_pred, axis=1))

    result = classification_report(y_test_original, y_pred_labels_original)

    return result

def run_random_forest():
    # Emotion_dict
    emotion_num = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Sad': 4, 'Surprise': 5, 'Neutral': 6}

    # Frames
    df["frames"] = [i + 1 for i in range(0, len(df))]

    # Independent and Dependent Features
    X = df['frames'].values.reshape(-1,1)
    y = df['Dominant_Emotion'].values

    # Train and Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # RandomForest Classifier
    classifier = RandomForestClassifier(n_estimators=100)

    # Fit the data
    classifier.fit(X_train, y_train)

    # Predict the X_test data
    y_pred = classifier.predict(X_test)

    # Evaluate the accuracy
    result = classification_report(y_test, y_pred,zero_division=1)
    
    return result
	
if __name__ == "__main__":
    #print("Running Emotion Detection...")
    #run_emotion_detection()

    print("Running LSTM...")
    result1 = run_lstm()
    print(result1)

    print("Running Random Forest...")
    result2 = run_random_forest()
    print(result2)