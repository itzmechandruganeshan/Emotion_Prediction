import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense
from keras.models import Sequential 
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

emotion_num = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Sad': 4, 'Surprise': 5, 'Neutral': 6}

st.title("Emotion Predictions Based on the Previous sequence of emotions")

col1 , col2 =st.columns([5,5])

def run_lstm():
    df = pd.read_csv("emotions_data.csv")
    emotion_num_inverse={}
    for key, value in emotion_num.items():
        emotion_num_inverse[value]=key
        
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

    # Label Encoding
    label_encoder = LabelEncoder()
    df['Dominant_Emotion'] = label_encoder.fit_transform(df['Dominant_Emotion'])
    
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
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test),verbose=0, callbacks=[early_stopping, reduce_lr])

    # Reshape X_test for prediction
    X_test_reshaped = X_test[-1].reshape(1, window_size, 1)
    
    # Predict using the trained model
    y_pred = model.predict(X_test_reshaped)
    
    # Reverse label encoding to get the original labels
    y_test_original = label_encoder.inverse_transform(y_test)
    y_pred_labels_original = label_encoder.inverse_transform(np.argmax(y_pred, axis=1))
    
    for i, j in enumerate(y_pred[0]):
        predicted_label = int(label_encoder.inverse_transform([i])[0])
        st.success(f"{emotion_num_inverse[predicted_label]} - {j*100} %")

    return "Successfully Predicted Using LSTM"

def run_random_forest():
    df = pd.read_csv("emotions_data.csv")
    emotion_num_inverse={}
    for key, value in emotion_num.items():
        emotion_num_inverse[value]=key
        
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
    result = list()
    for i in range(len(df),len(df)+1):
        # Predict the X_test data
        predicted_result = classifier.predict([[i]])
        result.append(emotion_num_inverse[int(predicted_result[0])])
        
    st.success(f"Predicted Next Emotion:{result}")   
    
    return "Successfully Predicted Using RandomForestClassifier"
    
with col1:
    st.subheader("Probability of the Emotions in Next frame")
    run_lstm()
with col2:
    st.subheader("Prediction Result as Positive or Negative")
    run_random_forest()