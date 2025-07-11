import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from audio_processing import load_and_preprocess_audio

# Function to create the LSTM model
def create_lstm_model(n_classes=10):
    model = models.Sequential()
    model.add(layers.Input(shape=(None, 13)))
    model.add(layers.Bidirectional(layers.LSTM(64)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(n_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Main function to train the model
def main(data_folder):
    audio_files = []
    labels = []

    for label in range(10):
        class_folder = os.path.join(data_folder, f'd{label}')
        if os.path.isdir(class_folder):
            for filename in os.listdir(class_folder):
                if filename.endswith('.wav'):
                    audio_files.append(os.path.join(class_folder, filename))
                    labels.append(label)

    if len(labels) == 0:
        raise ValueError("No audio files found. Please check your dataset structure.")

    labels = np.eye(10)[labels]
    train_files, test_files, train_labels, test_labels = train_test_split(audio_files, labels, test_size=0.3, random_state=42)

    x_train = np.array([load_and_preprocess_audio(file)[1] for file in train_files])
    x_test = np.array([load_and_preprocess_audio(file)[1] for file in test_files])

    mfcc_mean = np.mean(x_train, axis=(0, 1))
    mfcc_std = np.std(x_train, axis=(0, 1))

    x_train = (x_train - mfcc_mean) / mfcc_std
    x_test = (x_test - mfcc_mean) / mfcc_std

    model = create_lstm_model()
    model.fit(x_train, train_labels, epochs=30, batch_size=32, validation_data=(x_test, test_labels))

    return model, mfcc_mean, mfcc_std
