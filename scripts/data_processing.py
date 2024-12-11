import os
import numpy as np
import librosa

def load_data(data_dir):
    labels = {
        'awake': 0, 'diaper': 1, 'hug': 2,
        'hungry': 3, 'sad': 4, 'sleepy': 5, 'uncomfortable': 6
    }
    X, y = [], []

    for label, idx in labels.items():
        folder = os.path.join(data_dir, label)
        if not os.path.exists(folder):
            print(f"Directory not found: {folder}")
            continue
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            try:
                audio, sr = librosa.load(file_path, sr=None)
                mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
                if mfccs.size > 0:
                    X.append(mfccs.T)
                    y.append(idx)
                else:
                    print(f"Empty MFCCs for file: {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    print(f"Loaded {len(X)} samples from {data_dir}.")
    return X, y
