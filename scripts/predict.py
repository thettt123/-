from tensorflow.keras.models import load_model
import librosa
import numpy as np

def predict_cry(model_path, audio_path, max_timesteps):
    model = load_model(model_path)

    # 오디오 로드 및 처리
    audio, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfccs_padded = np.pad(mfccs.T, ((0, max_timesteps - mfccs.shape[1]), (0, 0)), mode='constant')
    mfccs_padded = mfccs_padded / np.max(mfccs_padded)  # 정규화
    mfccs_padded = mfccs_padded[np.newaxis, ...]  # 배치 차원 추가

    # 예측
    prediction = model.predict(mfccs_padded)
    labels = ['awake', 'diaper', 'hug', 'hungry', 'sad', 'sleepy', 'uncomfortable']
    return labels[np.argmax(prediction)]
