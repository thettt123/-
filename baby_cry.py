import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from scripts.data_processing import load_data
from scripts.crnn_model import create_crnn_model
from scripts.predict import predict_cry
from scripts.utils import ensure_dir_exists

def main():
    # 경로
    data_dir = 'data/'                  # 데이터셋 경로
    model_path = 'models/baby_cry_model_crnn.keras'  # 모델을 저장할 경로
    test_audio_dir = 'test_audio/'      # 테스트 파일 경로

    # 디렉토리 확인
    ensure_dir_exists('models/')

    # 데이터 불러오기
    print("Loading data...")
    X, y = load_data(data_dir)

    # 데이터셋 유효성 검사
    if len(X) == 0 or len(y) == 0:
        raise ValueError("No valid samples found in the dataset. Check the data directory.")

    # 패딩
    max_timesteps = max(x.shape[0] for x in X if len(x.shape) == 2)
    X_padded = np.array([
        np.pad(
            x,
            ((0, max(0, max_timesteps - x.shape[0])), (0, 0)),
            mode='constant'
        )
        for x in X
    ])
    X_padded = X_padded / np.max(X_padded)

    # 데이터셋 분할
    X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

    # NumPy 배열로 변환
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print(f"X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
    print(f"y_train shape: {y_train.shape}, dtype: {y_train.dtype}")

    # 모델 생성 및 학습
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(set(y))
    model = create_crnn_model(input_shape, num_classes)

    print("Training CRNN model...")
 
    model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32)

    # 모델 저장
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # 모델 불러오기 및 컴파일
    print("Loading and compiling the model...")
    loaded_model = load_model(model_path)
    loaded_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # 모델 평가
    loss, accuracy = loaded_model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

    # 테스트 파일 예측
    print(f"Predicting audio files in: {test_audio_dir}")
    for file_name in os.listdir(test_audio_dir):
        if file_name.endswith('.wav'):
            file_path = os.path.join(test_audio_dir, file_name)
            print(f"Processing: {file_name}")
            result = predict_cry(model_path, file_path, max_timesteps)
            print(f"The baby is likely: {result}")
        else:
            print(f"Skipping non-WAV file: {file_name}")

if __name__ == "__main__":
    main()
