import librosa
import os

def augment_audio(audio, sr):
    augmented = []
    # 속도
    augmented.append(librosa.effects.time_stretch(audio, rate=0.9))
    augmented.append(librosa.effects.time_stretch(audio, rate=1.1))
    # 피치
    augmented.append(librosa.effects.pitch_shift(audio, sr, n_steps=2))
    augmented.append(librosa.effects.pitch_shift(audio, sr, n_steps=-2))
    return augmented

def ensure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
