import os 
import librosa   # for audio processing
import numpy as np
from scipy.io import wavfile  # for audio processing
import warnings 
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import pandas as pd
warnings.filterwarnings("ignore")

labels = ["yes", "no", "up", "down", "left", "right", 
          "on", "off", "stop", "go"]
train_audio_path = 'data/train/audio'
all_wave = []
all_label = []
for label in labels:
    print(label)
    waves = [f for f in os.listdir(train_audio_path + '/' + label) if f.endswith('.wav')]
    for wav in waves:
        samples, sample_rate = librosa.load(train_audio_path + '/' + label + '/' + wav)
        samples = librosa.resample(samples, sample_rate, 8000)
        if (len(samples) == 8000):
            all_wave.append(samples)
            all_label.append(label)

le = LabelEncoder()
y = le.fit_transform(all_label)
classes = list(le.classes_)
y = np_utils.to_categorical(y, num_classes=len(labels))
# Reshape the 2D array to 3D array since the input to the 
# Conv1D must be a 3D array
all_wave = np.array(all_wave).reshape(-1, 8000, 1)


# Split into train and validation set
x_tr, x_val, y_tr, y_val = train_test_split(np.array(all_wave), np.array(y),
                                            stratify=y, test_size=0.2,
                                            random_size=777, shuffle=True)
