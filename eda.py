import os 
import librosa   # for audio processing
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile  # for audio processing
import warnings 
warnings.filterwarnings("ignore")


train_auto_path = 'data/train/audio/'
samples, sample_rate = librosa.load(train_auto_path + 'yes/0a7c2a8d_nohash_0.wav', sr=16000)
fig = plt.figure(figsize=(14, 8))
ax1 = fig.add_subplot(211)
ax1.set_title('Raw wave of ' + 'data/train/audio/yes/0a7c2a8d_nohash_0.wav')
ax1.set_xlabel('Time')
ax1.set_ylabel('Amplitude')
ax1.plot(np.linspace(0, sample_rate/len(samples), sample_rate), samples)
plt.show()

"""
Resample the data
# The sampling rate of the signal is 16000 Hz. It would be better
# to re-sample it to 8000 Hz since most of the speech-related 
# frequencies are present at 8000 Hz
"""
# samples = librosa.resample(samples, sample_rate, 8000)


"""
Understand the number of recodings of each voice command
"""
labels = os.listdir(train_auto_path)
# find count of each label and plot bar graph
no_of_recordings = []
for label in labels:
    waves = [f for f in os.listdir(train_auto_path + '/' + label) if f.endswith('.wav')]
    no_of_recordings.append(len(waves))

# plot
plt.figure(figsize=(30,5))
index = np.arange(len(labels))
plt.bar(index, no_of_recordings)
plt.xlabel('Commands', fontsize=12)
plt.ylabel('No of recordings', fontsize=12)
plt.xticks(index, labels, fontsize=15, rotation=60)
plt.title('No. of recordings for each command')
plt.show()

print(labels)


"""
# Make a look at the distribution of the duration of recordings

duration_of_recordings = []
for label in labels:
    waves = [f for f in os.listdir(train_auto_path + '/' + label) if f.endswith('.wav')]
    for wav in waves:
        sample_rate, samples = wavfile.read(train_auto_path + '/' + label + '/' + wav)
        duration_of_recordings.append(float(len(samples)/sample_rate))
    print('ok')
plt.hist(np.array(duration_of_recordings))
plt.show()
"""
