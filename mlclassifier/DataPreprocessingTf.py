import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

import tensorflow as tf 
#from tensorflow.keras import layers
#from tensorflow.keras import models
from IPython import display

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)


DATASET_PATH = pathlib.Path.cwd()/'Data'/'genres_original'
print(DATASET_PATH)

genres = np.array(tf.io.gfile.listdir(str(DATASET_PATH)))
genres = genres[(genres != 'README.md') & (genres != '.DS_Store')]
print('Genres:', genres)

import librosa

# Load the audio file
file_path = DATASET_PATH/'blues/blues.00000.wav'
audio, sampling_rate = librosa.load(file_path, sr=None)  # sr=None to use the native sampling rate
secondsLength=3
print(f"Sampling Rate: {sampling_rate} Hz")


train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=DATASET_PATH,
    batch_size=64,
    validation_split=0.2,
    label_mode="categorical",
    seed=0,
    output_sequence_length=sampling_rate*secondsLength,  # 3 seconds at 22,050 Hz
    subset='both',
    verbose=True
)

label_names = np.array(train_ds.class_names)
print()
print("label names:", label_names)

print(train_ds.element_spec)

#This dataset only contains single channel audio, so use the tf.squeeze function to drop the extra axis:
#ffmpeg -i blues.00000.wav   
# Input #0, wav, from 'blues.00000.wav':
# Duration: 00:00:30.01, bitrate: 352 kb/s
# Stream #0:0: Audio: pcm_s16le ([1][0][0][0] / 0x0001), 22050 Hz, mono, s16, 352 kb/s
def squeeze(audio, labels):
  audio = tf.squeeze(audio, axis=-1)
  return audio, labels

train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)

test_ds = val_ds.shard(num_shards=2, index=0)
val_ds = val_ds.shard(num_shards=2, index=1)

print(label_names[[1,1,3,0]])

for example_audio, example_labels in train_ds.take(1):  
  print(example_audio.shape)
  print(example_labels.shape)

plt.figure(figsize=(16, 10))
rows = 3
cols = 3
n = rows * cols
for i in range(n):
  plt.subplot(rows, cols, i+1)
  audio_signal = example_audio[i]
  plt.plot(audio_signal)
  index = tf.argmax(example_labels[i], axis=0).numpy()
  plt.title(label_names[index])
  plt.yticks(np.arange(-1.2, 1.2, 0.2))
  plt.ylim([-1.1, 1.1])

#plt.show()

def get_spectrogram(waveform):
  # Convert the waveform to a spectrogram via a STFT.
  #spectrogram = tf.signal.stft(
  #    waveform, frame_length=255, frame_step=128)
  spectrogram = tf.signal.stft(
      waveform, frame_length=2048, frame_step=512, fft_length=2048)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

for i in range(3):
  index = tf.argmax(example_labels[i], axis=0).numpy()
  label = label_names[index]
  waveform = example_audio[i]
  spectrogram = get_spectrogram(waveform)

  print('Label:', label)
  print('Waveform shape:', waveform.shape)
  print('Spectrogram shape:', spectrogram.shape)
  print('Audio playback')
  #display.display(display.Audio(waveform, rate=sampling_rate))

def plot_spectrogram(spectrogram, ax):
  if len(spectrogram.shape) > 2:
    assert len(spectrogram.shape) == 3
    spectrogram = np.squeeze(spectrogram, axis=-1)
  # Convert the frequencies to log scale and transpose, so that the time is
  # represented on the x-axis (columns).
  # Add an epsilon to avoid taking a log of zero.
  log_spec = np.log(spectrogram.T + np.finfo(float).eps)
  height = log_spec.shape[0]
  width = log_spec.shape[1]
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)

fig, axes = plt.subplots(2, figsize=(12, 8))
timescale = np.arange(waveform.shape[0])
axes[0].plot(timescale, waveform.numpy())
axes[0].set_title('Waveform')
axes[0].set_xlim([0, sampling_rate])

plot_spectrogram(spectrogram.numpy(), axes[1])
axes[1].set_title('Spectrogram')
plt.suptitle(label.title())
#plt.show()

def make_spec_ds(ds):
  return ds.map(
      map_func=lambda audio,label: (get_spectrogram(audio), label),
      num_parallel_calls=tf.data.AUTOTUNE)

train_spectrogram_ds = make_spec_ds(train_ds)
val_spectrogram_ds = make_spec_ds(val_ds)
test_spectrogram_ds = make_spec_ds(test_ds)

for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
  break

rows = 3
cols = 3
n = rows*cols
fig, axes = plt.subplots(rows, cols, figsize=(16, 9))

for i in range(n):
    r = i // cols
    c = i % cols
    ax = axes[r][c]
    plot_spectrogram(example_spectrograms[i].numpy(), ax)

    index = tf.argmax(example_spect_labels[i], axis=0).numpy()
    ax.set_title(label_names[index])

plt.show()