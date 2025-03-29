import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import librosa

import AudioProcessor

DATASET_PATH = pathlib.Path.cwd()/'Data'/'genres_original'
#DATASET_PATH = pathlib.Path.cwd()/'mlclassifier'/'Data'/'genres_original'

def remove_ds_store_files(path):
    for root, _, files in os.walk(path):
        for file in files:
            if file == ".DS_Store":
                try:
                    os.remove(os.path.join(root, file))
                    print(f"🧹 Removed .DS_Store from {root}")
                except Exception as e:
                    print(f"Failed to remove .DS_Store: {e}")

def squeeze(audio, labels):
    audio = tf.squeeze(audio, axis=-1)
    return audio, labels

def PlotWaveform(waveform,fs):
    plt.figure(figsize=(12, 6))
    t = np.arange(len(waveform)) / fs
    plt.plot(t,waveform)
    plt.yticks(np.arange(-1.2, 1.2, 0.2))
    plt.ylim([-1.1, 1.1])
    plt.show()

def LoadAll():
    remove_ds_store_files(DATASET_PATH)
    genres = np.array(tf.io.gfile.listdir(str(DATASET_PATH)))
    genres = genres[(genres != 'README.md') & (genres != '.DS_Store')]
    print('Genres:', genres)

    tfAll = tf.keras.utils.audio_dataset_from_directory(
        directory=DATASET_PATH,
        batch_size=64,
        shuffle=False  # Ensures files are loaded in directory order
    )
    return tfAll

def Tensorflow():
    print(DATASET_PATH)
    genres = np.array(tf.io.gfile.listdir(str(DATASET_PATH)))
    genres = genres[(genres != 'README.md') & (genres != '.DS_Store')]
    print('Genres:', genres)

    train_ds = tf.keras.utils.audio_dataset_from_directory(
        directory=DATASET_PATH,
        batch_size=64,
        shuffle=False  # Ensures files are loaded in directory order
    )

    label_names = np.array(train_ds.class_names)
    print("label names:", label_names)
    print(train_ds.element_spec)

    train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)

    for example_audio, example_labels in train_ds.take(1):  
         print(example_audio.shape)
         print(example_labels.shape)

    audio_signal = example_audio[0]
    fs=22050
    #PlotWaveform(audio_signal,fs)
    return np.array(audio_signal)

# def CoreGetSpecrogramTf(waveform, sample_rate=22050):
#     n_fft = 2048
#     hop_length = 512
#     pad = n_fft // 2
#     waveform = tf.pad(waveform, [[pad, pad]], mode='CONSTANT')

#     # Perform STFT with matching params
#     spectrogram = tf.signal.stft(
#         waveform,
#         frame_length=n_fft,
#         frame_step=hop_length,
#         fft_length=n_fft,
#         window_fn=tf.signal.hann_window,
#         pad_end=False  # we already padded manually
#     )

#     return spectrogram

def GetSpectrogramTf(waveform, sample_rate=22050):
    coreSpectrogram = AudioProcessor.CoreGetSpecrogramTf(waveform, sample_rate)
    stft = np.abs(coreSpectrogram.numpy().T)
    return librosa.amplitude_to_db(stft, ref=np.max)

def get_spectrogram(waveform):
  # Convert the waveform to a spectrogram via a STFT.
    spectrogram = AudioProcessor.CoreGetSpecrogramTf(waveform)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

def make_spec_ds(ds):
  return ds.map(
      map_func=lambda audio,label: (get_spectrogram(audio), label),
      num_parallel_calls=tf.data.AUTOTUNE)

def Librosa():
    file_path = DATASET_PATH/'blues/blues.00000.wav'
    audio_lib, sampling_rate = librosa.load(file_path, sr=None)  # sr=None to use the native sampling rate
    secondsLength=30
    print(f"Sampling Rate: {sampling_rate} Hz")
    return audio_lib
    #AudioProcessor.PlotWaveform(audio_lib,sampling_rate)

def GetSpectrogramLib(waveform,n_fft = 2048,hop_length = 512):
    stft = np.abs(librosa.stft(y=waveform, n_fft=n_fft, hop_length=hop_length))
    stft = librosa.amplitude_to_db(stft, ref=np.max) # ref - set max to 0 dB
    return stft

def is_valid_wav(filepath):
    try:
        audio_binary = tf.io.read_file(filepath)
        tf.audio.decode_wav(audio_binary)
        return True
    except Exception as e:
        print(f"❌ Corrupted or invalid: {filepath}\n   ↳ Reason: {e}")
        return False

def scan_for_corrupted_files(data_dir):
    corrupted_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                full_path = os.path.join(root, file)
                if not is_valid_wav(full_path):
                    corrupted_files.append(full_path)
    return corrupted_files

if __name__ == "__main__":
    audioTf=Tensorflow()
    audioLib=Librosa()

    assert np.array_equal(audioTf, audioLib)," audio tf is not equal to load from librosa"
    
    print("audioTf ",audioTf.shape)
    spectrogramTf = AudioProcessor.GetSpectrogramTf(audioTf)
    spectrogramLib = AudioProcessor.GetSpectrogramLib(audioLib)

    assert spectrogramTf.shape == spectrogramLib.shape, \
        f"Shape mismatch: TensorFlow shape = {spectrogramTf.shape}, Librosa shape = {spectrogramLib.shape}"
    
    assert np.allclose(spectrogramTf, spectrogramLib, atol=1e-3)
 
    tfAll=LoadAll()
    num_batches = tf.data.experimental.cardinality(tfAll).numpy()
    print(f"tfAll has {num_batches} batches")

    bad_files = scan_for_corrupted_files(DATASET_PATH)
    print(f"\n🧼 Found {len(bad_files)} corrupted files.")

    if bad_files:
        print("🔍 Bad files:")
        for f in bad_files:
            print(f"   - {f}")

    for batch_audio, batch_labels in tfAll:
        print(batch_audio.shape)
        print(batch_labels.shape)
        for batchIndex in range(batch_audio.shape[0]):
            waveform = np.array(tf.squeeze(batch_audio[batchIndex], axis=-1))
            #print("waveform shape ", waveform.shape)
            spectrogramTf = AudioProcessor.GetSpectrogramTf(waveform)
            spectrogramLib = AudioProcessor.GetSpectrogramLib(waveform)
            assert spectrogramTf.shape == spectrogramLib.shape, \
                        f"Shape mismatch: TensorFlow shape = {spectrogramTf.shape}, Librosa shape = {spectrogramLib.shape}"
            assert np.allclose(spectrogramTf, spectrogramLib, atol=1e-2)


    #tfAll2=load_audio_dataset_safe(DATASET_PATH)
    #num_batches = tf.data.experimental.cardinality(tfAll2).numpy()
    #print(f"tfAll2 has {num_batches} batches")
    # last_index_batch=num_batches-1

    # for batchesCount, (batch_audio, batch_labels) in enumerate(tfAll):
    #     if batchesCount ==last_index_batch:
    #         break

    #sampling_rate=22050
    #plt.figure(figsize=(10,4))
    # librosa.display.specshow(data=spectrogramTf, sr=sampling_rate, x_axis='time', y_axis='log', cmap='viridis')
    # plt.title('Spectrogram')
    # plt.colorbar(format='%+02.0f dB')
    # plt.tight_layout()
    # plt.show()
