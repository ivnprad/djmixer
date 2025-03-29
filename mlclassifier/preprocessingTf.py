import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import librosa

import AudioProcessor
DATASET_PATH = pathlib.Path.cwd()/'Data'/'genres_original'


def GetSpectrogram(waveform):
    #spectrogram = AudioProcessor.CoreGetSpecrogramTf(waveform)
    tf.debugging.assert_rank(waveform, 1, message="Waveform must be 1D")

    n_fft = 2048
    hop_length = 512
    pad = n_fft // 2
    waveform = tf.pad(waveform, [[pad, pad]], mode='CONSTANT')

    spectrogram = tf.signal.stft(
        waveform,
        frame_length=n_fft,
        frame_step=hop_length,
        fft_length=n_fft,
        window_fn=tf.signal.hann_window,
        pad_end=False
    )

    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

#   This dataset only contains single channel audio, so use the tf.squeeze function to drop the extra axis:
def Squeeze(audio, labels):
    audio = tf.squeeze(audio, axis=-1)
    return audio, labels

def LoadData(directory, fs=22050,seconds=30):
    if not directory.exists():
        raise ValueError(f"No valid directory: {directory}")
    
    # genres = np.array(tf.io.gfile.listdir(str(DATASET_PATH)))
    # genres = genres[(genres != 'README.md') & (genres != '.DS_Store')]
    # print('Genres:', genres)

    train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
                                directory=DATASET_PATH,
                                batch_size=64,
                                validation_split=0.2,
                                seed=0,
                                output_sequence_length=fs*seconds,
                                subset='both')

    return train_ds,val_ds

def PlotExample(exampleAudio,exampleLabels,label_names):
    plt.figure(figsize=(16, 10))
    rows = 3
    cols = 3
    n = rows * cols
    for i in range(n):
        plt.subplot(rows, cols, i+1)
        audio_signal = exampleAudio[i]
        plt.plot(audio_signal)
        plt.title(label_names[exampleLabels[i]])
        plt.yticks(np.arange(-1.2, 1.2, 0.2))
        plt.ylim([-1.1, 1.1])
    plt.show()

def PlotSpectrogram(spectrogram):
    spectrogram = np.squeeze(spectrogram, axis=-1)
    spectrogram = spectrogram.T
    print(spectrogram.shape)
    stft = np.abs(spectrogram)
    stft_db = librosa.amplitude_to_db(stft, ref=np.max)
    fig_size=(12,6)
    print(f" stft_db shape {stft_db.shape}")
    plt.figure(figsize=fig_size)
    librosa.display.specshow(data=stft_db, sr=fs, x_axis='time', y_axis='log', cmap='viridis')
    plt.title('Spectrogram')
    plt.colorbar(format='%+02.0f dB')
    plt.tight_layout()
    plt.show()

def MakeSpecDs(ds):

    def map_batch(batch_audio, batch_labels):
        batch_specs = tf.map_fn(GetSpectrogram, batch_audio)
        return batch_specs, batch_labels

    return (
        ds
        .map(map_batch, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    # return ds.map(
    #   map_func=lambda audio,label: (GetSpectrogram(audio), label),
    #   num_parallel_calls=tf.data.AUTOTUNE)

def make_spec_ds(ds):
    return ds.map(
      map_func=lambda audio,label: (GetSpectrogram(audio), label),
      num_parallel_calls=tf.data.AUTOTUNE)

def preprocess(audio, label):
    spectrogram = GetSpectrogram(audio)
    return spectrogram, label

def GetSpectrogramLib(waveform):
    n_fft = 2048 
    hop_length = 512 
    stft = np.abs(librosa.stft(y=waveform, n_fft=n_fft, hop_length=hop_length))
    #stft = np.abs(librosa.stft(y=waveform, n_fft=n_fft, hop_length=hop_length, window='hann', center=True))
    #stft = librosa.amplitude_to_db(stft, ref=np.max) # ref - set max to 0 dB
    return stft

def CheckTfSpectrogram(tfSpectrogram,waveform):
    assert tfSpectrogram.shape == (1292, 1025, 1), f"Unexpected spectrogram shape: {tfSpectrogram.shape}"
    #spectrogram = tf.squeeze(tfSpectrogram, axis=-1)
    #stft = np.abs(spectrogram.numpy().T)
    #stft_db=librosa.amplitude_to_db(stft, ref=np.max)

    coreSpectrogram = AudioProcessor.CoreGetSpecrogramTf(waveform, 22050)
    stft = np.abs(coreSpectrogram.numpy().T)
    stft_db = librosa.amplitude_to_db(stft, ref=np.max)
    assert stft_db.shape == (1025, 1292), f"Unexpected stft_db shape: {tfSpectrogram.shape}"

    spectrogramLib = GetSpectrogramLib(waveform.numpy())
    assert spectrogramLib.shape == (1025, 1292), f"Unexpected spectrogramLib shape: {tfSpectrogram.shape}"

    #assert np.allclose(stft_db, spectrogramLib, atol=1e-3)
    if not np.allclose(stft_db, spectrogramLib, atol=1e-3):
        diff = np.abs(stft_db - spectrogramLib)
        max_diff = np.max(diff)
        print("🚨 Max absolute difference:", max_diff)
        raise ValueError()
    
def CheckSpectrogramLibrosa(spectrogram,wave):
    spectrogram2D = tf.squeeze(spectrogram, axis=-1)
    stft = np.abs(spectrogram2D.numpy().T)
    spectrogramTf=librosa.amplitude_to_db(stft, ref=np.max)
    #spectrogramTf = AudioProcessor.GetSpectrogramTf(np.array(waveform))

    spectrogramLib = AudioProcessor.GetSpectrogramLib(np.array(wave))
    assert spectrogramTf.shape == spectrogramLib.shape, \
                f"Shape mismatch: TensorFlow shape = {spectrogramTf.shape}, Librosa shape = {spectrogramLib.shape}"
    assert np.allclose(spectrogramTf, spectrogramLib, atol=1e-2)

def make_spec_ds_unbatch(ds, batch_size=64):
    return (
        ds
        #.unbatch()  # Split batch into individual samples
        .map(lambda audio, label: (GetSpectrogram(audio), label), num_parallel_calls=tf.data.AUTOTUNE)
        #.batch(batch_size,drop_remainder=True)  # Rebatch for training
        .prefetch(tf.data.AUTOTUNE)
    )

if __name__ == "__main__":
    fs=22050
    seconds = 30

    # Set the seed value for experiment reproducibility.
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)

    DATASET_PATH = pathlib.Path.cwd()/'Data'/'genres_original'
    train_ds, val_ds=LoadData(DATASET_PATH)
    label_names = np.array(train_ds.class_names)
    print()
    print("label names:", label_names)
    print(train_ds.element_spec)

    #This dataset only contains single channel audio, so use the tf.squeeze function to drop the extra axis:
    train_ds = train_ds.map(Squeeze, tf.data.AUTOTUNE) 
    val_ds = val_ds.map(Squeeze, tf.data.AUTOTUNE)
    print(train_ds.element_spec)

    test_ds = val_ds.shard(num_shards=2, index=0)
    val_ds = val_ds.shard(num_shards=2, index=1)

    #print(label_names[[1,1,3,0]])

    # for example_audio, example_labels in train_ds:
    #     print(example_audio.shape)
    #     print(example_labels.shape)
    #     break  # optional — just to get the first batch

    #PlotExample(example_audio,example_labels,label_names)

    # for i in range(3):
    #     label = label_names[example_labels[i]]
    #     waveform = example_audio[i]
    #     print('Waveform shape:', waveform.shape)
    #     assert waveform.shape == (661500,), f"Unexpected waveform shape: {waveform.shape}"
    #     example_spectrogram = GetSpectrogram(waveform)
    #     print('Label:', label)
    #     print('Spectrogram shape:', example_spectrogram.shape)
    #     assert example_spectrogram.shape == (1292, 1025, 1), f"Unexpected spec shape: {example_spectrogram.shape}"

    #PlotSpectrogram(example_spectrogram)

    # all_spectrograms = []
    # all_labels = []
    # num_batches = tf.data.experimental.cardinality(train_ds).numpy()
    # print(f"train_ds has {num_batches} batches")
    # last_index_batch=num_batches-1

    # for batchesCount, (batch_audio, batch_labels) in enumerate(train_ds):
    #     if batchesCount ==last_index_batch:
    #         break
    #     for batchIndex in range(batch_audio.shape[0]):
    #         waveform = batch_audio[batchIndex]
    #         assert waveform.shape == (661500,), f"Unexpected waveform shape: {waveform.shape}"
    #         label = batch_labels[batchIndex]
    #         spectrogram = GetSpectrogram(waveform)
    #         assert spectrogram.shape == (1292, 1025, 1), f"Unexpected spectrogram shape: {waveform.shape}"
    #         CheckSpectrogramLibrosa(spectrogram,wave=waveform)
    #         all_spectrograms.append(spectrogram)
    #         all_labels.append(label)
    
    # train_spectrogram_ds = tf.data.Dataset.from_tensor_slices((all_spectrograms, all_labels)).batch(64)
    # print(train_spectrogram_ds.element_spec)
    # num_batches = tf.data.experimental.cardinality(train_spectrogram_ds).numpy()
    # print(f"train_spectrogram_ds has {num_batches} batches")


    train_spectrogram_ds = MakeSpecDs(train_ds)
    val_spectrogram_ds = MakeSpecDs(val_ds)
    test_spectrogram_ds = MakeSpecDs(test_ds)

    # train_spectrogram_ds = make_spec_ds_unbatch(train_ds)
    # val_spectrogram_ds = make_spec_ds_unbatch(val_ds)
    # test_spectrogram_ds = make_spec_ds_unbatch(test_ds)
    # num_batches = tf.data.experimental.cardinality(train_spectrogram_ds).numpy()
    # print(f"train_spectrogram_ds has {num_batches} batches")
    # count = sum(1 for _ in train_spectrogram_ds)
    # print(f"train_spectrogram_ds has {count} batches")

    print(train_spectrogram_ds.element_spec)
    input_shape= (1292, 1025, 1)
    num_labels = len(label_names)

    norm_layer = tf.keras.layers.Normalization()
    norm_layer.adapt(data=train_spectrogram_ds.map(map_func=lambda spec, label: spec))

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Resizing(32, 32),

        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.4),
        tf.keras.layers.RandomTranslation(0.2,0.2),
        tf.keras.layers.RandomContrast(0.4),
        tf.keras.layers.RandomZoom(0.2),

        norm_layer,
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_labels),
    ])

    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )   

    EPOCHS = 30
    history = model.fit(
        train_spectrogram_ds,
        validation_data=val_spectrogram_ds,
        epochs=EPOCHS,
        #callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
    )

    metrics = history.history
    plt.figure(figsize=(16,6))
    plt.subplot(1,2,1)
    plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.ylim([0, max(plt.ylim())])
    plt.xlabel('Epoch')
    plt.ylabel('Loss [CrossEntropy]')

    plt.subplot(1,2,2)
    plt.plot(history.epoch, 100*np.array(metrics['accuracy']), 100*np.array(metrics['val_accuracy']))
    plt.legend(['accuracy', 'val_accuracy'])
    plt.ylim([0, 100])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy [%]')
    plt.show()