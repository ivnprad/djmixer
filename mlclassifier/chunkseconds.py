import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import librosa

import AudioProcessor
#DATASET_PATH = pathlib.Path.cwd()/'Data'/'genres_original'

import os
import tensorflow as tf

def load_data(datadir, fs=22050, 
              batch_size=64, 
              validation_split=0.2, 
              seed=42,
              chunk_duration_seconds=None):
    """
    Loads audio data from directory in a manner similar to tf.keras.utils.audio_dataset_from_directory.
    
    Args:
        datadir: A pathlib.Path object or string path to the root data directory. 
                 Assumes that subdirectories are class names and contain .wav files.
        fs: Desired sampling rate.
        batch_size: Batch size.
        validation_split: Fraction of data to use for validation.
        seed: Random seed.
        chunk_duration_seconds: If provided, the audio will be chunked into pieces of this duration (in seconds).
        
    Returns:
        A tuple (train_ds, val_ds) of tf.data.Dataset objects.
    """
    # Create a pattern to match all .wav files in subdirectories.
    file_pattern = os.path.join(str(datadir), "*", "*.wav")
    files_ds = tf.data.Dataset.list_files(file_pattern, seed=seed)
    
    def decode_and_extract_label(filepath):
        # Read file contents.
        audio_binary = tf.io.read_file(filepath)
        # Decode the WAV file.
        waveform, sample_rate = tf.audio.decode_wav(audio_binary, desired_channels=1)
        waveform = tf.squeeze(waveform, axis=-1)
        # Optionally, if the file's sample rate isn't fs, you might want to resample.
        # For now, we assume that the file's sample rate is already fs.
        
        # Optionally chunk the waveform into segments of fixed duration.
        if chunk_duration_seconds is not None:
            chunk_samples = fs * chunk_duration_seconds
            # If the waveform is shorter than the desired chunk length, pad it.
            waveform = tf.pad(waveform, [[0, tf.maximum(0, chunk_samples - tf.shape(waveform)[0])]])
            # Frame into non-overlapping chunks.
            waveform = tf.signal.frame(waveform, frame_length=chunk_samples, frame_step=chunk_samples)
            # The waveform now has shape (num_chunks, chunk_samples)
        # Else, keep the full waveform.
        
        # Extract the label from the file path (assumes folder name is the label)
        parts = tf.strings.split(filepath, os.sep)
        label = parts[-2]
        return waveform, label

    # Map the decoding function over the files dataset.
    ds = files_ds.map(decode_and_extract_label, num_parallel_calls=tf.data.AUTOTUNE)
    
    # If chunking was applied, the waveform shape will be (num_chunks, chunk_samples).
    # To flatten out the chunks so that each one is an individual sample, use flat_map.
    if chunk_duration_seconds is not None:
        ds = ds.flat_map(
            lambda waveform, label: tf.data.Dataset.from_tensor_slices((waveform, tf.repeat(label, tf.shape(waveform)[0])))
        )
    
    # Cache for performance and shuffle the entire dataset.
    ds = ds.cache().shuffle(buffer_size=1000, seed=seed)
    
    # Compute the cardinality (number of samples).
    num_samples = tf.data.experimental.cardinality(ds).numpy()
    if num_samples <= 0:
        raise ValueError("No audio samples found.")
    
    num_val = int(validation_split * num_samples)
    
    # Split into validation and training datasets.
    val_ds = ds.take(num_val)
    train_ds = ds.skip(num_val)
    
    # Batch and prefetch both datasets.
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Assert that the datasets have at least one batch
    tf.debugging.assert_positive(
        tf.data.experimental.cardinality(train_ds).numpy(),
        message="Training batches number must be greater than zero!"
    )
    tf.debugging.assert_positive(
        tf.data.experimental.cardinality(val_ds).numpy(),
        message="Validation batches number must be greater than zero!"
    )

    return train_ds, val_ds


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


def load_audio_chunks_tf(filepath, chunk_samples, fs):
    audio_binary = tf.io.read_file(filepath)
    try:
        audio, sample_rate = tf.audio.decode_wav(audio_binary)
    except Exception as e:
        raise Exception(f"Failed to decode '{filepath}': file is corrupt or invalid.") from e
    
  
    if audio.shape[-1] == 2:
        raise Exception("Stereo audio not handled yet")
    else:
        audio = tf.squeeze(audio, axis=-1)

    sr_value = int(sample_rate.numpy())  # Convert sample_rate to a Python int (works in eager mode)
    if sr_value != fs:
        raise ValueError(f"sampling rate does not match expected fs {fs}") #audio = tf.signal.resample_poly(audio, fs, sr_value) # Resample: up factor = fs, down factor = sr_value
    
    total_samples = tf.shape(audio)[0]
    num_chunks = total_samples // chunk_samples
    truncated_audio = audio[:num_chunks * chunk_samples]
    chunks = tf.reshape(truncated_audio, (num_chunks, chunk_samples))
    
    return chunks

def load_audio_chunks(filepath, chunk_samples, fs):
    # Load the audio file with the desired sampling rate
    audio, sr = librosa.load(filepath, sr=fs)
    
    # Calculate the number of complete chunks available
    num_chunks = len(audio) // chunk_samples
    
    # Split the audio into non-overlapping chunks
    chunks = [
        audio[i * chunk_samples : (i + 1) * chunk_samples]
        for i in range(num_chunks)
    ]
        
    return chunks


def LoadData(datadir
             ,fs=22050
             ,chunkDurationSeconds=3
             ,batch_size = 32
             ,validation_split=0.2
             ,seed=42):
    
    chunk_samples = fs * chunkDurationSeconds  # Total samples per chunk
    if not datadir.exists():
        raise Exception(f"No valid directory: {datadir}")
    audio_chunks = []
    labels = []  # if you have labels encoded by subdirectory names

    # Walk through all files in the dataset directory
    for root, dirs, files in os.walk(datadir):
        for file in files:
            if not file.lower().endswith(".wav"):
                continue
            filepath = os.path.join(root, file)
            #chunks = load_audio_chunks(filepath, chunk_samples, fs)
            chunks = load_audio_chunks_tf(filepath,chunk_samples,fs)
            audio_chunks.extend(chunks)
                
            # Optionally, extract a label from the subdirectory name
            label = os.path.basename(root)
            labels.extend([label] * len(chunks))


    tf.debugging.assert_positive(len(audio_chunks),message="No audio files were found")
    tf.debugging.assert_positive(len(label),message="No labels were found")
    num_samples = len(audio_chunks)
    validation_split = 0.2
    num_val = int(validation_split * num_samples)

    dataset = tf.data.Dataset.from_tensor_slices((audio_chunks, labels))
    dataset = dataset.shuffle(buffer_size=num_samples, seed=seed)

    val_ds = dataset.take(num_val)
    train_ds = dataset.skip(num_val)

    # Batch the datasets
    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)

    # Prefetch for improved performance
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    tf.debugging.assert_positive(tf.data.experimental.cardinality(train_ds).numpy(),
        message="Training batches number must be greater than zero!"
    )
    tf.debugging.assert_positive(tf.data.experimental.cardinality(val_ds).numpy(),
        message="Validation batches number must be greater than zero!"
    )

    unique_class_names = sorted(list(set(labels)))
    train_ds.class_names = unique_class_names
    val_ds.class_names = unique_class_names

    print(f"{num_samples} found beloging to {len(unique_class_names)} classes")

    return train_ds,val_ds

def GetSpectrogram(waveform, n_fft = 2048,hop_length = 512):
    #spectrogram = AudioProcessor.CoreGetSpecrogramTf(waveform)
    tf.debugging.assert_rank(waveform, 1, message="Waveform must be 1D")
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
    num_frames = 1 + (len(waveform) - n_fft) // hop_length
    num_bins = 1 + n_fft // 2
    tf.debugging.assert_equal(spectrogram.shape, (num_frames, num_bins))
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

def get_tf_mel_spectrogram(audio, sample_rate=22050, n_fft=2048, hop_length=512, 
                             n_mels=128, lower_edge_hertz=80.0, upper_edge_hertz=7600.0):
    """
    Computes a mel spectrogram using TensorFlow.
    
    Args:
        audio: 1D numpy array or tf.Tensor containing the waveform.
        sample_rate: Sampling rate of the audio.
        n_fft: FFT window size.
        hop_length: Hop length between frames.
        n_mels: Number of mel bins.
        lower_edge_hertz: Lower frequency bound.
        upper_edge_hertz: Upper frequency bound.
        
    Returns:
        mel_spectrogram: A tf.Tensor of shape (time, n_mels).
    """
    # Ensure audio is a TensorFlow tensor of type float32.
    if not isinstance(audio, tf.Tensor):
        audio = tf.convert_to_tensor(audio, dtype=tf.float32)
        
    # Pad the waveform to center the frames.
    pad = n_fft // 2
    audio = tf.pad(audio, [[pad, pad]], mode='CONSTANT')
    
    # Compute the STFT and take the magnitude.
    stft = tf.signal.stft(
        audio,
        frame_length=n_fft,
        frame_step=hop_length,
        fft_length=n_fft,
        window_fn=tf.signal.hann_window,
        pad_end=False
    )
    spectrogram = tf.abs(stft)
    
    # Create a mel weight matrix and apply it.
    num_spectrogram_bins = spectrogram.shape[-1]
    mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        n_mels,
        num_spectrogram_bins,
        sample_rate,
        lower_edge_hertz,
        upper_edge_hertz
    )
    mel_spectrogram = tf.tensordot(spectrogram, mel_weight_matrix, axes=1)
    # Set static shape for better debugging (optional).
    mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(mel_weight_matrix.shape[-1:]))
    
    return mel_spectrogram

def GetMelSpectrogram(waveform, sample_rate=22050, n_fft=2048, hop_length=512, 
                      num_mel_bins=64, lower_edge_hertz=80.0, upper_edge_hertz=7600.0):
    tf.debugging.assert_rank(waveform, 1, message="Waveform must be 1D")
    pad = n_fft // 2
    waveform = tf.pad(waveform, [[pad, pad]], mode='CONSTANT')

    # Compute the STFT.
    stft = tf.signal.stft(
        waveform,
        frame_length=n_fft,
        frame_step=hop_length,
        fft_length=n_fft,
        window_fn=tf.signal.hann_window,
        pad_end=False
    )
    spectrogram = tf.abs(stft)
    
    # Create the mel weight matrix.
    num_spectrogram_bins = spectrogram.shape[-1]
    mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins,
        num_spectrogram_bins,
        sample_rate,
        lower_edge_hertz,
        upper_edge_hertz
    )
    
    # Apply the mel filter bank.
    mel_spectrogram = tf.tensordot(spectrogram, mel_weight_matrix, axes=1)
    
    # Optionally, set the static shape (useful if you want to assert expected dimensions)
    mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(mel_weight_matrix.shape[-1:]))
    
    # Add a channel dimension.
    mel_spectrogram = mel_spectrogram[..., tf.newaxis]
    return mel_spectrogram


def MakeSpecDs(ds):

    def map_batch(batch_audio, batch_labels):
        batch_specs = tf.map_fn(GetSpectrogram, batch_audio)
        return batch_specs, batch_labels

    return (
        ds
        .map(map_batch, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

# class EarlyStoppingCallback(tf.keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs=None):
#         logs = logs or {}
#         train_acc = logs.get("accuracy")
#         val_acc = logs.get("val_accuracy")
#         if train_acc is not None and val_acc is not None:
#             if train_acc >= 0.95 and val_acc >= 0.95:
#                 print(f"\nEpoch {epoch+1}: Reached 95% training and validation accuracy. Stopping training!")
#                 self.model.stop_training = True

class EarlyStoppingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs['accuracy']>0.95:
            self.model.stop_training = True
            print("\nReached 95% accuracy so cancelling training!")

if __name__ == "__main__":
    fs = 22050                   # Sampling rate
    chunk_duration = 3           # Duration of each chunk in seconds

    #DATASET_PATH = pathlib.Path.cwd()/'mlclassifier'/'Data'/'genres_original'
    DATASET_PATH = pathlib.Path.cwd()/'Data'/'genres_original'
    train_ds, val_ds=LoadData(DATASET_PATH,fs,chunk_duration)
    label_names = np.array(train_ds.class_names)
    print()
    print("label names:", label_names)
    print(train_ds.element_spec)

    test_ds = val_ds.shard(num_shards=2, index=0)
    val_ds = val_ds.shard(num_shards=2, index=1)

    # train_spectrogram_ds = MakeSpecDs(train_ds)
    # val_spectrogram_ds = MakeSpecDs(val_ds)
    # test_spectrogram_ds = MakeSpecDs(test_ds)

    # print(train_spectrogram_ds.element_spec)
    # input_shape= (130, 1025, 1)

    num_labels = len(label_names)
    #sample = next(iter(train_ds.take(1)))
    #print(sample[0].shape)  # If your dataset returns (features, labels)

    #tf.keras.layers.Input(shape=(66150,))
    # norm_layer = tf.keras.layers.Normalization()
    # norm_layer.adapt(data=train_spectrogram_ds.map(map_func=lambda spec, label: spec))

    n_frames = ((66150 - 1) // 512) + 1
    num_mel_bins = 128

    model_cnn3 = tf.keras.models.Sequential([
        # Input raw audio of length 66150
        tf.keras.layers.Input(shape=(66150,)),
        
        # Convert raw audio into a mel spectrogram.
        # This will output a tensor of shape (batch, num_mel_bins, n_frames)
        # With the provided parameters: num_mel_bins=128, sequence_stride=512 and fft_length=2048,
        # n_frames is computed as:
        #    n_frames = floor((66150 - 1) / 512) + 1 = 130
        tf.keras.layers.MelSpectrogram(
            num_mel_bins=num_mel_bins, 
            sampling_rate=22050, 
            sequence_stride=512, 
            fft_length=2048
        ),
        
        # Add an explicit channel dimension (now shape becomes: (128, 130, 1))
        tf.keras.layers.Reshape((128, n_frames, 1)),
        
        # First convolution block
        tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=3, strides=(2,2), padding='same'),
        tf.keras.layers.Dropout(0.2),
        
        # Second convolution block
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=3, strides=(2,2), padding='same'),
        tf.keras.layers.Dropout(0.1),
        
        # Third convolution block
        tf.keras.layers.Conv2D(64, 2, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=2, strides=(2,2), padding='same'),
        tf.keras.layers.Dropout(0.1),
        
        # Flatten and pass through Dense layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_labels)  # Assuming 10 classes for classification
    ])
    
    model_cnn3.summary()

    print("TensorFlow version:", tf.__version__)
    print("Available GPU devices:", tf.config.list_physical_devices('GPU'))





    lookup_layer = tf.keras.layers.StringLookup(vocabulary=label_names, num_oov_indices=0)
    def encode_labels(x, y):
        return x, lookup_layer(y)
    train_ds = train_ds.map(encode_labels)
    val_ds = val_ds.map(encode_labels)


    model_cnn3.compile(optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )   

    EPOCHS = 250
    history = model_cnn3.fit(train_ds, 
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose = 2,
        callbacks = [EarlyStoppingCallback()],
    )

    model_cnn3.save("cnn3.keras")

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

    model_cnn3.save("cnn3.keras")



