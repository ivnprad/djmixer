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

import tensorflow as tf
import numpy as np
import librosa

def get_tf_mel_spectrogram(audio, sample_rate=22050, n_fft=2048, hop_length=512, 
                             n_mels=64, lower_edge_hertz=80.0, upper_edge_hertz=7600.0):
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

def compare_mel_spectrograms(audio, fs, n_fft, hop_length, n_mels, tol=0.01):
    """
    Computes mel spectrograms with TensorFlow and librosa, compares them, 
    and checks if they are close within a specified tolerance.
    
    Args:
        audio: 1D numpy array containing the waveform.
        fs: Sampling rate.
        n_fft: FFT window size.
        hop_length: Hop length between frames.
        n_mels: Number of mel bins.
        tol: Absolute tolerance for closeness.
        
    Returns:
        is_close: Boolean indicating whether the two mel spectrograms are close.
    """
    # Compute mel spectrogram with librosa.
    # Note: librosa returns a matrix with shape (n_mels, time)
    librosa_mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=fs, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    
    # Compute mel spectrogram with TensorFlow.
    tf_mel_spec = get_tf_mel_spectrogram(
        audio, sample_rate=fs, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    # Convert to numpy and transpose to match librosa's shape.
    tf_mel_spec_np = tf_mel_spec.numpy().T
    
    # Use numpy to check if the matrices are close within the tolerance.
    is_close = np.allclose(tf_mel_spec_np, librosa_mel_spec, atol=tol)
    
    # Alternatively, you can use TensorFlow's assert_near (this will raise an exception if not close):
    try:
        tf.debugging.assert_near(
            tf.convert_to_tensor(tf_mel_spec_np, dtype=tf.float32),
            tf.convert_to_tensor(librosa_mel_spec, dtype=tf.float32),
            atol=tol
        )
        print("TF and Librosa mel spectrograms are close within tolerance", tol)
    except tf.errors.InvalidArgumentError as e:
        print("Mel spectrograms are not close:", e)
    
    return is_close

# Example usage:
# audio = librosa.load('your_audio_file.wav', sr=22050)[0]
# result = compare_mel_spectrograms(audio, fs=22050, n_fft=2048, hop_length=512, n_mels=64, tol=0.01)
# print("Are the mel spectrograms close?", result)


if __name__ == "__main__":
    
    DATASET_PATH = pathlib.Path.cwd()/'mlclassifier'/'Data'/'genres_original'/'blues/blues.00000.wav'
    audio = librosa.load(DATASET_PATH, sr=22050)[0]
    result = compare_mel_spectrograms(audio, fs=22050, n_fft=2048, hop_length=512, n_mels=64, tol=0.01)
    print("Are the mel spectrograms close?", result)