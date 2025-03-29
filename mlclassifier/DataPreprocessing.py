# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import json
import scipy
import librosa

import AudioProcessor

from pathlib import Path

#path_data=Path.cwd()/'mlclassifier'/'Data'/'genres_original'
path_data=Path.cwd()/'Data'/'genres_original'
genrePath=path_data/'blues'
filename = 'blues.00000.wav'
file_path = genrePath/filename

if not (os.path.exists(file_path)):
    print(f"does not exist {file_path}")

fs = 22050 # sampling rate for librosa to resample to
audio_ex, fs = librosa.load(path=file_path, sr=fs) # load audio and sampling rate

#AudioProcessor.PlotWaveform(audio_ex,fs)
#AudioProcessor.PlotSpectrum(audio_ex,fs)

#Plot Spectrogram
n_fft = 2048 # FFT window size
hop_length = 512 # number audio of frames between STFT columns
dB = True # True for dB scale
AudioProcessor.PlotSpectrogram(audio_ex,fs, n_fft,hop_length,dB)

# Plot Mel-Spectrogram
#n_mfccs = 256 # number of mel coeffs
#n_fft = 2048 # FFT window size
#hop_length = 512 # number audio of frames between STFT columns
#plot_mel_spectrogram_audio(audio_ex, fs, n_mfccs=n_mfccs, n_fft=n_fft, hop_length=hop_length, fig_size=(12,6))
#AudioProcessor.PlotMelSpectrogramAudio(audio_ex,fs,n_mfccs,n_fft,hop_length)

# Set up the parameters for the MFCC conversion
# n_mfcc = 13
# n_fft = 2048
# hop_length = 512
# num_segments = 10 # split teack into 10 segments (3 sec each)
# track_duration = 30 # Length of tracks (sec)

# mfccs, genres, genre_nums = AudioProcessor.GetMfccs(directory_path=path_data,
#                           fs=fs,
#                           duration=track_duration,
#                           n_fft=n_fft,
#                           hop_length=hop_length,
#                           n_mfcc=n_mfcc, 
#                           num_segments=num_segments)

# # Review mfccs and genres for the correct shape
# print(f"MFCCs: {mfccs.shape}")
# print(f"genres: {genres.shape}")

# # Map target genre to number
# genre_map = dict(zip(sorted(set(genres)), np.arange(0, 10)))
# genres_num = np.array(pd.Series(genres).map(genre_map))
# # list(zip(genres_num, genres)) # view mapped target

# # Plot an MFCC example
# idx = 0
# print(f"{genres[idx].title()}")
# AudioProcessor.PlotMfcc(mfccs[idx].T, fs)
# plt.title(f"{genres[idx].title()}")
