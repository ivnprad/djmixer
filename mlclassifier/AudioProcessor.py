import librosa
import matplotlib.pyplot as plt
import scipy
import numpy as np
import json
import math
import os
import tensorflow as tf

def CheckType(value,expectedType):
    if not isinstance(value, expectedType):
        raise TypeError(f"The parameter must be a {expectedType.__name__}.")

def PlotWaveform(audio:np.ndarray, fs:int):
    """Plots the waveform of audio in the time domain.
    
    Parameters:
        audio (numpy.ndarray): audio signal
        fs (int): sampling frequency (Hz) of audio signal
        
    """
    CheckType(audio,np.ndarray)
    CheckType(fs,int)

    plt.figure(figsize = (12, 6))
    librosa.display.waveshow(audio, sr=fs, alpha=0.8)
    plt.title('Audio Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel("Amplitude")
    plt.show()

def CalcSpectrumDb(audio:np.ndarray, kind='mag'):
    """Calculates the spectrum of an audio signal.
    
    Parameters:
        audio (numpy.ndarray): audio signal
        kind (str): type of spectrum to calculate
            - 'mag' = magnitude spectrum (real values)
            - 'phs' = phase spectrum (imaginary values)
            - 'com' = complex spectrum (both magnitude and phase)
    Returns:
        spec_db (numpy.ndarray) = the audio spectrum in db scale (re 20 μPa)
    """
    CheckType(audio,np.ndarray)
    spec = scipy.fft.fft(x=audio)
    
    if kind.lower() == 'm' or 'mag' or 'magnitude':
        spec_db = 20*np.log10(np.abs(spec))
    elif kind.lower() == 'p' or 'pha' or 'phase':
        spec_db = 20*np.log10(np.imag(spec))
    elif kind.lower() == 'c' or 'com' or 'complex':
        spec_db = 20*np.log10(spec)
    else:
        raise ValueError(f"Unknown kind: {kind}")
        
    return spec_db

# Function to plot spectrum
def PlotSpectrum(audio, fs, kind='mag'):
    CheckType(audio,np.ndarray)
    """Plots the magnitude spectrum of an audio signal/
    
    Parameters:
        audio (numpy.ndarray): audio signal
        fs (int): sampling frequency (Hz) of audio signal
        kind (str): type of spectrum to calculate
            - 'mag' = magnitude spectrum (real values)
            - 'phs' = phase spectrum (imaginary values)
            - 'com' = complex spectrum (both magnitude and phase)
    """
    # Calculate fft
    spec_db = CalcSpectrumDb(audio, kind=kind)
    f_axis = np.linspace(0, fs, len(spec_db))#, endpoint=False) # create 
    
    nyquistFrency=2.5 # for safety we use 2.5
    f_axis = f_axis[:int(len(spec_db)/nyquistFrency)]
    spec_db = spec_db[:int(len(spec_db)/nyquistFrency)]
    
    # Plot
    ax = plt.figure(figsize = (12, 6))
    plt.plot(f_axis, spec_db, alpha=1.0)
    plt.xscale('log')
    plt.xlim(1, int(fs/nyquistFrency))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    if fs < 44100:
        plt.xticks([1, 2, 4, 8, 16, 31, 63, 125, 250,500,1000,2000,5000,10000], 
                   ["1", "2", "4", "8", "16", "31", "63", "125", "250", "500", "1K", "2K", "5K", "10K"])
    else:
        plt.xticks([1, 2, 4, 8, 16, 31, 63, 125, 250,500,1000,2000,5000,10000, 20000], 
                   ["1", "2", "4", "8", "16", "31", "63", "125", "250", "500", "1K", "2K", "5K", "10K", "20k"])
    plt.show()
        
def CalcStft(audio, fs, n_fft=2048, hop_length=512, dB=True):
    CheckType(audio,np.ndarray)
    """Calculate the sfort-time-fourier-transform over an audio signal
    
    Parameters:
        audio (numpy.ndarray): audio signal
        fs (int): sampling frequency (Hz) of audio signal
        n_fft (int): The length (i.e. resolution) of the FFT window (must be power of 2)
        hop_length (int): The number of samples between successive frames
        dB (str):
            - True: Convert to dB scale (aka log scale)
            - False: Do not convert to dB (aka linear scale)
    
    Returns:
        stft (numpy.ndarray): Short-time Fourier Transform of the audio signal (i.e. frequency domain data)
    """
    # Calculate STFTs (Short-time Fourier transform) over full audio length
    # absolute value to calculate magnitude (drop imag values)
    stft = np.abs(librosa.stft(y=audio, n_fft=n_fft, hop_length=hop_length))
    
    # Convert to dB (aka log-scale)
    if dB:
        stft = librosa.amplitude_to_db(stft, ref=np.max) # ref - set max to 0 dB

    return stft

def PlotSpectrogram(audio, fs, n_fft=2048, hop_length=512, dB=True, fig_size=(12,6)):
    CheckType(audio,np.ndarray)
    """Plots audio spectrogram.
    
    Parameters:
        audio (numpy.ndarray): audio signal
        fs (int): sampling frequency (Hz) of audio signal
        n_fft (int): The length (i.e. resolution) of the FFT window (must be power of 2)
        hop_length (int): The number of samples between successive frames
        dB (str):
            - True: Convert to dB scale (aka log scale)
            - False: Do not convert to dB (aka linear scale)
        fig_size (tuple): Dimensions of figure
    """
    # Calculate STFTs
    print(f" audio shape {audio.shape}")
    stft_db = CalcStft(audio, fs, n_fft=n_fft, hop_length=hop_length, dB=dB)
    
    print(f" stft_db shape {stft_db.shape}")
    plt.figure(figsize=fig_size)
    librosa.display.specshow(data=stft_db, sr=fs, x_axis='time', y_axis='log', cmap='viridis')

    # Put a descriptive title on the plot
    plt.title('Spectrogram')

    # draw a color bar
    plt.colorbar(format='%+02.0f dB')

    # Make the figure layout compact
    plt.tight_layout()
    plt.show()

def CalcMelSpectrogram(audio, fs, n_mfccs=128, n_fft=2048, hop_length=512):
    """calculates the Mel-scaled spectrogram.
    
    Parameters:
        audio (numpy.ndarray): audio signal
        fs (int): sampling frequency (Hz) of audio signal
        n_mfccs: The number of MFCCs to compute (i.e. dimensionality of mel spectrum)
        n_fft (int): The length (i.e. resolution) of the FFT window (must be power of 2)
        hop_length (int): The number of samples between successive frames

    Returns:
        mel_spec_db (numpy.ndarray): Mel-scaled spectrogram in dB
    """
    # Mel-scaled power spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=fs, n_mels=n_mfccs, n_fft=n_fft, hop_length=hop_length)

    # Convert to dB (aka log-scale)
    mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max) # ref - set max to 0 dB
    
    return mel_spec_db

def PlotMelSpectrogramAudio(audio, fs, n_mfccs=128, n_fft=2048, hop_length=512, fig_size=(12,6)):
    """Plots the mel-scaled spectrogram from audio signal.
    
    Parameters:
        audio (numpy.ndarray): audio signal
        fs (int): sampling frequency (Hz) of audio signal
        n_mfccs: The number of MFCCs to compute (i.e. dimensionality of mel spectrum)
        n_fft (int): The length (i.e. resolution) of the FFT window (must be power of 2)
        hop_length (int): The number of samples between successive frames
        fig_size (tuple): Dimensions of figure
    """
    # Calculate mel-spectrogram
    mel_spec_db = CalcMelSpectrogram(audio, fs, n_mfccs=n_mfccs, n_fft=n_fft, hop_length=hop_length)
    
    # Plot Spectrogram
    plt.figure(figsize=fig_size)
    librosa.display.specshow(data=mel_spec_db, sr=fs, x_axis='time', y_axis='mel', cmap='viridis')

    # Put a descriptive title on the plot
    plt.title('Mel Power Spectrogram')

    # draw a color bar
    plt.colorbar(format='%+02.0f dB')

    # Make the figure layout compact
    plt.tight_layout()
    plt.show()

def PlotMfcc(mfcc: np.array, fs, fig_size=(12,6)):
    """Plots the mel-scaled spectrogram from mfccs. This is performing the same task as
    'plot_mel_spectrogram_audio' with just a different input.
    
    Parameters:
        mfcc (numpy.ndarray): mfccs of an audio signal
        fs (int): sampling frequency (Hz) of audio signal
        fig_size (tuple): Dimensions of figure
    """
    # Plot Spectrogram
    plt.figure(figsize=fig_size)
    
    # Display the spectrogram on a mel scale
    # sample rate and hop length parameters are used to render the time axis
    # abs on signal for better visualization
    librosa.display.specshow(data=mfcc, sr=fs, x_axis='time', y_axis='linear', cmap='viridis')
    
    # Put a descriptive title on the plot
    #plt.title('MFCCs')

    # draw a color bar
    plt.colorbar(format='%+02.0f dB')

    # Make the figure layout compact
    plt.tight_layout()
    #plt.show()


# Load the audio files and convert them to MFCCs
# code modified from https://www.youtube.com/watch?v=szyGiObZymo&list=PL-wATfeyAMNrtbkCNsLcpoAyBBRJZVlnf&index=12

def GetMfccs(directory_path, fs=22500, duration=30, n_fft=2048, hop_length=512, n_mfcc=13, num_segments=10):
    """Reads through a directory of audio files and saves a dictionary of MFCCs and genres to a .json file. It also returns
    numpy.ndarrays for MFCCs, genre name, and genre number for each segment of audio signal.
    
    Parameters:
        audio (numpy.ndarray): audio signal
        fs (int): sampling frequency (Hz) of audio signal
        duration (int): duration of audio signal (sec)
        n_fft (int): The length (i.e. resolution) of the FFT window (must be power of 2)
        hop_length (int): The number of samples between successive frames
        n_mfccs: The number of MFCCs to compute (i.e. dimensionality of mel spectrum)
        num_segments (int): number of segments for the audio signal to be broken up into
        
    Returns:
        "mfcc" (numpy.ndarray): MFCC vectors for each segment
        "genre_name" (numpy.ndarray): name of the genre for each segment (i.e. blues, classical, etc.)
        "genre_num" (numpy.ndarray): number value of the genre for each segment (i.e. 0, 1, 2, etc.)
    """
    data = {
        "genre_name": [],   # name of the genre (i.e. blues, classical, etc.)
        "genre_num": [],    # number value of the genre (i.e. 0, 1, 2, etc.)
        "mfcc": []          # the mfcc vectors
    }

    
    # Calculate the number of samples per segment and the 
    samples_per_track = fs * duration # Calculate the number of samples for desired "duration" of track
    samps_per_segment = int(samples_per_track/num_segments) # number of samples per segment
    mfccs_per_segment = math.ceil(samps_per_segment/hop_length) # number of MFCC vectors per segment
    
    # Loop through all folders & files in the data directory
        # path_current: Path to the current folder (start at outermost folder, then 'walk' in)
        # folder_names: List of names of all folders within the current folder
        # file_names: names List of names of all files within the current folder
        # i: index of current iteration
    print("MFCC collection started!")
    print("========================")

    walk_generator = os.walk(directory_path)
    # Skip the first iteration
    next(walk_generator)

    for i, (path_current, folder_names, file_names) in enumerate(walk_generator):
        
        # Check to make sure that the current folder is not the parent folder
        if path_current is not directory_path:
        
            # Save 
            path_list = path_current.split('/') # split the path into a list
            genre_current = path_list[-1] # select last item in path list (name of folder = genre)
            
            # Loop through files for each genre (sub-directory)
            for file in file_names:
                
                # Load audio data
                file_path = os.path.join(path_current, file).replace(os.sep, '/') # create audio file path

                # try/except to skip a few files that create issues
                try:
                    # Load audio data and sampling frequency
                    audio, fs = librosa.load(file_path, sr=fs) # audio in samples, sampling rate

                    # Loop through audio file for specified number of segments to calculate MFCCs
                    for seg in range(num_segments):

                        # Calculate the samples to bound each segment
                        start_sample = seg * samps_per_segment # segment starting sample
                        end_sample = start_sample + samps_per_segment # segment ending sample

                        # Calculate segment MFCC
                        mfcc = librosa.feature.mfcc(y=audio[start_sample:end_sample],    # audio signal
                                                    sr=fs,                               # sampling rate (Hz)
                                                    n_fft=n_fft,                         # fft window size
                                                    hop_length=hop_length,               # hop size
                                                    n_mfcc=n_mfcc)                       # number of mfccs to compute

                        mfcc = mfcc.T # transpose for appropriate list appending

                        # Confirm correct number of mfccs for each segment, then append
                        if len(mfcc) == mfccs_per_segment:
                            data["genre_name"].append(genre_current) # append current genre to list of genres
                            data["genre_num"].append(i-1) # append current genre to list of genres
                            data["mfcc"].append(mfcc.tolist()) # append current mfcc to list of mfccs
                except:
                    continue

            # Print update status
            print(f"Collected MFCCs for {genre_current.title()}!")
    
    with open('data.json', "w") as filepath:
        print("========================")
        print("Saving data to disk...")
        json.dump(data, filepath, indent=4)# ensure_ascii=False
        print("Saving complete!")
        print("========================")
    
    # option to return MFCCs and genres
    return np.array(data["mfcc"]), np.array(data["genre_name"]), np.array(data["genre_num"])

def GetSpectrogramLib(waveform,n_fft = 2048,hop_length = 512):
    stft = np.abs(librosa.stft(y=waveform, n_fft=n_fft, hop_length=hop_length))
    stft = librosa.amplitude_to_db(stft, ref=np.max) # ref - set max to 0 dB
    return stft

def GetSpectrogramTf(waveform, sample_rate=22050):
    coreSpectrogram = CoreGetSpecrogramTf(waveform, sample_rate)
    stft = np.abs(coreSpectrogram.numpy().T)
    return librosa.amplitude_to_db(stft, ref=np.max)

def CoreGetSpecrogramTf(waveform, sample_rate=22050):

    n_fft = 2048
    hop_length = 512
    pad = n_fft // 2
    waveform = tf.pad(waveform, [[pad, pad]], mode='CONSTANT')

    # Perform STFT with matching params
    spectrogram = tf.signal.stft(
        waveform,
        frame_length=n_fft,
        frame_step=hop_length,
        fft_length=n_fft,
        window_fn=tf.signal.hann_window,
        pad_end=False  # we already padded manually
    )

    return spectrogram


# import tensorflow as tf
# import os

def load_and_chunk_wav(filepath, label, fs=22050, chunk_seconds=3):
    audio_binary = tf.io.read_file(filepath)
    waveform, _ = tf.audio.decode_wav(audio_binary)
    waveform = tf.squeeze(waveform, axis=-1)

    total_samples = fs * 30
    chunk_size = fs * chunk_seconds
    num_chunks = total_samples // chunk_size

    waveform = waveform[:total_samples]  # Trim or pad to 30s
    chunks = tf.reshape(waveform, (num_chunks, chunk_size))  # shape: (10, 66150)

    labels = tf.repeat(label, repeats=num_chunks)

    return chunks, labels

def flatten_chunks(waveform_chunks, labels):
    return tf.data.Dataset.from_tensor_slices((waveform_chunks, labels))

def build_chunked_dataset(data_dir, fs=22050, chunk_seconds=3, batch_size=64):
    class_names = sorted([
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
    ])
    class_to_index = {name: i for i, name in enumerate(class_names)}

    filepaths = []
    labels = []
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        for fname in os.listdir(class_dir):
            if fname.endswith('.wav'):
                filepaths.append(os.path.join(class_dir, fname))
                labels.append(class_to_index[class_name])

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    # Apply slicing and flattening
    ds = ds.map(lambda path, label: load_and_chunk_wav(path, label, fs, chunk_seconds),
                num_parallel_calls=tf.data.AUTOTUNE)
    
    # Flatten (chunk tensor, label) -> individual chunk-label pairs
    ds = ds.flat_map(flatten_chunks)

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
