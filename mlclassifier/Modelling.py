# Imports
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import librosa
from pathlib import Path

import AudioProcessor

# Sklean
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# TensorFlow
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.models import Sequential, save_model
# from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
# from tensorflow.keras.regularizers import l2
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf

path_data=Path.cwd()/'Data'/'data.json'

with open(path_data, "r") as fp:
    data = json.load(fp)

# Plot MFCC example to ensure data was imported properly
idx = 1000
fs = 22500

AudioProcessor.PlotMfcc(np.array(data["mfcc"])[idx].T, fs)
plt.title(f"{np.array(data['genre_name'])[idx].title()}")
plt.show()

