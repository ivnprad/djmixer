import tensorflow as tf 
import os
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
from chunkseconds import LoadData
#from tensorflow.keras.utils import plot_model

# Assuming cnn3 is your model
#tf.keras.utils.plot_model(model, to_file="cnn3_model.png", show_shapes=True, show_layer_names=True)


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

    model = tf.keras.models.load_model('cnn3.keras')
    #model.summary()

 

    lookup_layer = tf.keras.layers.StringLookup(vocabulary=label_names, num_oov_indices=0)
    def encode_labels(x, y):
        return x, lookup_layer(y)
    train_ds = train_ds.map(encode_labels)
    val_ds = val_ds.map(encode_labels)
    test_ds = test_ds.map(encode_labels)

    model.evaluate(test_ds, return_dict=True)

    y_pred = model.predict(test_ds)
    print("Raw prediction (probabilities):")
    print(y_pred[0])

    logits = np.array(y_pred[0])
    predicted_index = np.argmax(logits)
    print("Predicted index:", predicted_index)

    predicted_label = label_names[predicted_index]
    print("Predicted label:", predicted_label)

    x = pathlib.Path.cwd()/'Chris_Bell_Elevator_to_Heaven.wav'
    x = tf.io.read_file(str(x))
    x, sample_rate = tf.audio.decode_wav(x, desired_channels=1, desired_samples=22050*3,)
    x = tf.squeeze(x, axis=-1)
    waveform = x
    x_input = tf.expand_dims(x, axis=0)  # shape: (1, 66150)
    y_output = model.predict(test_ds)
    print(y_output[0])
    print("Predicted index:", np.argmax(np.array(y_output[0])))
    print("Predicted label:", label_names[np.argmax(np.array(y_output[0]))])
    
    # y_pred = tf.argmax(y_pred, axis=1)
    # y_true = tf.concat(list(test_ds.map(lambda s,lab: lab)), axis=0)

    # confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(confusion_mtx,
    #             xticklabels=label_names,
    #             yticklabels=label_names,
    #             annot=True, fmt='g')
    # plt.xlabel('Prediction')
    # plt.ylabel('Label')
    # plt.show()

    # cm = confusion_mtx.numpy()
    # per_class_accuracy = np.diag(cm) / np.sum(cm, axis=1)
    # for label, acc in zip(label_names, per_class_accuracy):
    #     print(f"{label:10s}: {acc:.2%}")

    # from sklearn.metrics import classification_report

    # print(classification_report(y_true, y_pred, target_names=label_names))

    # x = data_dir/'no/01bb6a2a_nohash_0.wav'
    # x = tf.io.read_file(str(x))
    # x, sample_rate = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000,)
    # x = tf.squeeze(x, axis=-1)
    # waveform = x
    # x = get_spectrogram(x)
    # x = x[tf.newaxis,...]

    # prediction = model(x)
    # x_labels = ['no', 'yes', 'down', 'go', 'left', 'up', 'right', 'stop']
    # plt.bar(x_labels, tf.nn.softmax(prediction[0]))
    # plt.title('No')
    # plt.show()