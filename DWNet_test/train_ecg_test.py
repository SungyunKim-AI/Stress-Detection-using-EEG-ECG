from load_ecg_data import load_ecg_data
from utils import plot_loss_curve, plot_acc_curve, normalization

import numpy as np
import tensorflow as tf

# mne imports
import mne
from mne import io

# tools for plotting confusion matrices
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


ECG, Labels = load_ecg_data()
ECG = normalization(ECG, "01")

ECG = np.array(ECG)
Labels = np.array(Labels)

X_train, X_validate, Y_train, Y_validate = train_test_split(
    ECG, Labels, test_size=0.4, random_state=42)
X_validate, X_test, Y_validate, Y_test = train_test_split(
    X_validate, Y_validate, test_size=0.5, random_state=42)


kernels, chans, samples = 1, 1, 10240

X_train = X_train.reshape(X_train.shape[0], chans, samples, kernels)
X_validate = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
X_test = X_test.reshape(X_test.shape[0], chans, samples, kernels)

print('X_train shape:', X_train.shape)


with tf.device('GPU:0'):

    model = testNet(nb_classes=2, Chans=chans, Samples=samples)

    # compile the model and set the optimizers
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    print(X_train.shape)
    print(X_validate.shape)


    fittedModel = model.fit(X_train, Y_train, batch_size=1, epochs=500,
                            verbose=2, validation_data=(X_validate, Y_validate))

    probs = model.predict(X_test)
    preds = probs.argmax(axis=-1)
    acc = np.mean(preds == Y_test.argmax(axis=-1))
    print("Classification accuracy: %f " % (acc))

    plot_loss_curve(fittedModel.history)
    plot_acc_curve(fittedModel.history)
