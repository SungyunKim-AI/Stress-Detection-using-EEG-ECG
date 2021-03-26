import numpy as np
import tensorflow as tf

# mne imports
import mne
from mne import io

# tools for plotting confusion matrices
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from utils import plot_loss_curve, plot_acc_curve, normalization
from load_ecg_data import load_ecg_data
from ecg_model import generate_model


ECG, Labels = load_ecg_data()
#ECG = normalization(ECG, "01")

#ECG = np.array(ECG)
#Labels = np.array(Labels)

X_train, X_validate, Y_train, Y_validate = train_test_split(
    ECG, Labels, test_size=0.3, random_state=42)
X_validate, X_test, Y_validate, Y_test = train_test_split(
    X_validate, Y_validate, test_size=0.5, random_state=42)


kernels, chans, samples = 1, 1, 10240

X_train = X_train.reshape(X_train.shape[0], samples, chans)
X_validate = X_validate.reshape(X_validate.shape[0], samples, chans)
X_test = X_test.reshape(X_test.shape[0], samples, chans)

###################### model ######################

model = generate_model()

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

fit_model = model.fit(
    X_train,
    Y_train,
    epochs=300,
    batch_size=20,
    validation_data=(X_validate, Y_validate)
)

plot_loss_curve(fit_model.history)
plot_acc_curve(fit_model.history)
