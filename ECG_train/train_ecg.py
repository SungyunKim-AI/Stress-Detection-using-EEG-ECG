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
from ECGModel_v1 import ECGModel_v1

# Load ECG Data
ECG, Labels = load_ecg_data()

kernels, chans, samples = 1, 1, ECG.shape[1]

# Train : Validate : Test = 7 : 1.5 : 1.5
X_train, X_validate, Y_train, Y_validate = train_test_split(
    ECG, Labels, test_size=0.3, random_state=42)
X_validate, X_test, Y_validate, Y_test = train_test_split(
    X_validate, Y_validate, test_size=0.5, random_state=42)

X_train = X_train.reshape(X_train.shape[0], samples, chans)
X_validate = X_validate.reshape(X_validate.shape[0], samples, chans)
X_test = X_test.reshape(X_test.shape[0], samples, chans)

print("Train Set Shape : ", X_train.shape)
print("Validate Set Shape : ", X_validate.shape)
print("Test Set Shape : ", X_test.shape)

###################### model ######################

model = ECGModel_v1(samples)

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
