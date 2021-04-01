import numpy as np
import tensorflow as tf

# mne imports
import mne
from mne import io

# tools for plotting confusion matrices
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from utils import plot_loss_curve, plot_acc_curve, normalization
from load_eeg_data import load_eeg_data

# Import Models
from model_EEGNet import EEGNet
from model_DeepConvNet import DeepConvNet


# Load ECG Data
EEG, Labels, numOfBaseline, numOfStimuli, samples = load_eeg_data()

kernels, chans = 1, 13

# Train : Validate : Test = 7 : 1.5 : 1.5
X_train, X_validate, Y_train, Y_validate = train_test_split(
    EEG, Labels, test_size=0.3, random_state=42)
X_validate, X_test, Y_validate, Y_test = train_test_split(
    X_validate, Y_validate, test_size=0.5, random_state=42)

X_train = X_train.reshape(X_train.shape[0], chans, samples, kernels)
X_validate = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
X_test = X_test.reshape(X_test.shape[0], chans, samples, kernels)

print("Train Set Shape : ", X_train.shape)
print("Validate Set Shape : ", X_validate.shape)
print("Test Set Shape : ", X_test.shape)


###################### model ######################

# model = EEGNet(nb_classes = 2, Chans = chans, Samples = samples, 
#                dropoutRate = 0.5, kernLength = 32, F1 = 8, D = 2, F2 = 16, 
#                dropoutType = 'Dropout')

model = DeepConvNet(nb_classes = 2, Chans = chans, Samples = samples, dropoutRate = 0.5)

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

fit_model = model.fit(
    X_train,
    Y_train,
    epochs=300,
    batch_size=16,
    validation_data=(X_validate, Y_validate)
)

# make prediction on test set.
probs = model.predict(X_test)
preds = probs.argmax(axis = -1)  
acc   = np.mean(preds == Y_test.argmax(axis=-1))
print("Classification accuracy: %f " % (acc))


plot_loss_curve(fit_model.history)
plot_acc_curve(fit_model.history)
