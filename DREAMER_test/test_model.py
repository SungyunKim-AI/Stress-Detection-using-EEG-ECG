import numpy as np

# mne imports
import mne
from mne import io

# EEGNet-specific imports
from EEGModels import EEGNet
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K

# PyRiemann imports
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.viz import plot_confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

# tools for plotting confusion matrices
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split    # data set split
# data load from dataloader.py
from test_dataloader import test_dataloader


# loss 그래프 그리기
def plot_loss_curve(history):

    plt.figure(figsize=(15, 10))

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()
   
# accuracy 그래프 그리기
def plot_acc_curve(history):
    plt.figure(figsize=(15, 10))
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()


# Data Loading
EEG_stimuli, Labels = test_dataloader()


X = EEG_stimuli
Y = Labels[0]   # Labels : 0 = Valence, 1 = Arousal, 2 = Dominance

print(X.shape)
print(Y.shape)


# split the data to train/validate/test -> 60 : 20 : 20 (248 : 83 : 83)
X_train, X_validate, Y_train, Y_validate = train_test_split(
    X, Y, test_size=0.4, random_state=42)
X_validate, X_test, Y_validate, Y_test = train_test_split(
    X_validate, Y_validate, test_size=0.5, random_state=42)


############################# EEGNet portion ##################################

# convert labels to one-hot encodings.
Y_train = np_utils.to_categorical(Y_train, num_classes=2)
Y_validate = np_utils.to_categorical(Y_validate, num_classes=2)
Y_test = np_utils.to_categorical(Y_test, num_classes=2)


# convert data to NHWC (trials, channels, samples, kernels) format. Data
# contains 60 channels and 151 time-points. Set the number of kernels to 1.
kernels, chans, samples = 1, 14, 8064
X_train = X_train.reshape(X_train.shape[0], chans, samples, kernels)
X_validate = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
X_test = X_test.reshape(X_test.shape[0], chans, samples, kernels)

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# EEGNet model init
model = EEGNet(nb_classes=2, Chans=chans, Samples=samples,
               dropoutRate=0.5, kernLength=64, F1=6, D=2, F2=12,
               dropoutType='Dropout')

# compile the model and set the optimizers
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# count number of parameters in the model
numParams = model.count_params()

# set a valid path for your system to record model checkpoints
checkpointer = ModelCheckpoint(
    filepath='/tmp/checkpoint.h5', verbose=1, save_best_only=True)

###############################################################################
# if the classification task was imbalanced (significantly more trials in one
# class versus the others) you can assign a weight to each class during
# optimization to balance it out. This data is approximately balanced so we
# don't need to do this, but is shown here for illustration/completeness.
###############################################################################

# the syntax is {class_1:weight_1, class_2:weight_2,...}. Here just setting
# the weights all to be 1
#class_weights = {0: 1, 1: 1, 2: 1, 3: 1}

################################################################################
# fit the model. Due to very small sample sizes this can get
# pretty noisy run-to-run, but most runs should be comparable to xDAWN +
# Riemannian geometry classification (below)
################################################################################
# fittedModel = model.fit(X_train, Y_train, batch_size=16, epochs=300,
#                         verbose=2, validation_data=(X_validate, Y_validate),
#                         callbacks=[checkpointer], class_weight=class_weights)
fittedModel = model.fit(X_train, Y_train, batch_size=20, epochs=500,
                        verbose=2, validation_data=(X_validate, Y_validate),
                        callbacks=[checkpointer])

# load optimal weights
# model.load_weights('/tmp/checkpoint.h5')

###############################################################################
# can alternatively used the weights provided in the repo. If so it should get
# you 93% accuracy. Change the WEIGHTS_PATH variable to wherever it is on your
# system.
###############################################################################

# WEIGHTS_PATH = /path/to/EEGNet-8-2-weights.h5
# model.load_weights(WEIGHTS_PATH)

###############################################################################
# make prediction on test set.
###############################################################################

probs = model.predict(X_test)
preds = probs.argmax(axis=-1)
acc = np.mean(preds == Y_test.argmax(axis=-1))
print("Classification accuracy: %f " % (acc))


############################# PyRiemann Portion ##############################

# code is taken from PyRiemann's ERP sample script, which is decoding in
# the tangent space with a logistic regression

n_components = 2  # pick some components

# set up sklearn pipeline
clf = make_pipeline(XdawnCovariances(n_components),
                    TangentSpace(metric='riemann'),
                    LogisticRegression())

preds_rg = np.zeros(len(Y_test))

# reshape back to (trials, channels, samples)
X_train = X_train.reshape(X_train.shape[0], chans, samples)
X_test = X_test.reshape(X_test.shape[0], chans, samples)

# train a classifier with xDAWN spatial filtering + Riemannian Geometry (RG)
# labels need to be back in single-column format
clf.fit(X_train, Y_train.argmax(axis=-1))
preds_rg = clf.predict(X_test)

# Printing the results
acc2 = np.mean(preds_rg == Y_test.argmax(axis=-1))
print("Classification accuracy: %f " % (acc2))

# plot the confusion matrices for both classifiers
# names = ['audio left', 'audio right', 'vis left', 'vis right']
# plt.figure(0)
# plot_confusion_matrix(preds, Y_test.argmax(axis=-1), names, title='EEGNet-8,2')

# plt.figure(1)
# plot_confusion_matrix(preds_rg, Y_test.argmax(
#     axis=-1), names, title='xDAWN + RG')

plot_loss_curve(fittedModel.history)
plot_acc_curve(fittedModel.history)



