

from load_ecg_data import load_ecg_data
import numpy as np
import tensorflow as tf

# mne imports
import mne
from mne import io

# EEGNet-specific imports
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv1D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm

# PyRiemann imports
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.viz import plot_confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

# tools for plotting confusion matrices
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

import DWNet1D


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


def normalization(X, hypes):
    print("Normalizing...")
    if hypes == "01":
        x_min = np.amin(X)
        x_max = np.amax(X)
        X = (X - x_min)/(x_max-x_min)
    elif hypes == "-11":
        std = np.std(np.array(X).ravel())
        mean = np.mean(np.array(X).ravel())
        X = (X - mean)/std
    return X


# def testNet(nb_classes, Chans=64, Samples=128,
#             dropoutRate=0.2, kernLength=25, F1=8,
#             D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):
#     if dropoutType == 'SpatialDropout2D':
#         dropoutType = SpatialDropout2D
#     elif dropoutType == 'Dropout':
#         dropoutType = Dropout
#     else:
#         raise ValueError('dropoutType must be one of SpatialDropout2D '
#                          'or Dropout, passed as a string.')

#     input1 = Input(shape=(Chans, Samples, 1))
#     ##################################################################
#     block1 = Conv1D(F1, kernLength, padding='same',
#                     input_shape=(Chans, Samples, 1),
#                     use_bias=False)(input1)
#     block1 = BatchNormalization()(block1)
#     block1 = Conv1D(F1, kernLength, padding='same',
#                     use_bias=False)(block1)
#     block1 = BatchNormalization()(block1)
#     block1 = Activation('relu')(block1)
#     block1 = AveragePooling2D((1, 2))(block1)
#     block1 = dropoutType(dropoutRate)(block1)

#     block2 = Conv1D(F1, kernLength, padding='same',
#                     use_bias=False)(block1)
#     block2 = BatchNormalization()(block2)
#     block2 = Activation('relu')(block2)
#     block2 = AveragePooling2D((1, 2))(block2)
#     block2 = dropoutType(dropoutRate)(block2)

#     flatten = Flatten(name='flatten')(block2)

#     dense = Dense(nb_classes, name='dense',
#                   kernel_constraint=max_norm(norm_rate))(flatten)
#     softmax = Activation('softmax', name='softmax')(dense)

#     return Model(inputs=input1, outputs=softmax)

def testNet(nb_classes, Chans=64, Samples=128,
            dropoutRate=0.2, kernLength=25, F1=8,
            D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):

    input1 = Input(shape=(Chans, Samples, 1))
    first_out = DWNet1D.first_level(input1)
    second_out = DWNet1D.second_level(first_out)
    third_out = DWNet1D.third_level(second_out)
    ##################################################################
    # flatten
    flatten = Flatten()(third_out)

    # Neural Network with Dropout
    nn = Dense(units=20,
               activation="relu", name="nn_layer")(flatten)
    do = Dropout(rate=0.1, name="drop")(nn)
    visual = False
    if not visual:
        # Classification Layer
        final_layer = Dense(
            units=2, activation="sigmoid", name="sigmoid_layer")(do)
    else:
        final_layer = do

    return Model(inputs=input1, outputs=final_layer)


ECG, Labels = load_ecg_data()
ECG = normalization(ECG, "01")

ECG = np.array(ECG)
Labels = np.array(Labels)

X_train, X_validate, Y_train, Y_validate = train_test_split(
    ECG, Labels, test_size=0.3, random_state=42)
X_validate, X_test, Y_validate, Y_test = train_test_split(
    X_validate, Y_validate, test_size=0.5, random_state=42)


kernels, chans, samples = 1, 1, 2900

X_train = X_train.reshape(X_train.shape[0], chans, samples, kernels)
X_validate = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
X_test = X_test.reshape(X_test.shape[0], chans, samples, kernels)

print('X_train shape:', X_train.shape)


with tf.device('GPU:0'):

    model = testNet(nb_classes=2, Chans=chans, Samples=samples)

    # compile the model and set the optimizers
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    fittedModel = model.fit(X_train, Y_train, batch_size=1, epochs=200,
                            verbose=2, validation_data=(X_validate, Y_validate),)

    probs = model.predict(X_test)
    preds = probs.argmax(axis=-1)
    acc = np.mean(preds == Y_test.argmax(axis=-1))
    print("Classification accuracy: %f " % (acc))

    plot_loss_curve(fittedModel.history)
    plot_acc_curve(fittedModel.history)
