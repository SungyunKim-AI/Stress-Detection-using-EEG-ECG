import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.layers.experimental.preprocessing import Rescaling


def ECGModel_v1(samples):

    # 모델 구성하기
    make_model = Sequential([
        Input(shape=(samples, 1), name='input_layer'),
        Conv1D(
            filters=30,
            kernel_size=256,
            strides=1,
            padding="same",
            activation='relu',
            name='conv_layer1'
        ),
        Dropout(0.5),
        # Conv1D(
        #     filters=256,
        #     kernel_size=256,
        #     strides=1,
        #     padding="same",
        #     activation='relu',
        #     name='conv_layer2'
        # ),
        # Dropout(0.5),
        Conv1D(
            filters=30,
            kernel_size=256,
            strides=1,
            padding="same",
            activation='relu',
            name='conv_layer3'
        ),
        MaxPooling1D(
            pool_size=2
        ),
        Conv1D(
            filters=30,
            kernel_size=256,
            strides=1,
            padding="same",
            activation='relu',
            name='conv_layer4'
        ),
        Dropout(0.5),
        MaxPooling1D(
            pool_size=4
        ),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(2, activation='softmax', name='output_layer')
    ])

    return make_model


# https://github.com/Apollo1840/deepECG
def DeepECGModel(input_dim, output_dim=2):
    model = Sequential()

    model.add(Conv1D(128, 55, activation='relu', input_shape=(input_dim, 1)))
    model.add(MaxPooling1D(10))
    model.add(Dropout(0.5))

    model.add(Conv1D(128, 25, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Dropout(0.5))

    model.add(Conv1D(128, 10, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Dropout(0.5))

    model.add(Conv1D(128, 5, activation='relu'))
    model.add(GlobalAveragePooling1D())

    # model.add(Flatten())
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(output_dim, kernel_initializer='normal', activation='softmax'))

    # model.compile(loss='categorical_crossentropy',
    #               optimizer='adam', metrics=['accuracy'])

    return model
