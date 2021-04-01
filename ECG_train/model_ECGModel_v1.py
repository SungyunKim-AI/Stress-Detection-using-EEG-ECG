import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dense, Flatten, Dropout, BatchNormalization
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


# model = ECGModel_v1()
# model.summary()
