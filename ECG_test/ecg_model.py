import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers.experimental.preprocessing import Rescaling


def generate_model():

    # 모델 구성하기
    make_model = Sequential([
        Input(shape=(2900, 1), name='input_layer'),
        Conv1D(
            filters=15,
            kernel_size=10,
            strides=1,
            padding="same",
            activation='relu',
            name='conv_layer1'
        ),
        MaxPooling1D(
            pool_size=2
        ),
        Dropout(0.1),
        Conv1D(
            filters=30,
            kernel_size=10,
            strides=1,
            padding="same",
            activation='relu',
            name='conv_layer2'
        ),
        MaxPooling1D(
            pool_size=2
        ),
        Dropout(0.1),
        Conv1D(
            filters=30,
            kernel_size=10,
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
            kernel_size=3,
            strides=1,
            padding="same",
            activation='relu',
            name='conv_layer4'
        ),
        MaxPooling1D(
            pool_size=2
        ),
        Dropout(0.1),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(2, activation='softmax', name='output_layer')
    ])

    return make_model


model = generate_model()
model.summary()