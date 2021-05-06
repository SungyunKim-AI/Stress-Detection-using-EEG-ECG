import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling1D, Activation
from tensorflow.keras.layers.experimental.preprocessing import Rescaling


def ECGModel_v1(samples):
    # 모델 구성하기
    make_model = Sequential([
        Input(shape=(samples, 1), name='input_layer'),
        Conv1D(
            filters=30,
            kernel_size=64,
            strides=1,
            padding="same",
            activation='relu',
            name='conv_layer1'
        ),
        Dropout(0.25),
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
            kernel_size=64,
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
            kernel_size=64,
            strides=1,
            padding="same",
            activation='relu',
            name='conv_layer4'
        ),
        Dropout(0.25),
        MaxPooling1D(
            pool_size=2
        ),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(64, activation='relu'),
        Dense(2, activation='softmax', name='output_layer')
    ])

    return make_model


# https://github.com/Apollo1840/deepECG
# def DeepECGModel(input_dim, output_dim=2):
    model = Sequential([
        Input(shape=(input_dim, 1), name='input_layer'),
        Conv1D(64, 55, activation='relu', input_shape=(input_dim, 1)),
        MaxPooling1D(2),
        Dropout(0.1),
        Conv1D(64, 25, activation='relu'),
        MaxPooling1D(2),
        Dropout(0.1),
        Conv1D(64, 10, activation='relu'),
        MaxPooling1D(2),
        Dropout(0.1),
        Conv1D(64, 5, activation='relu'),
        GlobalAveragePooling1D(),
        Flatten(),
        Dense(256, kernel_initializer='normal', activation='relu'),
        Dropout(0.1),
        Dense(128, kernel_initializer='normal', activation='relu'),
        Dropout(0.1),
        Dense(64, kernel_initializer='normal', activation='relu'),
        Dense(output_dim, kernel_initializer='normal', activation='softmax')
    ])

    return model

def DeepECGModel(input_dim, output_dim=2):
    model = Sequential([
        Input(shape=(input_dim, 1), name='input_layer'),
        Conv1D(64, 55, activation='relu', input_shape=(input_dim, 1)),
        MaxPooling1D(2),
        Dropout(0.1),
        Conv1D(64, 55, activation='relu'),
        MaxPooling1D(2),
        Dropout(0.1),
        Conv1D(64, 55, activation='relu'),
        MaxPooling1D(2),
        Dropout(0.1),
        Flatten(),
        Dense(256, kernel_initializer='normal', activation='relu'),
        Dropout(0.1),
        Dense(128, kernel_initializer='normal', activation='relu'),
        Dropout(0.1),
        Dense(64, kernel_initializer='normal', activation='relu'),
        Dense(output_dim, kernel_initializer='normal', activation='softmax')
    ])

    return model


# health moritoring ~~~ 논문
def HealthMonitoring_Model(input_dim):
    model = Sequential([
        Input(shape=(input_dim, 1), name='input_layer'),
        Conv1D(16, 64, activation='relu', input_shape=(input_dim, 1)),
        MaxPooling1D(2),
        Conv1D(64, 64, activation='relu'),
        MaxPooling1D(2),
        Conv1D(64, 64, activation='relu'),
        MaxPooling1D(2),
        Flatten(),
        Dense(1024, kernel_initializer='normal', activation='sigmoid'),
        Dropout(0.25),
        Dense(128, kernel_initializer='normal', activation='sigmoid'),
        Dropout(0.25),
        Dense(2, kernel_initializer='normal', activation='softmax')
    ])

    return model

def CustomModel(input_dim):
    input_main   = Input((input_dim, 1))

    layer = Conv1D(10, 128, input_shape=(input_dim, 1))(input_main)
    layer = Activation('relu')(layer)
    layer = BatchNormalization()(layer)
    layer = MaxPooling1D(2)(layer)
    layer =Dropout(0.1)(layer)

    layer = Conv1D(15, 128)(layer)
    layer = Activation('relu')(layer)
    layer = BatchNormalization()(layer)
    layer = MaxPooling1D(2)(layer)
    layer = Dropout(0.1)(layer)

    layer = Conv1D(20, 128)(layer)
    layer = Activation('relu')(layer)
    layer = BatchNormalization()(layer)
    layer = MaxPooling1D(2)(layer)
    layer =Dropout(0.1)(layer)

    layer = Flatten()(layer)
    layer = Dense(512, kernel_initializer='normal', activation='relu')(layer)
    layer = Dropout(0.1)(layer)
    layer = Dense(128, kernel_initializer='normal', activation='relu')(layer)
    output = Dense(2, kernel_initializer='normal', activation='softmax')(layer)

    return Model(inputs=input_main, outputs=output)