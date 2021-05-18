from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling1D


# https://github.com/Apollo1840/deepECG
def DeepECGModel(input_dim, output_dim=2, dropout=0.5):
    model = Sequential([
        Input(shape=(input_dim, 1), name='input_layer'),
        Conv1D(filters=128, kernel_size=55, activation='relu', input_shape=(input_dim, 1)),
        MaxPooling1D(pool_size=10),
        Dropout(dropout),
        Conv1D(128, 25, activation='relu'),
        MaxPooling1D(5),
        Dropout(dropout),
        Conv1D(128, 10, activation='relu'),
        MaxPooling1D(5),
        Dropout(dropout),
        Conv1D(128, 5, activation='relu'),
        GlobalAveragePooling1D(),
        # Flatten(),
        Dense(256, kernel_initializer='normal', activation='relu'),
        Dropout(dropout),
        Dense(128, kernel_initializer='normal', activation='relu'),
        Dropout(dropout),
        Dense(64, kernel_initializer='normal', activation='relu'),
        Dropout(dropout),
        Dense(output_dim, kernel_initializer='normal', activation='softmax')
    ])

    return model

