from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling1D

def ECGModel_v1(input_dim):
    # 모델 구성하기
    input_main = Input(input_dim)
    block1 = Conv1D(filters=128, kernel_size=50, padding='same', strides=3, activation='relu')(input_main)
    block1 = BatchNormalization()(block1)
    block1 = MaxPooling1D(pool_size=2, strides=3, padding = 'same')(block1)

    block2 = Conv1D(filters=32, kernel_size=7, padding='same', strides=1, activation='relu')(block1)
    block2 = BatchNormalization()(block2)
    block2 = MaxPooling1D(pool_size=2, strides=2, padding = 'same')(block2) 
    block2 = Dropout(0.5)(block2)

    block3 = Conv1D(32, 10, padding='same', activation='relu')(block2) 
    block3 = Conv1D(128, 5, padding='same', activation='relu', strides=2)(block3)
    block3 = MaxPooling1D(pool_size=2, strides=2, padding = 'same')(block3)
    block3 = Dropout(0.5)(block3)
    
    block4 = Conv1D(filters=256, kernel_size=15, padding = 'same', activation = 'relu')(block3)
    block4 = MaxPooling1D(pool_size=2, strides=2, padding = 'same')(block4)
    
    block5 = Conv1D(filters=512, kernel_size=5, padding='same', activation='relu')(block4)
    block5 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(block5)
    block5 = Dropout(0.5)(block5)
    
    flatten = Flatten()(block5)
    dense = Dense(units=512, activation = 'relu')(flatten)
    dropout = Dropout(0.5)(dense)
    softmax = Dense(units=2, activation='softmax')(dropout)

    return Model(inputs=input_main, outputs=softmax)


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
