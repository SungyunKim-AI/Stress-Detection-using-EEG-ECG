import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.constraints import max_norm



def EEGNet(nb_classes, Chans = 14, Samples = 128, 
             dropoutRate = 0.5, kernLength = 64, F1 = 8, 
             D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout'):

    # Setting Dropout Type
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    # Input
    input1 = Input(shape = (Chans, Samples, 1))

    # Model
    block1       = Conv2D(F1, (1, kernLength), padding = 'same',
                                   input_shape = (Chans, Samples, 1),
                                   use_bias = False)(input1)
    block1       = BatchNormalization()(block1)
    block1       = DepthwiseConv2D((Chans, 1), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1)
    block1       = BatchNormalization()(block1)
    block1       = Activation('relu')(block1)
    block1       = AveragePooling2D((1, 4))(block1)
    block1       = dropoutType(dropoutRate)(block1)
    
    block2       = SeparableConv2D(F2, (1, 16),
                                   use_bias = False, padding = 'same')(block1)
    block2       = BatchNormalization()(block2)
    block2       = Activation('relu')(block2)
    block2       = AveragePooling2D((1, 8))(block2)
    block2       = dropoutType(dropoutRate)(block2)
        
    flatten      = Flatten(name = 'flatten')(block2)
    
    dense        = Dense(nb_classes, name = 'dense', 
                         kernel_constraint = max_norm(norm_rate))(flatten)
    sigmoid      = Activation('sigmoid', name = 'sigmoid')(dense)
    
    return Model(inputs=input1, outputs=sigmoid)


# class EEGLstmNet3fc_82_200(nn.Module):

#     func = 0
#     learnRate = 1e-3
#     batchSize = 300
#     epoch = 5000

#     def __init__(self):
#         super(EEGLstmNet3fc_82_200, self).__init__()
#         # self.T = 120

#         # Layer 1
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(1, 8, (101, 1), padding=(50, 0)),
#             nn.BatchNorm2d(8, False),
#             nn.Sigmoid()
#         )


#         # Layer 2
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(8, 16, (1, 64), groups=8),
#             nn.BatchNorm2d(16, False),
#             nn.Sigmoid(),
#             nn.AvgPool2d((4, 1))
#         )

#         # Layer 3
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(16, 16, (25, 1), padding=(12, 0), groups=8),
#             nn.BatchNorm2d(16, False),
#             nn.Sigmoid(),
#             nn.Conv2d(16, 16, (1, 1), padding=0),
#             nn.BatchNorm2d(16, False),
#             nn.Sigmoid(),
#             nn.AvgPool2d((2, 1))
#         )


#         # LSTM Layer
#         self.rnn = nn.LSTM(
#             input_size=16,
#             hidden_size=16 * 25 * 2,
#             num_layers=1,
#             bidirectional=True,
#             batch_first=True,  # （batch,time_step,input）时是Ture
#         )
#         self.fc = nn.Sequential(
#             nn.BatchNorm1d(16 * 25 * 2 * 2, False),
#             # nn.Dropout(0.15),
#             nn.Linear(16 * 25 * 2 * 2, 800),
#             nn.BatchNorm1d(800, False),
#             nn.Sigmoid(),
#             # nn.Linear(800, 400),
#             # nn.Dropout(0.15),
#             # nn.BatchNorm1d(800, False),
#             # nn.Sigmoid(),
#             nn.Linear(800, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         # Layer 1
#         x = self.conv1(x)
#         x = F.dropout(x, 0.15)

#         # Layer 2
#         x = self.conv2(x)
#         x = F.dropout(x, 0.15)

#         # Layer 3
#         x = self.conv3(x)
#         x = F.dropout(x, 0.15)
#         #
#         # # LSTM Layer
#         x = x.view(-1, 16, 25)
#         x = x.permute(0, 2, 1)
#         x, (h_n, h_c) = self.rnn(x, None)
#         x = self.fc(x[:, -1, :])
#         return x