import os
import numpy as np
from matplotlib import pyplot as plt

# Load ECG Data numpy format
loadPath = "C:/Users/user/Desktop/numpy_dataset/ecg_dataset_256.npz"
data = np.load(loadPath)

x_Train = data['x_Train'][:,2,:]
x_Test = data['x_Test'][:,2,:]
x_Validate = data['x_Validate'][:,2,:]
y_Train = data['y_Train']
y_Test = data['y_Test']
y_Validate = data['y_Validate']

data.close()

# 1D
samples = x_Train.shape[1]
x_Train = x_Train.reshape(x_Train.shape[0], samples, 1)
x_Validate = x_Validate.reshape(x_Validate.shape[0], samples, 1)
x_Test = x_Test.reshape(x_Test.shape[0], samples, 1)

print("Train Set Shape : ", x_Train.shape)          # 
print("Test Set Shape : ", x_Test.shape)            # 
print("Validate Set Shape : ", x_Validate.shape)    # 
print("Train Labels Shape : ", y_Train.shape)       # 
print("Test Labels Shape : ", y_Test.shape)         # 
print("Validate Labels Shape : ", y_Validate.shape) 

print(x_Train[6100].shape)
plt.plot(x_Train[6100])
plt.show()