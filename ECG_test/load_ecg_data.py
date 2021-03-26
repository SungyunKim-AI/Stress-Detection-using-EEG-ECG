import numpy as np
import pandas as pd
import glob
from matplotlib import pyplot as plt


# accuracy 그래프 그리기
def plot(data):
    plt.figure()
    plt.ylabel('signal')
    plt.xlabel('time')
    plt.plot(data)
    plt.show()



def load_ecg_data():
    print("ECG Data Loading...")

    numOfData = 40

    Labels = []
    ECG = []

    # 데이터 경로 설정
    baseline_paths = glob.glob(
        "C:/Users/user/Desktop/data_preprocessed/ECG_preprocessed/normalized_data/baseline/*")
    stimuli_paths = glob.glob(
        "C:/Users/user/Desktop/data_preprocessed/ECG_preprocessed/normalized_data/stimuli/*")

    for path in baseline_paths:
        data = pd.read_csv(path, header=None)
        for i in range(0,12):
            part_data = data[i*512:10240+i*512][0]
            ECG.append(part_data)
            Labels.append([1, 0])
        

    for path in stimuli_paths:
        data = pd.read_csv(path, header=None)
        for i in range(0,12):
            part_data = data[i*512:10240+i*512][0]
            ECG.append(part_data)
            Labels.append([0, 1])

    ECG = np.array(ECG)
    Labels = np.array(Labels)

    print(ECG.shape)
    print(Labels.shape)

    return ECG, Labels


load_ecg_data()
