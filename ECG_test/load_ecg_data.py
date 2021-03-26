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
        "C:/Users/kia34/문서/GitHub/Graduation-Project/ECG_test/ECG/baseline/*")
    stimuli_paths = glob.glob(
        "C:/Users/kia34/문서/GitHub/Graduation-Project/ECG_test/ECG/stimuli/*")

    for path in baseline_paths:
        data = pd.read_csv(path, header=None)
        data = data[:2900][0]
        ECG.append(data)
        Labels.append([1, 0])

    for path in stimuli_paths:
        data = pd.read_csv(path, header=None)
        data = data[:2900][0]
        ECG.append(data)
        Labels.append([0, 1])

    ECG = np.array(ECG)
    Labels = np.array(Labels)

    print(ECG.shape)
    print(Labels.shape)

    return ECG, Labels


load_ecg_data()
