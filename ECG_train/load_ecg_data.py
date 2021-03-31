import numpy as np
import pandas as pd
import glob
from tqdm import tqdm

def load_ecg_data():
    print("ECG Data Loading...")

    Fs = 512    # Sample Frequency (Hz)
    Ss = 30     # Sample second (sec)
    step = 2    # Overlapping Step (sec)
    samples = Fs * Ss

    # Dataset Path
    baseline_paths = glob.glob(
        "C:/Users/user/Desktop/data_preprocessed/ECG_preprocessed/normalized_data/baseline/*")
    stimuli_paths = glob.glob(
        "C:/Users/user/Desktop/data_preprocessed/ECG_preprocessed/normalized_data/stimuli/*")

    ECG = []
    Labels = []

    numOfBaseline = 0
    numOfStimuli = 0
    for category in [baseline_paths, stimuli_paths]:
        for path in tqdm(category):
            dataset = pd.read_csv(path, header=None)
            data_frames = pd.DataFrame(dataset)
            data = np.array(data_frames.values)

            # data overlapping
            for i in range(0, int(data.shape[0]/Fs) - Ss + 1, step):
                part_data = data[(i*Fs) : ((i+Ss)*Fs), :]
                ECG.append(np.transpose(part_data))

                # Labels one-hot encoding
                if category == baseline_paths:
                    numOfBaseline += 1
                    Labels.append([1, 0])
                elif category == stimuli_paths:
                    numOfStimuli += 1
                    Labels.append([0, 1])

    ECG = np.array(ECG)
    Labels = np.array(Labels)

    print("ECG Data Shape : ", ECG.shape)
    print("Labels Shape : ", Labels.shape)
    print("numOfBaseline : ", numOfBaseline)    # 1795
    print("numOfStimuli : ", numOfStimuli)      # 1157

    return ECG, Labels, numOfBaseline, numOfStimuli, samples


#[ECG, Labels, numOfBaseline, numOfStimuli, samples] = load_ecg_data()
