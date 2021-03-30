import numpy as np
import pandas as pd
import glob
from tqdm import tqdm

def load_eeg_data():
    print("EEG Data Loading...")

    Fs = 128    # Sample Frequency (Hz)
    Ss = 40     # Sample second (sec)
    step = 2    # Overlapping Step (sec)

    # Dataset Path
    baseline_paths = glob.glob("C:/Users/user/Desktop/data_preprocessed/ASR_CAR_preprocessed/EEG/baseline/alpha/*")
    stimuli_paths = glob.glob("C:/Users/user/Desktop/data_preprocessed/ASR_CAR_preprocessed/EEG/stimuli/alpha/*")

    EEG = []
    Labels = []

    numOfBaseline = 0
    numOfStimuli = 0
    for category in [baseline_paths, stimuli_paths]:
        for path in tqdm(category):
            data = pd.read_csv(path, header=None)

            # data overlapping
            for i in range(0, int(data.shape[1]/Fs) - Ss, step):
                part_data = data[:][i*Fs : (i+Ss)*Fs]
                EEG.append(part_data)

                # Labels one-hot encoding
                if category == baseline_paths:
                    Labels.append([1, 0])     
                    numOfBaseline += 1  
                elif category == stimuli_paths:
                    Labels.append([0, 1])
                    numOfStimuli += 1
        
    EEG = np.array(EEG)
    Labels = np.array(Labels)

    print("EEG Data Shape : ", EEG.shape)       # (2952, )
    print("Labels Shape : ", Labels.shape)      # (2952, 2)
    print("numOfBaseline : ", numOfBaseline)    # 1795
    print("numOfStimuli : ", numOfStimuli)      # 1157

    return EEG, Labels, numOfBaseline, numOfStimuli

# [EEG, Labels, numOfBaseline, numOfStimuli] = load_eeg_data()
