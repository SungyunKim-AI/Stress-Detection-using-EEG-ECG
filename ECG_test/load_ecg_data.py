import numpy as np
import pandas as pd
import glob

def load_ecg_data():
    print("ECG Data Loading...")

    Fs = 512    # Sample Frequency (Hz)
    Ss = 20     # Sample second (sec)
    step = 3    # Overlapping Step (sec)
    cptTime = 62

    # Dataset Path
    baseline_paths = glob.glob("C:/Users/user/Desktop/data_preprocessed/ECG_preprocessed/normalized_data/baseline/*")
    stimuli_paths = glob.glob("C:/Users/user/Desktop/data_preprocessed/ECG_preprocessed/normalized_data/stimuli/*")

    ECG = []
    Labels = []

    for category in [baseline_paths, stimuli_paths]:
        for path in category:
            data = pd.read_csv(path, header=None)
            
            # data overlapping
            
            for i in range(0, cptTime - Ss, step):
                part_data = data[i*Fs : (i+Ss)*Fs][2]
                ECG.append(part_data)

                # Labels one-hot encoding
                if category == baseline_paths:
                    Labels.append([1, 0])       
                elif category == stimuli_paths:
                    Labels.append([0, 1])
        
    ECG = np.array(ECG)
    Labels = np.array(Labels)

    print(ECG.shape)
    print(Labels.shape)

    return ECG, Labels


load_ecg_data()
