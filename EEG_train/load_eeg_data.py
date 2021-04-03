import numpy as np
import pandas as pd
import glob
import os
from tqdm import tqdm

# 전체 세트 오버래핑
def load_eeg_data():
    print("EEG Data Loading...")

    Fs = 128    # Sample Frequency (Hz)
    Ss = 40     # Sample second (sec)
    step = 2    # Overlapping Step (sec)
    samples = Fs * Ss

    # Dataset Path _ ASR + CAR
    baseline_paths = glob.glob("C:/Users/user/Desktop/data_preprocessed/ASR_CAR_preprocessed/EEG/baseline/alpha/*")
    stimuli_paths = glob.glob("C:/Users/user/Desktop/data_preprocessed/ASR_CAR_preprocessed/EEG/stimuli/alpha/*")

    # Dataset Path _ CAR
    # baseline_paths = glob.glob("C:/Users/user/Desktop/data_preprocessed/CAR_preprocessed/EEG/baseline/alpha/*")
    # stimuli_paths = glob.glob("C:/Users/user/Desktop/data_preprocessed/CAR_preprocessed/EEG/stimuli/alpha/*")

    EEG = []
    Labels = []

    numOfBaseline = 0
    numOfStimuli = 0
    for category in [baseline_paths, stimuli_paths]:
        for path in tqdm(category):
            dataset = pd.read_csv(path, header=None)
            data_frames = pd.DataFrame(dataset)
            data = np.array(data_frames.values)

            # data overlapping
            for i in range(0, int(data.shape[1]/Fs) - Ss + 1, step):
                part_data = data[0:13, (i*Fs) : ((i+Ss)*Fs)]
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

    print("EEG Data Shape : ", EEG.shape)       # (3032, )
    print("Labels Shape : ", Labels.shape)      # (3032, 2)
    print("numOfBaseline : ", numOfBaseline)    # 1831
    print("numOfStimuli : ", numOfStimuli)      # 1201
    print("Samples : ", samples)

    return EEG, Labels, numOfBaseline, numOfStimuli, samples


# Train, Test, Validate 미리 나눠서 오버레핑하고 셔플
def load_eeg_data_2():
    print("EEG Data Loading...")

    Subjects = 10
    Fs = 128    # Sample Frequency (Hz)
    Ss = 40     # Sample second (sec)
    step = 2    # Overlapping Step (sec)
    samples = Fs * Ss

    # Dataset Path _ ASR + CAR
    baseline_paths = "C:/Users/user/Desktop/data_preprocessed/ASR_CAR_preprocessed/EEG/baseline/alpha/"
    stimuli_paths = "C:/Users/user/Desktop/data_preprocessed/ASR_CAR_preprocessed/EEG/stimuli/alpha/"

    # Dataset Path _ CAR
    # baseline_paths = glob.glob("C:/Users/user/Desktop/data_preprocessed/CAR_preprocessed/EEG/baseline/alpha/*")
    # stimuli_paths = glob.glob("C:/Users/user/Desktop/data_preprocessed/CAR_preprocessed/EEG/stimuli/alpha/*")

    x_Train = []
    x_Test = []
    x_Validate = []
    y_Train = []
    y_Test =[]
    y_Validate = []

    for category in range(2):
        for subject in tqdm(range(1, Subjects+1)):
            if subject == 2:
                continue

            for sample in range(1, 11):
                path = baseline_paths + "s" + str(subject) + "_" + str(sample) + ".csv"
                dataset = pd.read_csv(path, header=None)
                data_frames = pd.DataFrame(dataset)
                data = np.array(data_frames.values)

                # data overlapping
                for i in range(0, int(data.shape[1]/Fs) - Ss + 1, step):
                    part_data = data[0:13, (i*Fs) : ((i+Ss)*Fs)]
                    
                    if subject == 10:
                        x_Validate.append(part_data)
                        y_Validate.append(label_append(category, subject))
                    elif subject == 9:
                        x_Test.append(part_data)
                        y_Test.append(label_append(category, subject))
                    else:
                        x_Train.append(part_data)
                        y_Train.append(label_append(category, subject))

    [x_Train, y_Train] = data_shuffle(x_Train, y_Train)
    [x_Test, y_Test] = data_shuffle(x_Test, y_Test)
    [x_Validate, y_Validate] = data_shuffle(x_Validate, y_Validate)

    print("Train Data Shape : ", x_Train.shape)         # (2868, 13, 5120)
    print("Test Data Shape : ", x_Test.shape)           # (394, 13, 5120)
    print("Validate Data Shape : ", x_Validate.shape)   # (400, 13, 5120)
    print("Train Labels Shape : ", y_Train.shape)       # (2868, 2)
    print("Test Labels Shape : ", y_Test.shape)         # (394, 2)
    print("Validate Labels Shape : ", y_Validate.shape) # (400, 2)


    return x_Train, x_Test, x_Validate, y_Train, y_Test, y_Validate

def label_append(category, subject):
    # Labels one-hot encoding
    if category == 0:       # 0: baseline
        return [1, 0]
    elif category == 1:     # 1: stimuli
        return [0, 1]

    

def data_shuffle(x, y):
    x = np.array(x)
    y = np.array(y)
    s = np.arange(x.shape[0])
    np.random.shuffle(s)

    x = x[s]
    y = y[s]
    
    return x, y

#[EEG, Labels, numOfBaseline, numOfStimuli, samples] = load_eeg_data()
#[x_Train, x_Test, x_Validate, y_Train, y_Test, y_Validate] = load_eeg_data_2()

