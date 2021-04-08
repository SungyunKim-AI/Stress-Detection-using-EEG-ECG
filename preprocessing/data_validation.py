import numpy as np
import pandas as pd
import glob

#load_path = glob.glob("여기에 경로 추가/*")
load_path = glob.glob("C:/Users/user/Desktop/Experiment Data/ECG/*")
Fs = 512
header_names = ['Shimmer_866B_TimestampSync_Unix_CAL', 
                'Shimmer_866B_ECG_EMG_Status1_CAL', 
                'Shimmer_866B_ECG_EMG_Status2_CAL', 
                'Shimmer_866B_ECG_LA-RA_24BIT_CAL', 
                'Shimmer_866B_ECG_LL-LA_24BIT_CAL', 
                'Shimmer_866B_ECG_LL-RA_24BIT_CAL', 
                'Shimmer_866B_ECG_Vx-RL_24BIT_CAL']

for path in load_path:
    print("file : " + path)
    dataset = pd.read_csv(path, delimiter='\t', names=header_names)
    data_frames = pd.DataFrame(dataset)
    timeStamp = np.array(data_frames.values[3:,0])

    totalTime = round((float(timeStamp[-1][:-3]) - float(timeStamp[0][:-3])) * 1000000000, 3)
    obtain = timeStamp.shape[0] / (totalTime * 512) * 100

    print("데이터 취득률 : " + str(obtain) + "%\n")