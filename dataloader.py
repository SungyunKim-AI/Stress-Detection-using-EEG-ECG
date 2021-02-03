import numpy as np
import scipy.io as sio
import tensorflow as tf

path=u'DREAMER.mat'
data=sio.loadmat(path)
    
# 23명의 실험참여자가 18개의 비디오 클립을 시청하고, 128Hz로 60초간 14개의 채널로 측정한 RAW 데이터
numOfSubjects = 23      # 실험 참여자 수
numOfVideo = 18         # 시청한 비디오 클립 수
numOfEEGChannel = 14    # EEG 측정 채널 수
numOfECGChannel = 2     # ECG 측정 채널 수
numOfCategory = 3       # 감성 분류 카테고리 수 (Valence, Arousal, Dominance)

EEG_baseline = []
EEG_stimuli = []
ECG_baseline = []
ECG_stimuli = []
Labels = np.zeros((numOfCategory, numOfSubjects, numOfVideo))

for iter_subject in range(0, numOfSubjects):
    for iter_video in range(0, numOfVideo):
        for iter_ch in range(0, numOfEEGChannel):
            baseline_e = data['DREAMER'][0,0]['Data'][0,iter_subject]['EEG'][0,0]['baseline'][0,0][iter_video,0][:,iter_ch]
            stimuli_e  = data['DREAMER'][0,0]['Data'][0,iter_subject]['EEG'][0,0]['stimuli'][0,0][iter_video,0][-7808:,iter_ch]
        EEG_baseline.append(baseline_e)
        EEG_stimuli.append(stimuli_e)

        for iter_ch in range(0, numOfECGChannel):
            baseline_c = data['DREAMER'][0,0]['Data'][0,iter_subject]['ECG'][0,0]['baseline'][0,0][iter_video,0][:,iter_ch]
            stimuli_c = data['DREAMER'][0,0]['Data'][0,iter_subject]['ECG'][0,0]['stimuli'][0,0][iter_video,0][-15616:,iter_ch] 
        ECG_baseline.append(baseline_c)
        ECG_stimuli.append(stimuli_c)

        Labels[0, iter_subject, iter_video] = data['DREAMER'][0,0]['Data'][0,iter_subject]['ScoreValence'][0,0][iter_video, 0]
        Labels[1, iter_subject, iter_video] = data['DREAMER'][0,0]['Data'][0,iter_subject]['ScoreArousal'][0,0][iter_video,0]
        Labels[2, iter_subject, iter_video] = data['DREAMER'][0,0]['Data'][0,iter_subject]['ScoreDominance'][0,0][iter_video,0]

EEG_baseline = np.asarray(EEG_baseline)
EEG_stimuli = np.asarray(EEG_stimuli)
ECG_baseline = np.asarray(ECG_baseline)
ECG_stimuli = np.asarray(ECG_stimuli)

# Tensorflow 배열 타입으로 변형
EEG_baseline = tf.convert_to_tensor(EEG_baseline, dtype=tf.float32)
EEG_stimuli = tf.convert_to_tensor(EEG_stimuli, dtype=tf.float32)
ECG_baseline = tf.convert_to_tensor(ECG_baseline, dtype=tf.float32)
ECG_stimuli = tf.convert_to_tensor(ECG_stimuli, dtype=tf.float32)

print(EEG_baseline.shape)
print(EEG_stimuli.shape)
print(ECG_baseline.shape)
print(ECG_stimuli.shape)