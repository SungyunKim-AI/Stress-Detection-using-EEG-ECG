import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import signal
from sklearn import preprocessing as pre

# python에서 EEGLAB을 사용하기 위해서는 Octave가 필요하다 (함수 정리 블로그 : https://gorakgarak.tistory.com/448)

# EEG 신호는 여러 artifact, noise로 인해 오염되므로, denoising 이 필요하다.
# ECG 신호는 상대적으로 높은 전압 진폭으로 측정하므로 따로 처리하지 않는다.
# EEG 신호에서 ocular artifacts(안구 인공물)은 4Hz 미만, 근육 운동은 30Hz 이상, 전력선 노이즈는 50 or 60Hz에서 나타난다.
# affect recognition task(영향 인식 작업)과 관련된 정보를 포함하는 주파수는 4~30Hz 범위에 있다.
# theta : 4 ~ 8Hz, alpha : 8 ~ 13Hz, beta : 12 ~ 30Hz (12~25Hz = 저베타파, 25~30Hz = 고베타파)

# EEGLAB의 함수들을 이용해 아래의 Pre-processing을 구현할 수 있다. (Documents : https://eeglab.org/tutorials/)
# 관심 범위 내의 주파수만 추출하기 위해서 "Three separate bandpass Hamming sinc linear phase FIR filters"를 적용한다.
# DC offsets의 형태로 EEG 데이터의 시작 또는 끝에 도입되었을 수 있는 artifact는 resampling 하기 전에 DC constant를 처음과 끝에 padding 함으로써 해결한다.
# Finally, the filtered EEG data are shifted by the filter’s group delay.
# 참고로 EPOC EEG 측정기에는 50 or 60 Hz 노이즈는 자동 필터된다. 하지만 필터링 프로세스는 모든 artifact를 제거 할 수는 없기 때문에 추가적인 처리가 필요하다.
# Kothe [31]가 제안한 ASR (Artifact Subspace Reconstruction) 방법은 인공물 제거에 사용됩니다. 
# ASR 방법은 슬라이딩 윈도우 PCA (Principal Component Analysis)로 구성되며, 상대적으로 인공물이없는 자동 감지 된 데이터 섹션의 공분산에 대해 임계 값을 초과하는 고 분산 신호 구성 요소를 통계적으로 보간합니다. 
# 추가 분석을 위해 EEG 데이터를 준비하는 마지막 단계는 모든 전극의 평균 값을 계산하고, 각 전극의 각 샘플에서 빼는 Cohen [32]에서 권장하는 CAR (Common Average Reference) 방법을 적용하는 것입니다.
# 불량 채널의 제거는 가능한 불량 채널로 인해 모든 채널에 노이즈가 유입되는 것을 방지하기 위해 CAR 방법 이전에 수행됩니다.

###### 정리 #######
# 1. Hamming sinc linear phase FIR filters 적용
# 2. DC constant를 처음과 끝에 padding
# 3. shift by the filter’s group delay
# 4. Kothe [31]가 제안한 ASR (Artifact Subspace Reconstruction)
# 5. Cohen [32]에서 권장하는 CAR (Common Average Reference)

def preprocessing(input, feature):
    # theta, alpha, beta hamming filter 생성 : signal.firwin(numtaps, cutoff, window)
    # numtaps = 필터의 길이
    # cutoff = 필터의 차단 주파수
    # window = 필터링에 사용할 window
    overall = signal.firwin(9, [0.0625, 0.46875],   window = 'hamming')
    theta   = signal.firwin(9, [0.0625, 0.125],     window = 'hamming')
    alpha   = signal.firwin(9, [0.125, 0.203125],   window = 'hamming')
    beta    = signal.firwin(9, [0.203125, 0.46875], window = 'hamming')

    # Data filtering : signal.filtfilt(b, a, x)
    # b = 필터의 분자 coefficient vector
    # a = 필터의 분모 coefficient vector
    # x = 필터링할 데이터 배열
    filtedData  = signal.filtfilt(overall, 1, input)
    filtedtheta = signal.filtfilt(theta,   1, filtedData)
    filtedalpha = signal.filtfilt(alpha,   1, filtedData)
    filtedbeta  = signal.filtfilt(beta,    1, filtedData)

    ftheta,psdtheta = signal.welch(filtedtheta,nperseg=256)
    falpha,psdalpha = signal.welch(filtedalpha,nperseg=256)
    fbeta,psdbeta = signal.welch(filtedbeta,nperseg=256)
    feature.append(max(psdtheta))
    feature.append(max(psdalpha))
    feature.append(max(psdbeta))
    return feature


if __name__ == '__main__':
    total=0
    path=u'DREAMER.mat'
    data=sio.loadmat(path)
    print("EEG signals are being feature extracted...")
    EEG_tmp=np.zeros((23,18,42))
    for k in range(0,23):
        for j in range(0,18):
            for i in range(0,14):
                B,S=[],[]
                baseline=data['DREAMER'][0,0]['Data'][0,k]['EEG'][0,0]['baseline'][0,0][j,0][:,i]
                stimuli=data['DREAMER'][0,0]['Data'][0,k]['EEG'][0,0]['stimuli'][0,0][j,0][:,i]
                B=preprocessing(baseline,B)
                S=preprocessing(stimuli,S)
                Extrod=np.divide(S,B)   # np.divide(S, B) == S / B
                total+=1
                EEG_tmp[k,j,3*i]=Extrod[0]      #theta
                EEG_tmp[k,j,3*i+1]=Extrod[1]    #alpha
                EEG_tmp[k,j,3*i+2]=Extrod[2]    #betha
                print("\rprogress: %d%%" %(total/(23*18*14)*100),end="")
    col=[]
    for i in range(0,14):
        col.append('psdtheta_'+str(i + 1)+'_un')
        col.append('psdalpha_'+str(i + 1)+'_un')
        col.append('psdbeta_'+str(i + 1)+'_un')
    EEG=pd.DataFrame(EEG_tmp.reshape((23 * 18,EEG_tmp.shape[2])),columns=col)
    scaler=pre.StandardScaler()
    for i in range(len(col)):
        EEG[col[i][:-3]]=scaler.fit_transform(EEG[[col[i]]])
    EEG.drop(col,axis=1,inplace=True)
    print(EEG)
    EEG.to_csv('Extracted_EEG.csv')
