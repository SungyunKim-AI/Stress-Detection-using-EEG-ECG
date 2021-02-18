% =============== 전처리 과정 =====================
% 1. Hamming sinc linear phase FIR filters 적용
% 2. DC constant를 처음과 끝에 padding
% 3. shift by the filter’s group delay
% 4. Kothe [31]가 제안한 ASR (Artifact Subspace Reconstruction)
% 5. Cohen [32]에서 권장하는 CAR (Common Average Reference)

filepath = 'C:\Users\user\Desktop\Graduation-Project\DREAMER.mat';
% filepath = '/Users/kok_ksy/Documents/GitHub/Graduation-Project/DREAMER.mat';
Dataset = load(filepath);
NumOfSub = 23;      % 실험 대상의 수
NumOfClip = 18;     % 비디오 클립 수
SamplingRate = 128;

FilterOrder = 212;
overallFilter = [30 4];
thetaFilter = [8 4];
alphaFilter = [13 8];
betaFilter = [30 12];

eeglab;

% 0. Data import
for sub = 1:NumOfSub
    subject = Dataset.DREAMER.Data(1,sub);
    
    for clip = 1:NumOfClip
        baseline = subject{1}.EEG.baseline(clip,1);
        baseline = baseline{1}.';
        EEG = pop_importdata('dataformat','array','nbchan',0,'data','baseline','srate',SamplingRate,'pnts',0,'xmin',0);
        EEG = pop_firws(EEG, 'fcutoff', overallFilter, 'ftype', 'bandpass', 'wtype', 'hamming', 'forder', FilterOrder, 'minphase', 0, 'usefftfilt', 0, 'plotfresp', 0, 'causal', 0);
        setname = char("baseline" + sub + "_" + clip);
        [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 0,'setname',setname,'gui','off');
        
        stimuli = subject{1}.EEG.stimuli(clip,1);
        stimuli = stimuli{1}.';
        stimuli(:,1:end - 7808) = [];
        EEG = pop_importdata('dataformat','array','nbchan',0,'data','stimuli','srate',SamplingRate,'pnts',0,'xmin',0);
        EEG = pop_firws(EEG, 'fcutoff', overallFilter, 'ftype', 'bandpass', 'wtype', 'hamming', 'forder', FilterOrder, 'minphase', 0, 'usefftfilt', 0, 'plotfresp', 0, 'causal', 0);
        setname = char("stimuli" + sub + "_" + clip);
        [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 0,'setname',setname,'gui','off');
    end
end

% 데이터셋 저장
EEG = eeg_checkset( EEG );
saveFilePath = 'C:\\Users\\user\\Desktop\\김성윤_졸프\\eeglab2020_0\\sample_data\\';
EEG = pop_saveset( EEG, 'filename','test.set','filepath', saveFilePath);
[ALLEEG EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);



% 1. Hamming sinc linear phase FIR filters : 4 ~ 30Hz
% EEG = pop_firws(EEG, 'fcutoff', [30 4], 'ftype', 'bandpass', 'wtype', 'hamming', 'forder', 48, 'minphase', 0, 'usefftfilt', 0, 'plotfresp', 0, 'causal', 0);
% [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1,'setname','filted_baseline','gui','off'); 
% EEG = pop_firws(EEG, 'fcutoff', [8 4], 'ftype', 'bandpass', 'wtype', 'hamming', 'forder', 48, 'minphase', 0, 'usefftfilt', 0, 'plotfresp', 0, 'causal', 0);
% [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 2,'setname','filtedtheta_baseline','gui','off'); 
% [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 3,'retrieve',2,'study',0); 
% EEG = pop_firws(EEG, 'fcutoff', [13 8], 'ftype', 'bandpass', 'wtype', 'hamming', 'forder', 48, 'minphase', 0, 'usefftfilt', 0, 'plotfresp', 0, 'causal', 0);
% [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 2,'setname','filtedalpha_baseline','gui','off'); 
% [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 4,'retrieve',2,'study',0); 
% EEG = pop_firws(EEG, 'fcutoff', [30 12], 'ftype', 'bandpass', 'wtype', 'hamming', 'forder', 48, 'minphase', 0, 'usefftfilt', 0, 'plotfresp', 0, 'causal', 0);
% [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 2,'setname','filtedbeta_baseline','gui','off'); 

% plot(ALLEEG(2).data)


% for i = 1:2
%     tmpSub = subjects(1,i);
%     subject = tmpSub{1};
%     for j = 1:18
%         baseline = FIRfiltering(subject.EEG.baseline(j,1));
%         stimuli = FIRfiltering(subject.EEG.stimuli(j,1));
%     end
% end
% 
% function filtedData = FIRfiltering(input)
%     val = input{1};
%     overall = fir1(9, [0.0625, 0.46875]);
%     theta = fir1(9, [0.0625, 0.125]);
%     alpha = fir1(9, [0.125, 0.203125]);
%     beta = fir1(9, [0.203125, 0.46875]);
%     
%     filted = filter(overall, 1, val);
%     filtedTheta = filter(theta, 1, filted);
%     plot(filtedTheta);
%     filtedAlpha = filter(alpha, 1, filted);
%     plot(filtedAlpha);
%     filtedBeta = filter(beta, 1, filted);
%     plot(filtedBeta);
%     filtedData = [filtedTheta, filtedAlpha, filtedBeta];
% end