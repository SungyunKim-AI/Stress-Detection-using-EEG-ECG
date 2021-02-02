% =============== 전처리 과정 =====================
% 1. Hamming sinc linear phase FIR filters 적용
% 2. DC constant를 처음과 끝에 padding
% 3. shift by the filter’s group delay
% 4. Kothe [31]가 제안한 ASR (Artifact Subspace Reconstruction)
% 5. Cohen [32]에서 권장하는 CAR (Common Average Reference)

filepath = 'C:\Users\user\Desktop\Graduation-Project\DREAMER.mat';
Dataset = load(filepath);
subjects = Dataset.DREAMER.Data;
eeglab;

for i = 1:23
    tmpSub = subjects(1,i);
    subject = tmpSub{1};
    for j = 1:18
        baseline = FIRfiltering(subject.EEG.baseline(j,1));
        stimuli = FIRfiltering(subject.EEG.stimuli(j,1));
        %pop_clean_rawdata()
    end
end

function filtedData = FIRfiltering(input)
    val = input{1};
    overall = fir1(9, [0.0625, 0.46875]);
    theta = fir1(9, [0.0625, 0.125]);
    alpha = fir1(9, [0.125, 0.203125]);
    beta = fir1(9, [0.203125, 0.46875]);
    
    filted = filter(overall, 1, val);
    filtedTheta = filter(theta, 1, filted);
    filtedAlpha = filter(alpha, 1, filted);
    filtedBeta = filter(beta, 1, filted);
    filtedData = [filtedTheta, filtedAlpha, filtedBeta];
end