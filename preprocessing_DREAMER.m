% =============== 전처리 과정 =====================
% 1. Hamming sinc linear phase FIR filters 적용
% 2. Remove DC offset
% 3. shift by the filter’s group delay
% 4. Kothe [31]가 제안한 ASR (Artifact Subspace Reconstruction) -> function
% 5. Cohen [32]에서 권장하는 CAR (Common Average Reference) -> function X

% 0. Data import
filepath = 'C:\Users\user\Desktop\Graduation-Project\DREAMER.mat';
% filepath = '/Users/kok_ksy/Documents/GitHub/Graduation-Project/DREAMER.mat';
Dataset = load(filepath);
NumOfSub = 23;      % 실험 대상의 수
NumOfClip = 18;     % 비디오 클립 수
SamplingRate = 128;

FilterOrder = 212;      
Filter = {[30 4], [8 4], [13 8], [30 12]};      % 1: overall, 2: theta, 3: alpha, 4: beta
FilterName = {'overall', 'theta', 'alpha', 'beta'};

eeglab;

for sub = 1:NumOfSub
    subject = Dataset.DREAMER.Data(1,sub);
    
    for clip = 1:NumOfClip
        % ===================== baseline =====================
        baseline = subject{1}.EEG.baseline(clip,1);
        baseline = baseline{1}.';
        % 0. Data import
        EEG_baseline = pop_importdata('dataformat','array','nbchan',0,'data','baseline','srate',SamplingRate,'pnts',0,'xmin',0);
        % 1. Hamming sinc linear phase FIR filters
        EEG_baseline = pop_firws(EEG_baseline, 'fcutoff', Filter{1}, 'ftype', 'bandpass', 'wtype', 'hamming', 'forder', FilterOrder, 'minphase', 0, 'usefftfilt', 0, 'plotfresp', 0, 'causal', 0);
        % 2. ASR (Artifact Subspace Reconstruction)
        EEG_baseline = pop_clean_rawdata(EEG_baseline, 'FlatlineCriterion',5,'ChannelCriterion',0.8,'LineNoiseCriterion',4,'Highpass','off','BurstCriterion',20,'WindowCriterion',0.25,'BurstRejection','on','Distance','Euclidian','WindowCriterionTolerances',[-Inf 7] );
        % 3. Bad channels remove
       
        % 4. CAR (Common Average Reference)
        EEG_baseline = pop_reref( EEG_baseline, []);
        
        
        % ===================== stimuli =====================
        stimuli = subject{1}.EEG.stimuli(clip,1);
        stimuli = stimuli{1}.';
        stimuli(:,1:end - 7808) = [];
        % 0. Data import
        EEG_stimuli = pop_importdata('dataformat','array','nbchan',0,'data','stimuli','srate',SamplingRate,'pnts',0,'xmin',0);
        % 1. Hamming sinc linear phase FIR filters
        EEG_stimuli = pop_firws(EEG_stimuli, 'fcutoff', Filter{1}, 'ftype', 'bandpass', 'wtype', 'hamming', 'forder', FilterOrder, 'minphase', 0, 'usefftfilt', 0, 'plotfresp', 0, 'causal', 0);
        % 2. ASR (Artifact Subspace Reconstruction)
        EEG_stimuli = pop_clean_rawdata(EEG_stimuli, 'FlatlineCriterion',5,'ChannelCriterion',0.8,'LineNoiseCriterion',4,'Highpass','off','BurstCriterion',20,'WindowCriterion',0.25,'BurstRejection','on','Distance','Euclidian','WindowCriterionTolerances',[-Inf 7] );
        % 3. Bad channels remove
        
        % 4. CAR (Common Average Reference)
        EEG_stimuli = pop_reref( EEG_stimuli, []);
        
%         EEG = EEG_stimuli.data;
        
%         % divided
%         if EEG_baseline.pnts > EEG_stimuli.pnts
%             arrSize = EEG_stimuli.pnts;
%         else
%             arrSize = EEG_baseline.pnts;
%         end
        avs = EEG_stimuli.data(1:13,1:7111)./EEG_baseline.data(1:13,1:7111);

        
        % save csv file
        %filename = char("C:\\Users\\user\\Desktop\\testdata_\\stimuli" + sub + "_" + clip + ".csv");
        %writematrix(EEG, filename);
    end
end




% for sub = 1:NumOfSub
%     subject = Dataset.DREAMER.Data(1,sub);
%     
%     for clip = 1:NumOfClip
%         % baseline
%         baseline = subject{1}.EEG.baseline(clip,1);
%         baseline = baseline{1}.';
%         EEG = pop_importdata('dataformat','array','nbchan',0,'data','baseline','srate',SamplingRate,'pnts',0,'xmin',0);
%         EEG = pop_firws(EEG, 'fcutoff', Filter{1}, 'ftype', 'bandpass', 'wtype', 'hamming', 'forder', FilterOrder, 'minphase', 0, 'usefftfilt', 0, 'plotfresp', 0, 'causal', 0);
%         
%         saveFilePath = 'C:\\Users\\user\\Desktop\\dataset_filted\\baseline\\';
%         for i = 2:4
%             % 1. Hamming sinc linear phase FIR filters : 4 ~ 30Hz
%             filtedEEG = pop_firws(EEG, 'fcutoff', Filter{i}, 'ftype', 'bandpass', 'wtype', 'hamming', 'forder', FilterOrder, 'minphase', 0, 'usefftfilt', 0, 'plotfresp', 0, 'causal', 0);
%             
%             % 4. Kothe [31]가 제안한 ASR (Artifact Subspace Reconstruction)
%             %EEG = eeg_checkset( EEG );
%             EEG = pop_clean_rawdata(filtedEEG, 'FlatlineCriterion',5,'ChannelCriterion',0.8,'LineNoiseCriterion',4,'Highpass','off','BurstCriterion',20,'WindowCriterion',0.25,'BurstRejection','on','Distance','Euclidian','WindowCriterionTolerances',[-Inf 7] );
%             
%             % save dataset
%             setname = char("baseline" + sub + "_" + clip + "_" + FilterName{i});
%             [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 0,'setname',setname,'gui','off');
%             EEG = eeg_checkset( EEG );
%             EEG = pop_saveset( EEG, 'filename',strcat(setname, '.set'),'filepath', saveFilePath);
%             [ALLEEG EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);
%         end
%        
%         % stimuli
%         stimuli = subject{1}.EEG.stimuli(clip,1);
%         stimuli = stimuli{1}.';
%         stimuli(:,1:end - 7808) = [];
%         EEG = pop_importdata('dataformat','array','nbchan',0,'data','stimuli','srate',SamplingRate,'pnts',0,'xmin',0);
%         EEG = pop_firws(EEG, 'fcutoff', Filter{1}, 'ftype', 'bandpass', 'wtype', 'hamming', 'forder', FilterOrder, 'minphase', 0, 'usefftfilt', 0, 'plotfresp', 0, 'causal', 0);
%         
%         saveFilePath = 'C:\\Users\\user\\Desktop\\dataset_filted\\stimuli\\';
%         for i = 2:4
%             % 1. Hamming sinc linear phase FIR filters : 4 ~ 30Hz
%             filtedEEG = pop_firws(EEG, 'fcutoff', Filter{i}, 'ftype', 'bandpass', 'wtype', 'hamming', 'forder', FilterOrder, 'minphase', 0, 'usefftfilt', 0, 'plotfresp', 0, 'causal', 0);
%             %EEG = eeg_checkset( EEG );
%             EEG = pop_clean_rawdata(EEG, 'FlatlineCriterion',5,'ChannelCriterion',0.8,'LineNoiseCriterion',4,'Highpass','off','BurstCriterion',20,'WindowCriterion',0.25,'BurstRejection','on','Distance','Euclidian','WindowCriterionTolerances',[-Inf 7] );
%             
%             % save dataset
%             setname = char("stimuli" + sub + "_" + clip + "_" + FilterName{i});
%             [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 0,'setname',setname,'gui','off');
%             EEG = eeg_checkset( EEG );
%             EEG = pop_saveset( EEG, 'filename',strcat(setname, '.set'),'filepath', saveFilePath);
%             [ALLEEG EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);
%         end
%         
%     end
% end