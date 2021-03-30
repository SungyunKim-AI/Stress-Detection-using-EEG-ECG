% ===================== Preprocessing Experiment Data =====================
% [공통 과정]
% Hamming sinc linear phase FIR filters 적용
% Remove DC offset
% Shift by the filter’s group delay
% [선택 과정]
% 1. CAR (Common Average Reference)
% 2. ASR (Artifact Subspace Reconstruction) + CAR
% 3. CAR + FFT
% 4. ASR + CAR + FFT

noOfSubjects = 5;           % 실험 대상 수
noOfSamples = 10;           % 실험 수
samplingRate_EEG = 128;     % EEG Sampling Rate (Hz)
filter = ["overall", "theta", "alpha", "beta"];
eeglab;
%==========================================================================

% ================ Hamming sinc linear phase FIR filters =================
% load_path_EEG = "C:\\Users\\user\\Desktop\\data_preprocessed\\cutoff_preprocessed\\EEG\\";
% save_path_EEG = "C:\\Users\\user\\Desktop\\data_preprocessed\\band_filter_preprocessed\\EEG\\";
%
% for subject = 6:10
% %     if subject == 2
% %         continue;
% %     end
%     
%     for sample = 1:noOfSamples
%         % => baseline
%         fileName = char(load_path_EEG + "baseline\\s" + subject + "_" + sample + ".csv");
%         baseline = readtable(fileName);
%         baseline = baseline{:,:}.';
%         baseline = detrend(baseline);
%         baseline_EEG = pop_importdata('dataformat','array','nbchan',0,'data','baseline','srate',samplingRate_EEG,'pnts',0,'xmin',0);
%         
%         baseline_EEG = pop_eegfiltnew(baseline_EEG, 'locutoff',3.5,'hicutoff',30,'plotfreqz',0);
%         baseline_theta = pop_eegfiltnew(baseline_EEG, 'locutoff',3.5,'hicutoff',7.5,'plotfreqz',0);
%         baseline_alpha = pop_eegfiltnew(baseline_EEG, 'locutoff',7.5,'hicutoff',13,'plotfreqz',0);
%         baseline_beta = pop_eegfiltnew(baseline_EEG, 'locutoff',14,'hicutoff',30,'plotfreqz',0);
%         
%         % Save csv file
%         filename = char(save_path_EEG + "baseline\\overall\\s" + subject + "_" + sample + ".csv");
%         writematrix(baseline_EEG.data, filename);
%         filename = char(save_path_EEG + "baseline\\theta\\s" + subject + "_" + sample + ".csv");
%         writematrix(baseline_theta.data, filename);
%         filename = char(save_path_EEG + "baseline\\alpha\\s" + subject + "_" + sample + ".csv");
%         writematrix(baseline_alpha.data, filename);
%         filename = char(save_path_EEG + "baseline\\beta\\s" + subject + "_" + sample + ".csv");
%         writematrix(baseline_beta.data, filename);
%      
%         % => stimuli
%         fileName = char(load_path_EEG + "stimuli\\s" + subject + "_" + sample + ".csv");
%         stimuli = readtable(fileName);
%         stimuli = stimuli{:,:}.';
%         stimuli = detrend(stimuli);
%         stimuli_EEG = pop_importdata('dataformat','array','nbchan',0,'data','stimuli','srate',samplingRate_EEG,'pnts',0,'xmin',0);
%         
%         stimuli_EEG = pop_eegfiltnew(stimuli_EEG, 'locutoff',3.5,'hicutoff',30,'plotfreqz',0);
%         stimuli_theta = pop_eegfiltnew(stimuli_EEG, 'locutoff',3.5,'hicutoff',7.5,'plotfreqz',0);
%         stimuli_alpha = pop_eegfiltnew(stimuli_EEG, 'locutoff',7.5,'hicutoff',13,'plotfreqz',0);
%         stimuli_beta = pop_eegfiltnew(stimuli_EEG, 'locutoff',14,'hicutoff',30,'plotfreqz',0);
%         
%         % Save csv file
%         filename = char(save_path_EEG + "stimuli\\overall\\s" + subject + "_" + sample + ".csv");
%         writematrix(stimuli_EEG.data, filename);
%         filename = char(save_path_EEG + "stimuli\\theta\\s" + subject + "_" + sample + ".csv");
%         writematrix(stimuli_theta.data, filename);
%         filename = char(save_path_EEG + "stimuli\\alpha\\s" + subject + "_" + sample + ".csv");
%         writematrix(stimuli_alpha.data, filename);
%         filename = char(save_path_EEG + "stimuli\\beta\\s" + subject + "_" + sample + ".csv");
%         writematrix(stimuli_beta.data, filename);
%         
%     end
% end


% ==================== CAR (Common Average Reference) ====================
% load_path_EEG = "C:\\Users\\user\\Desktop\\data_preprocessed\\band_filter_preprocessed\\EEG\\";
% save_path_EEG = "C:\\Users\\user\\Desktop\\data_preprocessed\\CAR_preprocessed\\EEG\\";
% 
% for subject = 1:noOfSubjects
%     if subject == 2
%         continue;
%     end
%  
%     for sample = 1:noOfSamples
%         for i = 1:4
%             % => baseline
%             fileName = char(load_path_EEG + "baseline\\" + filter(i) + "\\s" + subject + "_" + sample + ".csv");
%             baseline = readtable(fileName);
%             baseline = baseline{:,:};
%             baseline_EEG = pop_importdata('dataformat','array','nbchan',0,'data','baseline','srate',samplingRate_EEG,'pnts',0,'xmin',0);
%             ASR_CAR_baseline = pop_reref(baseline_EEG, []);
%             
%             filename = char(save_path_EEG + "baseline\\" + filter(i) + "\\s" + subject + "_" + sample + ".csv");
%             writematrix(ASR_CAR_baseline.data, filename);
%             
%             % => stimuli
%             fileName = char(load_path_EEG + "stimuli\\" + filter(i) + "\\s" + subject + "_" + sample + ".csv");
%             stimuli = readtable(fileName);
%             stimuli = stimuli{:,:};
%             stimuli_EEG = pop_importdata('dataformat','array','nbchan',0,'data','stimuli','srate',samplingRate_EEG,'pnts',0,'xmin',0);
%             CAR_stimuli = pop_reref(stimuli_EEG, []);
%             
%             filename = char(save_path_EEG + "stimuli\\" + filter(i) + "\\s" + subject + "_" + sample + ".csv");
%             writematrix(CAR_stimuli.data, filename);
%            
%         end 
%     end
% end



% ============= ASR (Artifact Subspace Reconstruction) + CAR =============
load_path_EEG = "C:\\Users\\user\\Desktop\\data_preprocessed\\band_filter_preprocessed\\EEG\\";
save_path_EEG = "C:\\Users\\user\\Desktop\\data_preprocessed\\ASR_CAR_preprocessed\\EEG\\";

for subject = 6:10
%     if subject == 2
%         continue;
%     end
 
    for sample = 1:noOfSamples
        for i = 1:4
            % => baseline
            fileName = char(load_path_EEG + "baseline\\" + filter(i) + "\\s" + subject + "_" + sample + ".csv");
            baseline = readtable(fileName);
            baseline = baseline{:,:};
            baseline_EEG = pop_importdata('dataformat','array','nbchan',0,'data','baseline','srate',samplingRate_EEG,'pnts',0,'xmin',0);
           
            baseline_EEG = pop_clean_rawdata(baseline_EEG, 'FlatlineCriterion',5,'ChannelCriterion',0.8,'LineNoiseCriterion',4,'Highpass','off','BurstCriterion',20,'WindowCriterion','off','BurstRejection','on','Distance','Euclidian');
            ASR_CAR_baseline = pop_reref(baseline_EEG, []);
            
            filename = char(save_path_EEG + "baseline\\" + filter(i) + "\\s" + subject + "_" + sample + ".csv");
            writematrix(ASR_CAR_baseline.data, filename);
            
            % => stimuli
            fileName = char(load_path_EEG + "stimuli\\" + filter(i) + "\\s" + subject + "_" + sample + ".csv");
            stimuli = readtable(fileName);
            stimuli = stimuli{:,:};
            stimuli_EEG = pop_importdata('dataformat','array','nbchan',0,'data','stimuli','srate',samplingRate_EEG,'pnts',0,'xmin',0);
            
            stimuli_EEG = pop_clean_rawdata(stimuli_EEG, 'FlatlineCriterion',5,'ChannelCriterion',0.8,'LineNoiseCriterion',4,'Highpass','off','BurstCriterion',20,'WindowCriterion','off','BurstRejection','on','Distance','Euclidian');
            ASR_CAR_stimuli = pop_reref(stimuli_EEG, []);
            
            filename = char(save_path_EEG + "stimuli\\" + filter(i) + "\\s" + subject + "_" + sample + ".csv");
            writematrix(ASR_CAR_stimuli.data, filename);
           
        end 
    end
end




