% =========================== ECG Preprocessing ===========================
% 1. Removew baseline wander
% 2. Bandpass filter (1 ~ 50Hz)
% 3. Detect the R peaks of the QRS complex [24]
%    -  visual inspection was performed to correct the false acceptance and false rejection of R peaks.
% 3. Remove the abnormal RR Interval (RRI) [10]
% 4. NN (normal-to-normal) interval : Zero-one tranformation (R peaks = 1, others = 0)
% 5. Lomb Periodogram (0.04 to 20 Hz was adopted)

Fs = 512;       
load_path_ECG = "C:\\Users\\user\\Desktop\\data_preprocessed\\cutoff_preprocessed\\ECG\\";
save_path_ECG_128 = "C:\\Users\\user\\Desktop\\data_preprocessed\\band_filter_preprocessed\\ECG_128\\";
save_path_ECG_256 = "C:\\Users\\user\\Desktop\\data_preprocessed\\band_filter_preprocessed\\ECG_256\\";

for category = ["baseline", "stimuli"]
    for subject = 5:19
        for sample = 1:10
            file_path = char(load_path_ECG + category + "\s" + subject + "_" + sample + ".csv");
            
            try
                dataTable = readtable(file_path,"VariableNamingRule","preserve");
            catch
                continue;
            end
            
            for ch = 1:3              
                % 1. Remove basline wander
                [C, L] = wavedec (dataTable{:,ch},9,'bior3.7'); % Decomposition
                a9 = wrcoef ('a', C, L,'bior3.7',9); % Approximate Component
                d9 = wrcoef ('d', C, L,'bior3.7',9); % Detailed components
                d8 = wrcoef ('d', C, L,'bior3.7',8);
                d7 = wrcoef ('d', C, L,'bior3.7',7);
                d6 = wrcoef ('d', C, L,'bior3.7',6);
                d5 = wrcoef ('d', C, L,'bior3.7',5);
                d4 = wrcoef ('d', C, L,'bior3.7',4);
                d3 = wrcoef ('d', C, L,'bior3.7',3);
                d2 = wrcoef ('d', C, L,'bior3.7',2);
                d1 = wrcoef ('d', C, L,'bior3.7',1);
                y= d9+d8+d7+d6+d5+d4+d3+d2+d1;
                
                % 2. Bandpass Filter (1~50Hz)
                [b_pass,a_pass] = butter(3,[1,50]/(Fs/2), 'bandpass');
                dataTable{:,ch} = filtfilt(b_pass, a_pass, y);
                
                % filter data and plot it
%                 subplot (3,1,1), plot(b_sample), title ('ECG Signal with baseline wander'), grid on
%                 subplot (3,1,2), plot(y), title ('ECG Signal after baseline wander REMOVED'), grid on
%                 subplot(3,1,3), plot(dataTable{:,ch}), title ('ECG Signal after Bandpass filtering'), grid on
            end
            
            % Downsampling 256Hz
            downData_256 = downsample(dataTable{:,:},  2);
            fileName = char(save_path_ECG_256 + category + "\\s" + subject + "_" + sample + ".csv");
            writematrix(downData_256, fileName);
            
            % Downsampling 128Hz
            downData_128 = downsample(dataTable{:,:},  4);
            fileName = char(save_path_ECG_128 + category + "\\s" + subject + "_" + sample + ".csv");
            writematrix(downData_128, fileName);
        end
    end
end

% % 2. Detect the R peaks of the QRS complex
% % 함수 출처 : https://github.com/danielwedekind/qrsdetector/tree/d0efea0d883ea329b1110d3fa51802458d71f3b1
% [qrs_pos,filt_data,int_data,thF1,thI1] = pantompkins_qrs(data(:,:), Fs);
% 
% % Data Normalize
% filt_data = normalize(filt_data,'range');
% 
% % 4. Zero-one tranformation
% [row, col] = size(qrs_pos);
% qrspeaks = zeros(row, col);
% previousIndex = 1;
% for index = 1:col
%     filt_data(previousIndex:qrs_pos(index)-1) = 0;
%     qrspeaks(index) = filt_data(qrs_pos(index));
%     filt_data(qrs_pos(index)) = 1;
%     previousIndex = qrs_pos(index) + 1;
% end
% filt_data(previousIndex:end) = 0;
% 
% subplot(4,1,3);
% plot(filt_data);
% 
% % 5. Lomb Periodogram (0.04 to 20 Hz was adopted)
% [pxx,f] = plomb(qrspeaks,qrs_pos);
% plot(f,pxx);
% xlabel("Frequency");
% ylabel("Power");
% title("Lomb Periodogram");

