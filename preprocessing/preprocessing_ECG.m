% =========================== ECG Preprocessing ===========================
% 1. Bandpass filter (1 ~ 50Hz)
% 2. Detect the R peaks of the QRS complex [24]
%    -  visual inspection was performed to correct the false acceptance and false rejection of R peaks.
% 3. Remove the abnormal RR Interval (RRI) [10]
% 4. NN (normal-to-normal) interval : Zero-one tranformation (R peaks = 1, others = 0)
% 5. Lomb Periodogram (0.04 to 20 Hz was adopted)

Fs = 512;
load_path_ECG = "C:\\Users\\user\\Desktop\\data_preprocessed\\cutoff_preprocessed\\ECG\\";
save_path_ECG = "C:\\Users\\user\\Desktop\\data_preprocessed\\ECG_preprocessed\\QRS_128\\";

for category = ["baseline", "stimuli"]
    for subject = 5:9
        for sample = 1:10
            file_path = char(load_path_ECG + category + "\s" + subject + "_" + sample + ".csv");
            
            try
                dataTable = readtable(file_path,"VariableNamingRule","preserve");
                data = dataTable{:,2};
                
                % 2. Detect the R peaks of the QRS complex
                % 함수 출처 : https://github.com/danielwedekind/qrsdetector/tree/d0efea0d883ea329b1110d3fa51802458d71f3b1
                [qrs_pos,filt_data,int_data,thF1,thI1] = pantompkins_qrs(data(:,:), Fs);
                
                % Data Normalize
                filt_data = normalize(filt_data,'range');
                
%                 % 4. Zero-one tranformation
%                 [row, col] = size(qrs_pos);
%                 qrspeaks = zeros(row, col);
%                 previousIndex = 1;
%                 for index = 1:col
%                     filt_data(previousIndex:qrs_pos(index)-1) = 0;
%                     qrspeaks(index) = filt_data(qrs_pos(index));
%                     filt_data(qrs_pos(index)) = 1;
%                     previousIndex = qrs_pos(index) + 1;
%                 end
%                 filt_data(previousIndex:end) = 0;
%                 
%                 subplot(4,1,3);
%                 plot(filt_data);
%                 
%                 % 5. Lomb Periodogram (0.04 to 20 Hz was adopted)
%                 [pxx,f] = plomb(qrspeaks,qrs_pos);
%                 plot(f,pxx);
%                 xlabel("Frequency");
%                 ylabel("Power");
%                 title("Lomb Periodogram");
            catch
                continue;
            end
            downData = downsample(filt_data,  4);
            fileName = char(save_path_ECG + category + "\\s" + subject + "_" + sample + ".csv");
            writematrix(downData, fileName);
        end
    end
end

