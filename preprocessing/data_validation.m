load_path_EEG = "C:\\Users\\user\\Desktop\\Experiment Data\\EEG\\";
load_path_ECG = "C:\\Users\\user\\Desktop\\Experiment Data\\ECG\\";

for sub = 6:10
    for sample = 1:10
        fprintf('========= Sample %d of Subject %d =========\n',sample,sub);
        file_path = char(load_path_EEG + "s" + sub + "_" + sample + ".csv");
        dataTable = readtable(file_path,"VariableNamingRule","preserve");
        data = dataTable{:,[1,6]};
        
        [row, col] = size(data);
        totalTime = (data(row, 1) - data(1,1));
        hz = row / (totalTime * 128) * 100;
        
        disp(hz);
    end
end