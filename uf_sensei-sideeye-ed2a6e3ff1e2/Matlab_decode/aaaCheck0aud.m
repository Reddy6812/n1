folder = 'S:\AudioMNIST\test41_scenarios\test_logitechspeaker_full\audio\a';
delete('log.txt')
fid = fopen('log.txt', 'w+');

files = lsfiles(folder, 'wav');

progressbar()

for i_f = 1:length(files)
    [wav, ~] = audioread(fullfile(folder, files{i_f}));
    
    if all(wav == 0)
        fprintf(fid, replace(files{i_f}, '.wav', '\n'));
        delete(fullfile(folder, files{i_f}))
        disp(files{i_f})
    end
    
    progressbar(i_f/length(files))
    
end

fclose(fid);