folder = 'S:\AudioMNIST\eval_MR\data\test38_px2_logitechspeaker\audio';
delete('log.txt')
fid = fopen('log.txt', 'w+');

files = lsfiles(folder, 'wav');

for i_f = 1:length(files)
    [wav, ~] = audioread(fullfile(folder, files{i_f}));
    
    if all(wav == 0)
        fid = fopen('log.txt', 'a+');
        fprintf(fid, [vidname(1:end-4),'.wav\n']);
        fclose(fid);
    end

fclose(fid);