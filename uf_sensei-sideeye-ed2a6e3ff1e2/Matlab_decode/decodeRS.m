function decodeRS(testrt, testname, set_size, set_num, num_workers, vidsuffix, delmat, ordercorrect, varargin)
folder = fullfile(testrt, testname);  % video folder, results will be in the same folder
Fr = 34000;        % Sample rate of rolling shutter (row-scanning frequency)
audfolder = fullfile(folder, 'audio');
matfolder = fullfile(folder, 'mat');
mkdir(audfolder);
mkdir(matfolder);
% start paralell workers
pl = gcp('nocreate');
if isempty(pl)
    parpool('local',num_workers)
else
    if pl.NumWorkers ~= num_workers
        delete(pl)
        parpool('local',num_workers)
    end
end

p = inputParser();
addOptional(p, 'DMiter', 5);
addOptional(p, 'DMsmooth', 1.0);
addOptional(p, 'DMlevel', 3);
parse(p, varargin{:});
reg_iters = p.Results.DMiter;
reg_smooth = p.Results.DMsmooth;
reg_level = p.Results.DMlevel;


% wordlog = readtable(fullfile(folder, ['wordlog_py_',testname, '.csv']));
wordlog = readtable(fullfile(folder, ['wordlog_py.csv']));

tic
num_samples = size(wordlog, 1);
calc_rows = (set_num-1)*set_size+1 : min(num_samples, (set_num)*set_size);
disp([num2str(calc_rows(1)), '-', num2str(calc_rows(end))])
parfor i_row = calc_rows
    vidname = table2cell(wordlog(i_row,2));
    vidname = replace(vidname{1}, 'mp4', vidsuffix);
    output_aud = fullfile(audfolder, replace(vidname, vidsuffix, 'wav'));
    output_mat = fullfile(matfolder, replace(vidname, vidsuffix, 'mat'));
    if isfile(output_aud)
        fprintf(['Already there: ', [vidname(1:end-4),'.wav\n']])
        continue;
    end
    tmp = table2cell(wordlog(i_row,10));
    need_proc = tmp{1};
    if ~need_proc
        disp(['No Need: ', [vidname(1:end-4),'.wav\n']])
        continue
    else
        try
            fprintf(['Processing: ', [vidname(1:end-4),'.wav\n']])
            vidset_folder = ['set',num2str(floor((i_row-1)/20)+1),'_upload'];
            if strcmp(vidsuffix, 'mat')
                vidset_folder = [vidset_folder, '_EXT_MAT'];
            end
            NG = NG_recover(fullfile(folder, vidset_folder),...
                vidname, [1,inf], [1 0], 'DMiter',reg_iters, 'DMsmooth', reg_smooth, 'DMlevel', reg_level);% consecutive decoding no amp
            if strcmp(vidsuffix, 'mat') && delmat
                delete(fullfile(folder, vidset_folder, vidname))
            end
            sweep_data = [NG.xx, NG.yy];

            if ordercorrect
                sweep_data = reshape(sweep_data, 1080, [], 8);
                sweep_data = reshape(sweep_data(end:-1:1, :, :), [], 8);
            end

            sweep_data = sweep_data - mean(sweep_data);
            parsave(output_mat, sweep_data)

            sweep_data = sweep_data ./ max(abs(sweep_data));
            audiowrite(output_aud, sweep_data, Fr);
        catch
            fid = fopen('log.txt', 'a+');
            fprintf(fid, [vidname(1:end-4),'.wav\n']);
            fclose(fid);
        end
        
    end
end
toc

