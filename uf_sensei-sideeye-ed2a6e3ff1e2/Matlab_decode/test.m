%%

clear;
basefolder = 'S:\AudioMNIST\test41_scenarios\test38_px2_bag_pocket\set1_upload';  % video vidfolder, results will be in the same vidfolder
testcase_folders = {''}; % lsfiles(basefolder, ''); %  ;
Fr = 34000;        % Sample rate of rolling shutter (row-scanning frequency)
% you need to do a sweep test to determine the exact Fr of a device

num_workers = 8;    % Matlab parallel pool workers
% start paralell workers
p = gcp('nocreate'); 
if isempty(p)
    parpool('local',num_workers)
else
    if p.NumWorkers ~= num_workers
        delete(p)
        parpool('local',num_workers)
    end
end


% vidnames = { 'movie.mp4'};


% logfile = 'S:\AudioMNIST\test38\250hz\wordlog_py_test38_250hz.csv';
% a = readtable(logfile);
% vidnames = table2cell(a(:,2));






tic

vid_imu = [1, 0];

param = 'NormFrame'; 
test_param = [0];
test_param = test_param(end:-1:1);

for i_case = 1:length(testcase_folders)
    vidfolder = fullfile(basefolder, testcase_folders{i_case});
    
    vidnames = lsfiles(vidfolder, 'mp4');

    for i_param = 1:length(test_param)
        param_val = test_param(i_param);

        audfolder = 'init';   % ['norm',  num2str(param_val)]; 
        if ~isfile(fullfile(vidfolder, audfolder)) 
            mkdir(fullfile(vidfolder, audfolder))
        end

        parfor i_vid = 1:length(vidnames)
            vidname = vidnames{i_vid};
            disp([vidname, ', ',param, ': ', num2str(param_val)])

            
            if vid_imu(1)
                wavname = [vidname(1:end-4),'.wav'];
                
                outname = fullfile(vidfolder, audfolder, wavname);
                
                if isfile(outname)
                    disp([wavname, ' already there'])
                    continue
                end

    
                NG = NG_recover(vidfolder, vidname, [1,inf], [1 0], param, param_val);% consecutive decoding no amp
                sweep_data = [NG.xx, NG.yy];
                
%                 % !!!comment this out if not resampling!!!
%                 sweep_data = resample(sweep_data, 30000, 176000);

                % order correction 
                sweep_data = sweep_data(1:1080*floor(size(sweep_data,1)/1080),:);
                sweep_data = reshape(sweep_data, 1080, [], 8);
                sweep_data = reshape(sweep_data(end:-1:1, :, :), [], 8);
                
                for ii = 1:size(sweep_data,2)
                    sweep_data(:,ii) = sweep_data(:,ii) - mean(sweep_data(:,ii));
                    sweep_data(:,ii) = sweep_data(:,ii)/max(abs(sweep_data(:,ii)));
                end
                
                if all(sweep_data == 0, 'all' )
                    error(['Getting all zeros for ',wavname])
                end

                audiowrite(fullfile(vidfolder, audfolder, wavname),sweep_data, Fr);
                
                [sweep_data_check, ~] = audioread(fullfile(vidfolder, audfolder, wavname));
                if all(sweep_data_check == 0, 'all' )
                    error(['All zeros in output wav for ',wavname])
                end



%                 NG_recover(vidfolder, vidname, [1,inf], [1 0], param, param_val)
                
                

            end
            
            % imu data
            if vid_imu(2)
                wavname = [vidname(1:end-4),'_imu.wav'];
                filename = fullfile(vidfolder, replace(vidname,'.mp4','.csv'));
                tab = readtable(filename);
                dat = table2array(tab(:,2:7));
                dat = dat - mean(dat);
                dat = dat ./ max(abs(dat));
                fs = round(1/(mean(diff(table2array(tab(:,1))))/1e9));
                % dat = resample(dat, 30000, fs);
                audiowrite( fullfile(vidfolder, audfolder, wavname), dat, fs);
            end
        end
    end
    
end

toc



%%
folder = 'S:\AudioMNIST\test24_stab\px2-mars-oison\zero\';
csvname = 'gyro_accel.csv';
filename = fullfile(folder, csvname);
tab = readtable(filename);
dat = table2array(tab(:,2:7));
dat = dat - mean(dat);
dat = dat ./ max(abs(dat));

fs = round(1/(mean(diff(table2array(tab(:,1))))/1e9));
% dat = resample(dat, 30000, fs);
audiowrite( fullfile(folder, replace(csvname,'.csv','.wav')), dat, fs);



%% calculate the distance between different repititions of the same sampels 
% --> stability 
close all;
testbase_folder = 'S:\AudioMNIST\test24\px2-stabtests';
audfolder = '';

testcase_folders = {'stab_on', 'stab_off'};
dists_allcases = {};
num_chs = 8;
figure;
comp_groups = {1:5, 6:10, 11:15};
for i_case = 1:length(testcase_folders)
    testfolder = fullfile(testbase_folder, testcase_folders{i_case}, audfolder);
    disp(['case: ',testfolder])
    downsample_fac = 10;
    
    dists_allgroups = [];
    for i_group = 1:length(comp_groups)
        disp(['group: ',num2str(i_group)])
        group_idx = comp_groups(i_group);
        wavs = {};
        for i_wav = group_idx{1}
            wavname = fullfile(testfolder, replace(vidnames{i_wav},{'mkv','mp4','YUV'},'wav'));
            [yy, ~] = audioread(wavname);
            wavs = [wavs, yy];   % use the first channel for comparison 
        end
        % get the dtw distance 
        combos = nchoosek(1:length(wavs),2);
        oneD_dists = zeros(size(combos, 1), 1);
        twoD_dists = oneD_dists;
        for i_comp = 1:size(combos, 1)
            disp(['comp: ',num2str(i_comp)])
            wavA = wavs{combos(i_comp,1)};
            wavB = wavs{combos(i_comp,2)};
            for i_ch = 1:num_chs
                sigA = downsample(wavA(:,i_ch), downsample_fac);
                sigB = downsample(wavB(:,i_ch), downsample_fac);
                spA = abs(spectrogram(sigA,256));
                spB = abs(spectrogram(sigB,256));
                oneD_dists(i_comp) = oneD_dists(i_comp) + dtw(sigA, sigB);
                twoD_dists(i_comp) = twoD_dists(i_comp) + dtw(spA, spB);
            end
        end
        dists_allgroups = [dists_allgroups; [oneD_dists, twoD_dists]];

    end
    dists_allgroups = dists_allgroups/num_chs;
    dists_allcases = [dists_allcases,dists_allgroups];
    for i = 1:2
        subplot(1,2,i); plot(dists_allcases{i_case}(:,i),'LineWidth',1.5); hold on; title([num2str(i),'d dists: similarity'])
        xlabel('Pair Number'); ylabel('DTW Distance');
        legend(testcase_folders,'Interpreter','none'); set(gca,'fontsize', 18)
    end
    
end


all_means_sim = [];
all_stds_sim = [];
for i_case = 1:length(testcase_folders)
    all_means_sim = [all_means_sim; mean(dists_allcases{i_case})];
    all_stds_sim = [all_stds_sim; std(dists_allcases{i_case})];
end


% calculate the distance between different samples here 
comp_group_pairs = [1,2; 1,3; 2,3];
dists_allcases = {};
figure;
for i_case = 1:length(testcase_folders)
    testfolder = fullfile(testbase_folder, testcase_folders{i_case}, audfolder);
    disp(['case: ',testfolder])
    downsample_fac = 10;
    oneD_dists = zeros(5*5*3,1);
    twoD_dists = oneD_dists;
    count = 0;
    for i_grouppair = 1:length(comp_group_pairs)
        disp(['group pair: ',num2str(i_grouppair)])
        grouppair = comp_group_pairs(i_grouppair,:);
        group1 = comp_groups{grouppair(1)};
        group2 = comp_groups{grouppair(2)};
        for i_sample1 = group1
            wavname = fullfile(testfolder, replace(vidnames{i_sample1},{'mkv','mp4','YUV'},'wav'));
            [wavA, ~] = audioread(wavname);
            for i_sample2 = group2
            	wavname = fullfile(testfolder, replace(vidnames{i_sample2},{'mkv','mp4','YUV'},'wav'));
                [wavB, ~] = audioread(wavname);
                disp(['sample pair: ',num2str(i_sample1), '-', num2str(i_sample2)])
                count = count + 1;
                for i_ch = 1:num_chs
                    sigA = downsample(wavA(:,i_ch), downsample_fac);
                    sigB = downsample(wavB(:,i_ch), downsample_fac);
                    spA = abs(spectrogram(sigA,256));
                    spB = abs(spectrogram(sigB,256));
                    oneD_dists(count) = oneD_dists(count) + dtw(sigA, sigB);
                    twoD_dists(count) = twoD_dists(count) + dtw(spA, spB);
                end
            end
        end
    end
    oneD_dists = oneD_dists/num_chs;
    twoD_dists = twoD_dists/num_chs;
    tmp = [oneD_dists, twoD_dists];
    dists_allcases = [dists_allcases, tmp];
    for i = 1:2
        subplot(1,2,i); plot(tmp(:,i),'LineWidth',1.5); hold on; title([num2str(i),'d dists: dissimilarity'])
        xlabel('Pair Number'); ylabel('DTW Distance');
        legend(testcase_folders,'Interpreter','none'); set(gca,'fontsize', 18)
    end
end

all_means_dis = [];
all_stds_dis = [];
for i_case = 1:length(testcase_folders)
    all_means_dis = [all_means_dis; mean(dists_allcases{i_case})];
    all_stds_dis = [all_stds_dis; std(dists_allcases{i_case})];
end


figure;
set(groot,'defaultAxesTickLabelInterpreter','none');  
for i = 1:2
    subplot(1,2,i); 
    errorbar(all_means_sim(:,i), all_stds_sim(:,i), '-s', 'LineWidth',2,'MarkerSize',10,...
    'MarkerEdgeColor','blue','MarkerFaceColor','blue'); 
    hold on
    errorbar(all_means_dis(:,i), all_stds_dis(:,i), '-o', 'LineWidth',2,'MarkerSize',10,...
    'MarkerEdgeColor','red','MarkerFaceColor','red')
    title([num2str(i),'d dists'])
    xlim([.5, length(testcase_folders)+.5])
    xticks(1:length(testcase_folders))
    xticklabels(testcase_folders); 
    ylabel('DTW Distance');
    set(gca,'fontsize', 18)
end
legend({'Inner-class', 'Inter-class'})