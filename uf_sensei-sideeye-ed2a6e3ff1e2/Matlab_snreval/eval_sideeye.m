%% get audio from yan 
% clear; close all
% org_all = [];
% deg_all = [];
% degnoisy_all = [];
% orgfolder = 'S:\Corpus\AudioMNIST\data';
% degfolder = 'S:\AudioMNIST\test38_px2\audio';
% tab = readtable('eval\wordlog_py.csv');
% idx = table2array(tab(:,end)) == 3;
% tab = tab(idx, :);
% 
% deg_fs = 34000;
% fs = 16000;
% 
% progressbar()
% addpath('L:\ubuntu\SPQR\Projects\Camera\rolling\CODE\SpeechNoise');
% 
% for i_f =1:50
%     person = table2array(tab(i_f, 5));
%     orgname = table2array(tab(i_f, 1));
%     orgname = orgname{1};
%     orgpath = fullfile(orgfolder, num2str(person), orgname);
%     [org_wav, org_fs] = audioread(orgpath);
%     org_wav = resample(org_wav, fs, org_fs);
%     
%     degname = table2array(tab(i_f, 2));
%     degname = replace(degname{1}, 'mp4', 'wav');
%     degpath = fullfile(degfolder, degname);
%     [deg_wav, ~] = audioread(degpath);
%     deg_wav = resample(deg_wav, fs, deg_fs);
%     
%  	degnoisy_wav = deg_wav;
%     
%     
%     % find the best channel 
%     nists = zeros(8,1);
%     for i_c = 1:8
%         chwav = deg_wav(:,i_c);        
%         nists(i_c) = nist_stnr(chwav, fs, 0);
%     end
%     [~, maxc] = max(nists);
%     deg_wav = deg_wav(:, maxc);
%     for i_rep = 1:5
%         deg_wav = SSBoll79(deg_wav, fs, .2);
%     end
%     
%     
%     degnoisy_wav = degnoisy_wav(:, maxc);
%     degnoisy_wav = degnoisy_wav - mean(degnoisy_wav);
%     degnoisy_wav = degnoisy_wav/max(abs(degnoisy_wav));
%     
%     
%     
%     deg_wav = deg_wav - mean(deg_wav);
%     deg_wav = deg_wav/max(abs(deg_wav));
%     deg_wav = [deg_wav; zeros(fs*.2, 1)];
%     
%     org_wav = org_wav - mean(org_wav);
%     org_wav = org_wav/max(abs(org_wav));
%     org_wav = [org_wav; zeros(length(deg_wav)-length(org_wav), 1)];
% 
% 
%     org_all = [org_all;  org_wav];
%     deg_all = [deg_all;  deg_wav];
%     degnoisy_all = [degnoisy_all; degnoisy_wav];
%     
%     progressbar(i_f/50)
% 
% end
%  
% figure; 
% subplot(2,1,1); plot(org_all);
% subplot(2,1,2); plot(deg_all);
% distFig
% 
% % audiowrite('eval\raw.wav', [org_all, deg_all], fs);
% % audiowrite('eval\degnoisy.wav', degnoisy_all, fs);





%% 


clear; close all
addpath('L:\ubuntu\SPQR\Projects\Camera\rolling\CODE\SpeechNoise');
addpath('intelligibility_stoi\')
degbase = 'L:\tmpdata\';
cases = lsfiles(degbase, '');


tab = readtable('eval\wordlog_py.csv');
idx = table2array(tab(:,end)) == 3;
tab = tab(idx, :);

fs = 16000;

progressbar()

nn = 1536;   % number of samples used

denoise_mat2wav_score = [0 0 0 1];

score_stats = [];
cd('L:\ubuntu\SPQR\Projects\Camera\rolling\snreval-master')
for i_case = 2:length(cases)
    
    cd('L:\ubuntu\SPQR\Projects\Camera\rolling\snreval-master')
    
    casename = cases{i_case};
    casefolder = fullfile(degbase, casename);
    scores = zeros(nn, 2); % nist, intl
    
    for i_f = 1:nn
        
        % proc from yan to pirouz by noise reduction 
        if denoise_mat2wav_score(1)
        
            orgfolder = 'S:\Corpus\AudioMNIST\data';

            deg_fs = getPhoneSampleRate(casename);

            tmp = lsfiles(casefolder,'audio');
            degfolder = fullfile(casefolder, tmp{1});
            csvoutfolder = fullfile(casefolder, 'CSV2Py_1536');
            if ~isfile(csvoutfolder)
                mkdir(csvoutfolder)
            end


            person = table2array(tab(i_f, 5));
            orgname = table2array(tab(i_f, 1));
            orgname = orgname{1};
            orgpath = fullfile(orgfolder, num2str(person), orgname);


            degname = table2array(tab(i_f, 2));
            degname = replace(degname{1}, 'mp4', 'wav');
            degpath = fullfile(degfolder, degname);

            disp([cases{i_case}, ', ',degname]);
            disp([num2str(i_f),'/',num2str(nn)])

            checkname = fullfile(csvoutfolder, replace(degname, '.wav', '_ORG.csv'));
            if isfile(checkname)
                disp('Already there')
                continue;
            end


            [org_wav, org_fs] = audioread(orgpath);
            org_wav = resample(org_wav, fs, org_fs);
            [deg_wav, ~] = audioread(degpath);
            deg_wav = resample(deg_wav, fs, deg_fs);


            % find the best channel 
            nists = zeros(8,1);
            for i_c = 1:8
                chwav = deg_wav(:,i_c);        
                nists(i_c) = nist_stnr(chwav, fs, 0);
            end
            [~, maxc] = max(nists);
            deg_wav = deg_wav(:, maxc);
            for i_rep = 1:5
                deg_wav = SSBoll79(deg_wav, fs, .2);
            end
            
            if all(deg_wav == 0)
                error(['All 0 wav file: ', degpath])
                deg_wav = rand(size(deg_wav));
            end

            deg_wav = deg_wav - mean(deg_wav);
            deg_wav = deg_wav/max(abs(deg_wav));

            org_wav = org_wav - mean(org_wav);
            org_wav = org_wav/max(abs(org_wav));

            csvwrite(fullfile(csvoutfolder, replace(degname, '.wav', '_EXT.csv')), deg_wav)
            csvwrite(fullfile(csvoutfolder, replace(degname, '.wav', '_ORG.csv')), org_wav)

            disp('File done')
        end
        
        
        
        
        % get trimmed wav audio  from pirouz's mat 
        if denoise_mat2wav_score(2)
            
            
            orgfolder = fullfile(casefolder, 'CSV2Py_1536_output');
            degfolder = orgfolder;

            prefix = table2array(tab(i_f, 2)); prefix = prefix{1};
            orgpath = fullfile(orgfolder, replace(prefix, '.mp4', '_ORG.mat'));
            org_wav = load(orgpath); org_wav = org_wav.mydata';

            degpath = fullfile(orgfolder, replace(prefix, '.mp4', '_EXT.mat'));
            deg_wav = load(degpath); deg_wav = deg_wav.mydata';

            deg_wav = deg_wav - mean(deg_wav);
            deg_wav = deg_wav/max(abs(deg_wav));

            org_wav = org_wav - mean(org_wav);
            org_wav = org_wav/max(abs(org_wav));


            % this is for scores of individual average  
            rep = 2;
            outfolder = ['compwav_rep',num2str(rep)];
            if ~isfile(fullfile(orgfolder, outfolder))
                mkdir( fullfile(orgfolder, outfolder))
            end
            deg_wav = repmat(deg_wav, rep,1);
            degname = fullfile(orgfolder, outfolder, replace(prefix, '.mp4', '_deg.wav'));
            audiowrite(degname, deg_wav, fs)
            org_wav = repmat(org_wav, rep,1);
            orgname = fullfile(degfolder, outfolder, replace(prefix, '.mp4', '_org.wav'));
            audiowrite(orgname, org_wav, fs)
            
            
        end
        
        
%         get audio metric scores  
        if denoise_mat2wav_score(3)
            
            orgfolder = fullfile(casefolder, 'CSV2Py_1536_output\compwav_rep2');
            degfolder = orgfolder;
            tmp = lsfiles(casefolder, 'audio');
            degnoisyfolder = fullfile(degbase, casename, tmp{1});
            disp([casename, ': ',num2str(i_f)])
            prefix = table2array(tab(i_f, 2)); prefix = prefix{1};
            orgpath = fullfile(orgfolder, replace(prefix, '.mp4', '_org.wav'));
            [org_wav, fs] = audioread(orgpath); 
            degpath = fullfile(degfolder, replace(prefix, '.mp4', '_deg.wav'));
            [deg_wav, ~] = audioread(degpath);
            degnoisypath = fullfile(degnoisyfolder, replace(prefix, '.mp4', '.wav'));
            [degnoisy_wav, fsnoisy] = audioread(degnoisypath);

            scores(i_f, 1) = nist_stnr(degnoisy_wav, fsnoisy, 0);
            scores(i_f, 2) = stoi(org_wav, deg_wav, fs);
            
        end
        
    end
    
    if denoise_mat2wav_score(1)
        cd(casefolder)
        cmd = 'tar cvf CSV2Py_1536.tar CSV2Py_1536';
        system(cmd)
    end
    
    if denoise_mat2wav_score(3)
        save(fullfile(casefolder, 'aud_scores.mat'), 'scores')
    end
    
    if denoise_mat2wav_score(4)
        load(fullfile(casefolder, 'aud_scores.mat'));
        scores = scores(1:nn, :);
        score_stats = [score_stats; nanmean(scores), nanstd(scores)];
    end

    progressbar(i_case/length(cases))

end

if denoise_mat2wav_score(4)
    score_stats = [cell2table(cases'), array2table(score_stats)]
end




 








%% calc scores individual average  
% directly calculating induvidual intl scores produce nan very often
% maybe because the duration's too short
% so I tried repeating the same signals. 2 reps have similar scores as 10
clear; close all
org_all = [];
deg_all = [];

test = 'test38_px2_-20db';

orgfolder = ['S:\AudioMNIST\',test,'\CSV2Py_1536\compwav_rep2'];
degfolder = ['S:\AudioMNIST\',test,'\CSV2Py_1536\compwav_rep2'];
degnoisyfolder = ['S:\AudioMNIST\',test,'\audio'];
tab = readtable('eval\wordlog_py.csv');
idx = table2array(tab(:,end)) == 3;
tab = tab(idx, :);

checkfiles = 1:10; % 
% checkfiles = [];

scores = zeros(length(checkfiles), 2); % nist, intl
addpath('intelligibility_stoi\')
% preparePESQ()
progressbar()
for ii = 1:length(checkfiles)
    i_f = checkfiles(ii);
    disp(num2str(i_f))
    prefix = table2array(tab(i_f, 2)); prefix = prefix{1};
    orgpath = fullfile(orgfolder, replace(prefix, '.mp4', '_org.wav'));
    [org_wav, fs] = audioread(orgpath); 
    degpath = fullfile(degfolder, replace(prefix, '.mp4', '_deg.wav'));
    [deg_wav, ~] = audioread(degpath);
    degnoisypath = fullfile(degnoisyfolder, replace(prefix, '.mp4', '.wav'));
    [degnoisy_wav, fsnoisy] = audioread(degnoisypath);
    
%     [SNRstnro,moslqo, rawmos] = snreval(degpath,'-clean',orgpath);
    
    scores(i_f, 1) = nist_stnr(degnoisy_wav, fsnoisy, 0);
    
    scores(i_f, 2) = stoi(org_wav, deg_wav, fs);
    
    progressbar(ii/length(checkfiles))

end


out_scores = scores;

aa = out_scores(:,1);
idx = ~isnan(aa);
out_scores = out_scores(idx, :);
aa = out_scores(:,2);
idx = ~isnan(aa);
out_scores = out_scores(idx, :);

nanmean(out_scores)

% save(fullfile(orgfolder, 'ascores.mat'), 'out_scores')

