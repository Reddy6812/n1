%%

clear;
basefolder = './';  % video vidfolder, results will be in the same vidfolder
testcase_folders = {''}; % lsfiles(basefolder, ''); if multiple subfolders
Fr = 34000;        % Sample rate of rolling shutter (row-scanning frequency)


tic


for i_case = 1:length(testcase_folders)
    vidfolder = fullfile(basefolder, testcase_folders{i_case});
    
    vidnames = lsfiles(vidfolder, 'mp4');


    audfolder = 'extracted_wav';   % wav file output folder
    if ~isfile(fullfile(vidfolder, audfolder)) 
        mkdir(fullfile(vidfolder, audfolder))
    end

    for i_vid = 1:length(vidnames)
        vidname = vidnames{i_vid};
        disp(vidname)

        
        wavname = [vidname(1:end-4),'.wav'];
        
        outname = fullfile(vidfolder, audfolder, wavname);
        
        if isfile(outname)
            disp([wavname, ' already there'])
            continue
        end


        NG = NG_recover(vidfolder, vidname, [1,inf], [1 0], 'NormFrame', 0);% 
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
            
    end
    
    
end

toc