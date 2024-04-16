function varargout = cNG_recover(folder, filename, useframes, mag_org, varargin)
% The phaseAmplify stuff can be ignored
    fps = 30;
    samplingRate = fps; % Hz
    alpha = 15;    
    sigma = 3;   % Pixels
    pyrType =  'octave';

    compMethod = 'cons';
%     compMethod = 'init';

    p = inputParser();
    addOptional(p, 'DMiter', 5);
    addOptional(p, 'DMsmooth', 1.0);
    addOptional(p, 'DMlevel', 3);
    addOptional(p, 'NormFrame', false);
    addOptional(p, 'DownFac', 1);
    
    parse(p, varargin{:});
    reg_iters = p.Results.DMiter;
    reg_smooth = p.Results.DMsmooth;
    reg_level = p.Results.DMlevel;
    normalize_frame = p.Results.NormFrame;
    downsample_fac = p.Results.DownFac;
    

    num_output_groups = 4;

    in_name = fullfile(folder, filename);
    for i_res = 1:2
        if i_res == 2
            tag = 'amped';
        else
            tag = 'noamp';
        end
        if ~mag_org(i_res)
            continue
        end
        if i_res == 1
            res = readVid2Array(in_name, useframes);   % For comparison, w.o. amplification
        elseif i_res == 2
            disp('Amplifying...')
            res = phaseAmplify_ly(in_name, alpha, samplingRate, '',...
                'sigma', sigma,'pyrType', pyrType,'scaleVideo', 1, 'useFrames', useframes);
        end

        num_frames = size(res, 3);
        
        % !!! note: flipping here !!!
        if size(res,1) > size(res,2)
            res = permute(res,[2 1 3]);
        end

        cat_xx = []; 
        cat_yy = [];
        for i_frame = 2:num_frames
            fprintf('Res: %d, frame:%d/%d\n', i_res, i_frame, num_frames)
            if strcmp(compMethod, 'init')
                img0 = res(:,:,1);
            elseif strcmp(compMethod, 'cons')
                img0 = res(:,:,i_frame-1);
            end
            img1 = res(:,:,i_frame);
            
            if downsample_fac ~= 1
                img0 = imresize(img0, downsample_fac);
                img1 = imresize(img1, downsample_fac);
            end
            
            
            [NG_field,~] = imregdemons(img0,img1,reg_iters,...
                'AccumulatedFieldSmoothing',reg_smooth,'PyramidLevels',reg_level,'DisplayWaitbar',false);
            MVx = NG_field(:,:,1);
            MVy = NG_field(:,:,2);
            
            
            frame_xx = mergeArrayByMean(MVx,2,num_output_groups);
            frame_yy = mergeArrayByMean(MVy,2,num_output_groups);
            if normalize_frame
                frame_xx = frame_xx - mean(frame_xx);
                frame_yy = frame_yy - mean(frame_yy);
                frame_xx = frame_xx./max(abs((frame_xx)));
                frame_yy = frame_yy./max(abs((frame_yy)));
            end
            cat_xx = cat(1, cat_xx, frame_xx);
            cat_yy = cat(1, cat_yy, frame_yy);
            
        end
        
        cat_xx(isnan(cat_xx)) = 0;
        cat_yy(isnan(cat_yy)) = 0;
        
        NG.xx = cat_xx;
        NG.yy = cat_yy;
        
        if nargout == 0
            save(fullfile(folder, [filename(1:end-4),'.mat']), 'NG');
        else
            varargout{1} = NG;
        end
    end

end

