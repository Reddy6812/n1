function res = readVid2Array(filename, frames)
    frame_start = frames(1);
    frame_end = frames(2);
    
    if contains(filename, '.mat')
        vid = load(filename);
        res = vid.frames;
        nf = size(res, length(size(res)));
        if frame_end == inf
            frame_end = nf;
        end
        if length(size(res)) == 4
            % TODO: rgb to gray
        elseif length(size(res)) == 3
            res = res(:,:,frame_start:frame_end);
        end
        
    elseif contains(filename, '.YUV')
        vidres = [800, 448];
        ii = fopen(filename);  res = fread(ii); fclose(ii);
        res = res(1:2:end);
        res = reshape(res, vidres(1), vidres(2), []);
        nf = size(res, length(size(res)));
        if frame_end == inf
            frame_end = nf;
        end
        res = res(:,:,frame_start:frame_end);
        
        
    else   
        vid = VideoReader(filename);
        nf = vid.NumFrames;
        if frame_end == inf
            frame_end = nf;
        end
        cf = read(vid,1);
        ss = size(cf);
        nf = frame_end-frame_start+1;
        res = zeros(ss(1),ss(2),nf, 'uint8');
        for i_frame = frame_start:frame_end
            try
                cf = read(vid,i_frame);
            catch
                break
            end
            if length(ss) > 2
                cf = rgb2gray(cf);
            end
            res(:,:,i_frame-frame_start+1) = cf;
        end
    end
end