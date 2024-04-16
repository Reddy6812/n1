function new_array = mergeArrayByMean(array, dim, newDimSize)
% array 
% dim = 1,2 merge on row, col
% newDimSize
    ss = size(array);
    ss_merge = ss(dim);
    mean_idx = floor(linspace(1, ss_merge, newDimSize+1));
    new_array = [];
    for i = 1:newDimSize
        if length(ss) == 2
            if dim == 1
                new_array = cat(dim, new_array, mean(array(mean_idx(i):mean_idx(i+1),:),dim));
            elseif dim == 2
                new_array = cat(dim, new_array, mean(array(:,mean_idx(i):mean_idx(i+1)),dim));
            end
        elseif length(ss) == 3
            if dim == 1
                new_array = cat(dim, new_array, mean(array(mean_idx(i):mean_idx(i+1),:,:),dim));
            elseif dim == 2
                new_array = cat(dim, new_array, mean(array(:,mean_idx(i):mean_idx(i+1),:),dim));
            elseif dim == 3
                new_array = cat(dim, new_array, mean(array(:,:,mean_idx(i):mean_idx(i+1)),dim));
            end
        end
    end