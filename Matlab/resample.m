function [mask] = resample(mask)
    TVB = unique(mask);
    %ROI,脑区的节点序号
    TVB = TVB(2:end);
    for j = 1:length(TVB)
        mask_index = find(mask == TVB(j)); %roi_id
        idx = randperm(length(mask_index));
        idx = idx(1:floor(length(mask_index)*2/3));
        b = mask_index(idx);
        mask(b) = 0;
    end
end