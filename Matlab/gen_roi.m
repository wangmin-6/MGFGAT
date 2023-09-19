function [] = gen_roi()

%读取模板AAL.nii
%[mask,head] =y_Read('D:\DPABI_V6.1_220101\Templates\AAL_61x73x61_YCG.nii');%读取数据
img = spm_vol('D:\DPABI_V6.1_220101\Templates\AAL_61x73x61_YCG.nii');
mask = spm_read_vols(img);
save_path = 'D:\matlab_data\ALL_ROI\resample_roi_';
    for j = 1:30
        mask2 = resample(mask);
        path = strcat(save_path,num2str(j));
        path = strcat(path,'.nii');
        %nii = make_nii(mask2);
        %save_nii(nii,path);
        
        img.fname = path;
        spm_write_vol(img, mask2);
    end
end



