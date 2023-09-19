function [] = corr_30time(path,name)
    %temp=dir(path);
    %���Եĸ���
    %temp=temp(3:end);

    roi_file_path = 'K:\matlab_code_nii\nii\resample_roi_';
    pa = spm_vol(strcat(path,'\CovRegressed_4DVolume.nii'));
    private = pa.private;
    dat = private.dat;
    dim = dat.dim;
    time_point = dim(4); 
    
    tempdata = spm_read_vols(pa);
    for roi_id = 1:5
        
        roi_path = strcat(roi_file_path,num2str(roi_id),'.nii');
        img = spm_vol(roi_path);
        mask = spm_read_vols(img);
        [m1,m2,m3] = size(mask);
        tempdata = reshape(tempdata, m1*m2*m3, []);
        TVB = unique(mask);
        %ROI,�����Ľڵ����
        TVB = TVB(2:end);
                
        fun_time = zeros(length(TVB),time_point);  
        %%����洢�������ӵľ��󣬽ڵ�*�ڵ�*����
        fun_con = zeros(length(TVB),length(TVB));
        
        for j = 1:length(TVB)
            index = find(mask == TVB(j));
            con_per = tempdata(index, : );
            fun_time(j,:) = mean(con_per);
        end
        fun_con(:,:) = corr(fun_time(:,:)');
        %save_mat_path = strcat('K:\data\ROICorr\SIEMENS\ROICorr_',name,'_',num2str(time_point),'_',num2str(roi_id),'.mat');
        %save_mat_path = strcat('K:\data\ROICorr\SIEMENS\ROICorr_',name,'_',num2str(roi_id),'.mat');
        
        save_mat_path = strcat('K:\data\ROICorr5\SIEMENS\ROICorr_',name,'_',num2str(roi_id),'.mat');
        save(save_mat_path, 'fun_con');
    end
end