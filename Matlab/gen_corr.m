function [] = gen_corr()
    path_Philips = 'F:\wangmin\Philips\work\FunImgARWSC';
    %path_SIEMENS = 'G:\MedicalImage\FunImgARWSC';
    path = path_Philips;
    temp=dir(path);
    %被试的个数
    temp=temp(3:end);
    for i = 1:length(temp)
        name = temp(i).name;
        pa_name = strcat(path,'\',name,'\');
        corr_30time(pa_name,name); 
        %gen_corr_not_resample(pa_name,name)
    end
    
end