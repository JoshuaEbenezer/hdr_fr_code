clear
addpath(genpath('./hdrvdp-3.0.6'))
video_dir = '/mnt/31393986-51f4-4175-8683-85582af93b23/videos/HDR_2022_SPRING_yuv/';
out_dir = './features/hdrvdp3_features/';
T = readtable('/home/zs5397/code/hdr_fr_code/spring2022_yuv_info.csv','Delimiter',',');
rng(0,'twister');
%parpool(15)
disp(T)

all_yuv_names = T.yuv;
ref_yuv_names = all_yuv_names((contains(all_yuv_names,'mu100000')));
framenums = T.framenos((contains(all_yuv_names,'mu100000')));

disp(ref_yuv_names)

width = 3840;
height = 2160;
pixels_per_degree =  hdrvdp_pix_per_deg( 65, [3840 2160], 1.455 );

for yuv_index = 9:length(ref_yuv_names)
    
    
	yuv_name = char(ref_yuv_names(yuv_index));
    outname = fullfile(out_dir,strcat(yuv_name(1:end-4),'.mat'));
    % get reference
    ref_upscaled_name = yuv_name;
    disp(ref_upscaled_name);
    
    full_ref_yuv_name = fullfile(video_dir,ref_upscaled_name);    
    
      
    splits = strsplit(yuv_name,'_');
    content = splits(1);
    dis_names = all_yuv_names((contains(all_yuv_names,content)));
    dis_names = dis_names(~(contains(dis_names,'mu100000')));


    r = randi([1,framenums(yuv_index)],10,1);
    Q_mat = zeros(length(dis_names),length(r));
    QJOD_mat = zeros(length(dis_names),length(r));
    
    for framenum_index=1:length(r)
        framenum = r(framenum_index);
        [refY,refU,refV,status_ref] = yuv_import(full_ref_yuv_name,[width,height],framenum,'YUV420_16');
        ref_YUV = cast(cat(3,refY,refU,refV),'uint16');
        ref_rgb_bt2020 = ycbcr2rgbwide(ref_YUV,10);
        ref_rgb_bt2020_linear = eotf_pq(ref_rgb_bt2020);
        if(status_ref==0)
            disp(strcat("Error reading frame in ",full_ref_yuv_name));
        end
        

        parfor dis_index = 1:length(dis_names)

            dis_name = char(dis_names(dis_index));
            dis_upscaled_name = dis_name;
            disp(dis_upscaled_name);

            full_yuv_name = fullfile(video_dir,dis_upscaled_name);
            [disY,disU,disV,status_dis] = yuv_import(full_yuv_name,[width,height],framenum,'YUV420_16');
            if(status_dis==0)
                disp(strcat("Error reading frame in ",full_yuv_name));
            end
            dis_YUV = cast(cat(3,disY,disU,disV),'uint16');
            dis_rgb_bt2020 = ycbcr2rgbwide(dis_YUV,10);
            dis_rgb_bt2020_linear = eotf_pq(dis_rgb_bt2020);
            
            vdp_result = hdrvdp3('quality',dis_rgb_bt2020_linear,ref_rgb_bt2020_linear,...
                'rgb-native',pixels_per_degree,{'rgb_display','oled'});
            Q_mat(dis_index,framenum) = vdp_result.Q;
            QJOD_mat(dis_index,framenum) = vdp_result.Q_JOD;  
            
        end
    end
  
    featMap.ref_name = string(ref_yuv_names(yuv_index));
    featMap.distorted_names = string(dis_names);
    featMap.Qfeatures = Q_mat;
    featMap.QJOD_features = QJOD_mat;
    save(outname,'featMap');

end

quit
