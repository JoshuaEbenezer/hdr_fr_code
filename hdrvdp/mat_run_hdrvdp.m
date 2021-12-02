
video_dir = '/media/josh-admin/seagate/fall2021_hdr_upscaled_yuv';
T = readtable('/home/josh-admin/hdr/qa/hdr_chipqa/fall2021_yuv_rw_info.csv');

disp(T)
all_yuv_names = T.yuv;
ref_yuv_names = all_yuv_names((contains(all_yuv_names,'ref')));
disp(ref_yuv_names)

width = 3840
height = 2160

for yuv_index = 1:length(ref_yuv_names)
    
	feats_mat = zeros(framenums,1);
	yuv_name = char(ref_yuv_names(yuv_index));
    ref_upscaled_name = strcat(yuv_name(1:end-4),char('_upscaled.yuv'));
    disp(ref_upscaled_name);
    full_yuv_name = fullfile(video_dir,dis_upscaled_name);    
    splits = strsplit(yuv_name,'_');
    content = splits(3);
    dis_names = all_yuv_names((contains(all_yuv_names,content)));
    for dis_index = 1:length(dis_names)
        dis_name = dis_names(dis_index)
        dis_upscaled_name = strcat(dis_name(1:end-4),char('_upscaled.yuv'));
        disp(dis_upscaled_name);
        full_yuv_name = fullfile(video_dir,dis_upscaled_name);
        [disY,disU,disV,status] = yuv_import(full_yuv_name,[width,height],framenum,'YUV420P_16');
        dis_rgb_bt2020 = ycbcr2rgbwide(cat(disY,disU,disV,3),10);
        dis_rgb_bt2020_linear = eotf_pq(dis_rgb_bt2020);
        for i=1:framenums
            vdp_result = hdrvdp3('quality',dis_rgb_bt2020_linear,ref_rgb_bt2020_linear,'')


        end
    end


end

quit
