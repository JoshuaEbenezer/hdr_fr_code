clear
parpool(15)
addpath(genpath('./hdrvdp-2.2.2'));
video_dir = '/mnt/31393986-51f4-4175-8683-85582af93b23/videos/HDR_2021_fall_yuv_upscaled/fall2021_hdr_upscaled_yuv/';
out_dir = '/media/zaixi/zaixi_nas/HDRproject/feats/fr_evaluate_HDRLIVE_correct/hdrvdp';
T = readtable('/mnt/31393986-51f4-4175-8683-85582af93b23/videos/HDR_2021_fall_yuv_upscaled/fall2021_yuv_rw_info.csv','Delimiter',',');
rng(0,'twister');
disp(T)

all_yuv_names = T.yuv;
ref_yuv_names = all_yuv_names((contains(all_yuv_names,'ref')));
framenums = T.framenos((contains(all_yuv_names,'ref')));

disp(ref_yuv_names)

width = 3840;
height = 2160;
pixels_per_degree =  hdrvdp_pix_per_deg( 65, [3840 2160], 1.455 );

% for yuv_index = 1:length(ref_yuv_names)
for yuv_index = 1:10
    
	yuv_name = char(ref_yuv_names(yuv_index));
    outname = fullfile(out_dir,strcat(yuv_name(1:end-4),'.mat'));
    % get reference
    ref_upscaled_name = yuv_name;
    disp(ref_upscaled_name);
    
    
    full_ref_yuv_name = fullfile(video_dir,ref_upscaled_name);    
    
    splits = strsplit(yuv_name,'_');
    content = splits(3);
    dis_names = all_yuv_names((contains(all_yuv_names,content)));
    dis_names = dis_names(~(contains(dis_names,'ref')));

    
    r = randi([1,framenums(yuv_index)-20],10,1);
    Q_mat = zeros(length(dis_names),length(r));
    Q_by_band_mat = zeros(length(dis_names),length(r),9);
    disp(size(Q_by_band_mat))
    for framenum_index=1:length(r)
        disp('out loop index');
        disp(framenum_index);
        framenum = r(framenum_index);
        [refY,~,~,status_ref] = yuv_import(full_ref_yuv_name,[width,height],framenum,'YUV420_16');
        refY_linear = eotf_pq(refY);
        if(status_ref==0)
            disp(strcat("Error reading frame in ",full_ref_yuv_name));
        end
        

        parfor dis_index = 1:length(dis_names)
        % for dis_index = 1:2
            disp(dis_index);
            dis_name = char(dis_names(dis_index));
            dis_upscaled_name = dis_name;
            disp(dis_upscaled_name);

            full_yuv_name = fullfile(video_dir,dis_upscaled_name);
            [disY,~,~,status_dis] = yuv_import(full_yuv_name,[width,height],framenum,'YUV420_16');
            if(status_dis==0)
                disp(strcat("Error reading frame in ",full_yuv_name));
            end
            disY_linear = eotf_pq(disY);
            
            vdp_result = hdrvdp(disY_linear,refY_linear,...
                'luminance',pixels_per_degree,{'rgb_display','led-lcd'});
            Q_mat(dis_index,framenum) = vdp_result.Q;
            disp(vdp_result.Q_by_band);
            Q_by_band_mat(dis_index,framenum,:) = vdp_result.Q_by_band;
        end
    end
  
    % featMap.ref_name = string(ref_yuv_names(yuv_index));
    % featMap.distorted_names = string(dis_names);
    % featMap.Qfeatures = Q_mat;
    % featMap.Q_by_band_features = Q_by_band_mat;
    % save(outname,'featMap');
    for i = 1:length(dis_names)

        outname = fullfile(out_dir,strcat(char(dis_names(i)),'.mat'));
        disp(outname);
        featMap.Q_by_band_features = squeeze(Q_by_band_mat(i,:,:));
        save(outname,'featMap');
    end
    end



quit
