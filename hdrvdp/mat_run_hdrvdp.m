
video_dir = '/media/josh-admin/seagate/fall2021_hdr_upscaled_yuv'
T = readtable('/home/josh-admin/hdr/qa/hdr_chipqa/fall2021_yuv_rw_info.csv')

disp(T)
yuv_names = T.yuv
disp(yuv_names)

width = 3840
height = 2160

for yuv_index = 1:length(yuv_names)
	feats_mat = zeros(framenums,1)
	yuv_name = char(yuv_names(yuv_index))
	upscaled_name = strcat(yuv_name(1:end-4),char('_upscaled.yuv'))
	disp(upscaled_name)
	full_yuv_name = fullfile(video_dir,upscaled_name)
	[Y,~,~,status] = yuv_import(full_yuv_name,[width,height],framenums,0,'YUV420P_16')
	for i=1:framenums
		vdp_result = hdrvdp3('quality',Y_dis_linear,Y_ref_linear,'luminance'
		
	end


	

end

quit
function res = hdrvdp3( task, test, reference, color_encoding, pixels_per_degree, options )
