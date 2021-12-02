function [Y,U,V,status]=yuv_import(filename,dims_2d,startfrm,yuvformat)
%Imports YUV sequence
%[Y,U,V]=yuv_import(filename,dims,numfrm,startfrm)
%
%Input:
% filename - YUV sequence file
% dims_2d - dimensions of the frame [width height]
% numfrm - number of frames to read
% startfrm - [optional, default = 0] specifies from which frame to start reading
%            with the convention that the first frame of the sequence is 0-
%            numbered
% yuvformat - [optional, default = 'YUV420_8']. YUV format, supported formats 
%             are: 
%             'YUV444_8' = 4:4:4 sampling, 8-bit precision 
%             'YUV420_8' = 4:2:0 sampling, 8-bit precision
%             'YUV420_16' = 4:2:0 sampling, 16-bit precision
%
%Output:
% Y, U ,V - cell arrays of Y, U and V components  
%
%Note:
% Supported YUV formats are (corresponding yuvformat variable):
%  'YUV420_8' = 4:2:0 sampling, 8-bit precision (default)
%  'YUV420_16' = 4:2:0 sampling, 16-bit precision
%
%Example:
% [Y, U, V] = yuv_import('FOREMAN_352x288_30_orig_01.yuv',[352 288],2);
% image_show(Y{1},256,1,'Y component');
% [Y, U, V] = yuv_import('sequence.yuv',[1920 1080],2,0,'YUV420_16');

fid=fopen(filename,'r');
if (fid < 0) 
    error('File does not exist!');
end

sampl = 420;


if (strcmp(yuvformat,'YUV420_16'))
    inprec = 'uint16'; %'ubit16=>uint16'
elseif (strcmp(yuvformat,'YUV444_8'))
    sampl = 444;
end

dims = dims_2d(1)*dims_2d(2);

if (sampl == 420)
    dimsUV_2d = dims_2d/2;
    dimsUV = dims / 4;
else
    dimsUV = dims;
end

frelem = dims + 2*dimsUV;

ret = fseek(fid, (startfrm-1) * frelem , 0); %go to the starting frame
if ret ~= -1
	status = 1;
	Y1d = fread(fid,dims,inprec);
	U1d = fread(fid,dimsUV,inprec);
	V1d = fread(fid,dimsUV,inprec);
    Y = reshape(Y1d,dims_2d(1),dims_2d(2))';
    U_half = reshape(U1d,[dimsUV_2d(1),dimsUV_2d(2)]);
    V_half = reshape(V1d,[dimsUV_2d(1),dimsUV_2d(2)]);
    U = imresize(U_half,2,'method','nearest')';
    V = imresize(V_half,2,'method','nearest')';
    
else
	Y=0;
	U=0;
	V=0;
	status = 0;
	fclose(fid);
end
