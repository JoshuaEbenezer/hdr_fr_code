function [Y,U,V,status]=yuv_import(filename,dims,startfrm,yuvformat)
%Imports YUV sequence
%[Y,U,V]=yuv_import(filename,dims,numfrm,startfrm)
%
%Input:
% filename - YUV sequence file
% dims - dimensions of the frame [width height]
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
end;

inprec = 'ubit8';
sampl = 420;
if (nargin < 4)
    startfrm = 0;
end;
if (nargin < 5)
    yuvformat = 'YUV420_8';
end;

if (strcmp(yuvformat,'YUV420_16'))
    inprec = 'uint16'; %'ubit16=>uint16'
elseif (strcmp(yuvformat,'YUV444_8'))
    sampl = 444;
end;

if (sampl == 420)
    dimsUV = dims / 2;
else
    dimsUV = dims;
end;
Yd = zeros(dims);
U = zeros(dimsUV);
V = zeros(dimsUV);
frelem = numel(Yd) + 2*numel(UVd);

ret = fseek(fid, startfrm * frelem , 0); %go to the starting frame
if ret ~= -1
	status = 1;
	Y = fread(fid,dims,inprec);
	U = fread(fid,dimsUV,inprec);
	V = fread(fid,dimsUV,inprec);


else
	Y=0;
	U=0;
	V=0;
	status = 0;
	fclose(fid);
end
