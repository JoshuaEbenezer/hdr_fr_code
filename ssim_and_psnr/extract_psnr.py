import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
import os
import glob
import cv2
from joblib import Parallel,delayed,dump
import scipy.ndimage
import pandas as pd
import skimage.util
import math
from hdr_utils import hdr_yuv_read

def Y_compute_gnl(Y,nl_method,nl_param):
    Y = Y.astype(np.float32)
    if(nl_method=='nakarushton'):
        Y_transform =  Y/(Y+avg_luminance) #
    elif(nl_method=='sigmoid'):
        Y_transform = 1/(1+(np.exp(-(1e-3*(Y-avg_luminance)))))
    elif(nl_method=='logit'):
        delta = nl_param
        Y_scaled = -0.99+1.98*(Y-np.amin(Y))/(1e-3+np.amax(Y)-np.amin(Y))
        Y_transform = np.log((1+(Y_scaled)**delta)/(1-(Y_scaled)**delta))
        if(delta%2==0):
            Y_transform[Y<0] = -Y_transform[Y<0] 
    elif(nl_method=='exp'):
        delta = nl_param
        Y = -4+(Y-np.amin(Y))* 8/(1e-3+np.amax(Y)-np.amin(Y))
        Y_transform =  np.exp(np.abs(Y)**delta)-1
        Y_transform[Y<0] = -Y_transform[Y<0]
    elif(nl_method=='custom'):
        Y = -0.99+(Y-np.amin(Y))* 1.98/(1e-3+np.amax(Y)-np.amin(Y))
        Y_transform = transform(Y,5)


    return Y_transform
def Y_compute_lnl(Y,nl_method,nl_param):
    Y = Y.astype(np.float32)

    if(nl_method=='logit'):
        maxY = scipy.ndimage.maximum_filter(Y,size=(31,31))
        minY = scipy.ndimage.minimum_filter(Y,size=(31,31))
        delta = nl_param
        Y_scaled = -0.99+1.98*(Y-minY)/(1e-3+maxY-minY)
        Y_transform = np.log((1+(Y_scaled)**delta)/(1-(Y_scaled)**delta))
        if(delta%2==0):
            Y_transform[Y<0] = -Y_transform[Y<0] 
    elif(nl_method=='exp'):
        maxY = scipy.ndimage.maximum_filter(Y,size=(31,31))
        minY = scipy.ndimage.minimum_filter(Y,size=(31,31))
        delta = nl_param
        Y = -4+(Y-minY)* 8/(1e-3+maxY-minY)
        Y_transform =  np.exp(np.abs(Y)**delta)-1
        Y_transform[Y<0] = -Y_transform[Y<0]
    elif(nl_method=='custom'):
        maxY = scipy.ndimage.maximum_filter(Y,size=(31,31))
        minY = scipy.ndimage.minimum_filter(Y,size=(31,31))
        Y = -0.99+(Y-minY)* 1.98/(1e-3+maxY-minY)
        Y_transform = transform(Y,5)
    elif(nl_method=='sigmoid'):
        avg_luminance = scipy.ndimage.gaussian_filter(Y,sigma=7.0/6.0)
        Y_transform = 1/(1+(np.exp(-(1e-3*(Y-avg_luminance)))))
    return Y_transform


def single_vid_psnr(i):

    dis_video_name = upscaled_yuv_names[i]
    ref_video_name = os.path.join('/mnt/31393986-51f4-4175-8683-85582af93b23/videos/HDR_2022_SPRING_yuv/',ref_names[i])
    if dis_video_name == ref_names[i]:
        return
    speed_outname = os.path.join(output_pth,os.path.splitext(os.path.basename(dis_video_name))[0]+'.z')
    if os.path.exists(speed_outname):
        return
    fps =25
    dis_video = open(os.path.join('/mnt/31393986-51f4-4175-8683-85582af93b23/videos/HDR_2022_SPRING_yuv/',upscaled_yuv_names[i]))

    ref_video = open(ref_video_name)

    width,height=int(3840),int(2160)
   
    print(ref_video_name,dis_video_name,height,width,fps)

    print(ref_video_name,dis_video_name,height,width,fps)
    psnr_list = []


    for framenum in range(framenos_list[i]):
        try:
            ref_y,_,_ =hdr_yuv_read(ref_video,framenum,height,width)
            dis_y,_,_ =hdr_yuv_read(dis_video,framenum,height,width) 

        except Exception as e:
            print(e)
            if(len(psnr_list)):
                dump(psnr_list,speed_outname)

            break
        psnr_val = psnr(ref_y,dis_y,data_range = 1024)

        psnr_list.append(psnr_val)


        dump(psnr_list,speed_outname)




    #psnr_command = ['./run_psnr.sh',ref_video,dis_vid,psnr_outname,dis_fps]
    #try:
    #subprocess.check_call(psnr_command)
    #subprocess.check_call(psnr_command)
    #except:
    #    return
    return


csv_file = '/home/zs5397/code/hdr_fr_code/spring2022_yuv_info.csv'
csv_df = pd.read_csv(csv_file)
files = csv_df["yuv"]
fps_list = 25
framenos_list = csv_df["framenos"]
upscaled_yuv_names = csv_df['yuv']
ref_names = csv_df['ref']
output_pth = './features/psnr_features/'
if not os.path.exists(output_pth):
    os.makedirs(output_pth)


Parallel(n_jobs=60)(delayed(single_vid_psnr)(i) for i in range(len(upscaled_yuv_names)))
