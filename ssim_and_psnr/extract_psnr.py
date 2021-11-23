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

csv_file = '/home/josh-admin/hdr/qa/hdr_vmaf/python_vmaf/fall2021_yuv_rw_info.csv'
csv_df = pd.read_csv(csv_file)
files = csv_df["yuv"]
ref_files = glob.glob('/media/josh-admin/seagate/fall2021_hdr_upscaled_yuv/4k_ref_*')
fps_list = csv_df["fps"]
framenos_list = csv_df["framenos"]
upscaled_yuv_names = [x[:-4]+'_upscaled.yuv' for x in csv_df['yuv']]
content_list = [f.split('_')[2] for f in upscaled_yuv_names]

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
    if('ref' in dis_video_name):
        return
    content =content_list[i] 
    fps =fps_list[i] 
    ref_video_name = os.path.join('/media/josh-admin/seagate/fall2021_hdr_upscaled_yuv/4k_ref_'+content+'_upscaled.yuv')
    dis_video = open(os.path.join('/media/josh-admin/seagate/fall2021_hdr_upscaled_yuv',dis_video_name))
    ref_video = open(ref_video_name)

    width,height=int(3840),int(2160)
    psnr_outname = os.path.join('./features/psnr_features/',os.path.splitext(os.path.basename(dis_video_name))[0]+'.z')
    psnr_exp_gnl1_outname = os.path.join('./features/psnr_features_global_m_exp1/',os.path.splitext(os.path.basename(dis_video_name))[0]+'.z')
    psnr_exp_gnl2_outname = os.path.join('./features/psnr_features_global_m_exp2/',os.path.splitext(os.path.basename(dis_video_name))[0]+'.z')
    psnr_exp_lnl1_outname = os.path.join('./features/psnr_features_local_m_exp1/',os.path.splitext(os.path.basename(dis_video_name))[0]+'.z')
    psnr_exp_lnl2_outname = os.path.join('./features/psnr_features_local_m_exp2/',os.path.splitext(os.path.basename(dis_video_name))[0]+'.z')
    psnr_logit_gnl1_outname = os.path.join('./features/psnr_features_global_logit1/',os.path.splitext(os.path.basename(dis_video_name))[0]+'.z')
    psnr_logit_gnl2_outname = os.path.join('./features/psnr_features_global_logit2/',os.path.splitext(os.path.basename(dis_video_name))[0]+'.z')
    psnr_logit_lnl1_outname = os.path.join('./features/psnr_features_local_logit1/',os.path.splitext(os.path.basename(dis_video_name))[0]+'.z')
    psnr_logit_lnl2_outname = os.path.join('./features/psnr_features_local_logit2/',os.path.splitext(os.path.basename(dis_video_name))[0]+'.z')
    if(os.path.exists(psnr_exp_gnl1_outname)):
        return
    print(ref_video_name,dis_video_name,height,width,fps,psnr_exp_gnl1_outname)
    psnr_list = []
    psnr_exp_lnl1_list = []
    psnr_exp_lnl2_list = []
    psnr_exp_gnl1_list = []
    psnr_exp_gnl2_list = []
    psnr_exp_lnl2_list = []
    psnr_logit_gnl1_list = []
    psnr_logit_gnl2_list = []
    psnr_logit_lnl1_list = []
    psnr_logit_lnl2_list = []


    for framenum in range(framenos_list[i]):
        try:
            ref_y,_,_ =hdr_yuv_read(ref_video,framenum,height,width)

            #exp local
            ref_y_lnl1 = Y_compute_lnl(ref_y,nl_method='exp',nl_param=1) 
            ref_y_lnl2 = Y_compute_lnl(ref_y,nl_method='exp',nl_param=2) 

            # logit lnl
            ref_y_logit_lnl1 = Y_compute_lnl(ref_y,nl_method='logit',nl_param=1) 
            ref_y_logit_lnl2 = Y_compute_lnl(ref_y,nl_method='logit',nl_param=2) 

            # exp gnl
            ref_y_gnl1 = Y_compute_gnl(ref_y,nl_method='exp',nl_param=1) 
            ref_y_gnl2 = Y_compute_gnl(ref_y,nl_method='exp',nl_param=2) 

            # logit gnl
            ref_y_gnl_logit1 = Y_compute_gnl(ref_y,nl_method='logit',nl_param=1) 
            ref_y_gnl_logit2 = Y_compute_gnl(ref_y,nl_method='logit',nl_param=2) 


            dis_y,_,_ =hdr_yuv_read(dis_video,framenum,height,width) 

            # exp lnl
            dis_y_lnl1 = Y_compute_lnl(dis_y,nl_method='exp',nl_param=1) 
            dis_y_lnl2 = Y_compute_lnl(dis_y,nl_method='exp',nl_param=2) 

            # exp gnl
            dis_y_gnl1 = Y_compute_gnl(dis_y,nl_method='exp',nl_param=1) 
            dis_y_gnl2 = Y_compute_gnl(dis_y,nl_method='exp',nl_param=2) 

            # logit lnl
            dis_y_lnl_logit1 = Y_compute_lnl(dis_y,nl_method='logit',nl_param=1) 
            dis_y_lnl_logit2 = Y_compute_lnl(dis_y,nl_method='logit',nl_param=2) 
            # logit gnl
            dis_y_gnl_logit1 = Y_compute_gnl(dis_y,nl_method='logit',nl_param=1) 
            dis_y_gnl_logit2 = Y_compute_gnl(dis_y,nl_method='logit',nl_param=2) 


        except Exception as e:
            print(e)
            print(frame_num, ' frames read')
            if(len(psnr_list)):
                dump(psnr_list,psnr_outname)
                dump(psnr_exp_lnl1_list,psnr_exp_lnl1_outname)
                dump(psnr_exp_lnl2_list,psnr_exp_lnl2_outname)
                dump(psnr_exp_gnl1_list,psnr_exp_gnl1_outname)
                dump(psnr_exp_gnl2_list,psnr_exp_gnl2_outname)
                dump(psnr_logit_gnl1_list,psnr_logit_gnl1_outname)
                dump(psnr_logit_gnl2_list,psnr_logit_gnl2_outname)
                dump(psnr_logit_lnl1_list,psnr_logit_lnl1_outname)
                dump(psnr_logit_lnl2_list,psnr_logit_lnl2_outname)
            break
        psnr_val = psnr(ref_y,dis_y,data_range = 1024)
        psnr_exp_lnl1 = psnr(ref_y_lnl1,dis_y_lnl1,data_range = 108)
        psnr_exp_lnl2 = psnr(ref_y_lnl2,dis_y_lnl2,data_range = 17772220)

        psnr_exp_gnl1 = psnr(ref_y_gnl1,dis_y_gnl1,data_range = 108)
        psnr_exp_gnl2 = psnr(ref_y_gnl2,dis_y_gnl2,data_range = 17772220)
        
        psnr_logit_gnl1 = psnr(ref_y_gnl_logit1,dis_y_gnl_logit1,data_range = 4.6)
        psnr_logit_gnl2 = psnr(ref_y_gnl_logit2,dis_y_gnl_logit2,data_range = 4)
        
        psnr_logit_lnl1 = psnr(ref_y_logit_lnl1,dis_y_lnl_logit1,data_range = 4.6)
        psnr_logit_lnl2 = psnr(ref_y_logit_lnl2,dis_y_lnl_logit2,data_range = 4)

        psnr_list.append(psnr_val)
        psnr_exp_lnl1_list.append(psnr_exp_lnl1)
        psnr_exp_lnl2_list.append(psnr_exp_lnl2)
        psnr_exp_gnl1_list.append(psnr_exp_gnl1)
        psnr_exp_gnl2_list.append(psnr_exp_gnl2)
        psnr_logit_lnl1_list.append(psnr_logit_lnl1)
        psnr_logit_lnl2_list.append(psnr_logit_lnl2)
        psnr_logit_gnl1_list.append(psnr_logit_gnl1)
        psnr_logit_gnl2_list.append(psnr_logit_gnl2)
    if(len(psnr_logit_lnl1_list)):
        dump(psnr_list,psnr_outname)
        dump(psnr_exp_lnl1_list,psnr_exp_lnl1_outname)
        dump(psnr_exp_lnl2_list,psnr_exp_lnl2_outname)
        dump(psnr_exp_gnl1_list,psnr_exp_gnl1_outname)
        dump(psnr_exp_gnl2_list,psnr_exp_gnl2_outname)
        dump(psnr_logit_gnl1_list,psnr_logit_gnl1_outname)
        dump(psnr_logit_gnl2_list,psnr_logit_gnl2_outname)
        dump(psnr_logit_lnl1_list,psnr_logit_lnl1_outname)
        dump(psnr_logit_lnl2_list,psnr_logit_lnl2_outname)




    #psnr_command = ['./run_psnr.sh',ref_video,dis_vid,psnr_outname,dis_fps]
    #try:
    #subprocess.check_call(psnr_command)
    #subprocess.check_call(psnr_command)
    #except:
    #    return
    return

Parallel(n_jobs=60)(delayed(single_vid_psnr)(i) for i in range(len(upscaled_yuv_names)))
