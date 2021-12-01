import numpy as np
import os
import glob
import cv2
from joblib import Parallel,delayed,dump
import scipy.ndimage
import pandas as pd
from skvideo.utils.mscn import gen_gauss_window
import skimage.util
import math
from hdr_utils import hdr_yuv_read

csv_file = '/home/labuser-admin/hdr/hdr_chipqa/fall2021_yuv_rw_info.csv'
csv_df = pd.read_csv(csv_file)
files = csv_df["yuv"]
ref_files = glob.glob('/mnt/b9f5646b-2c64-4699-8766-c4bba45fb442/fall2021_hdr_upscaled_yuv/4k_ref_*')
fps_list = csv_df["fps"]
framenos_list = csv_df["framenos"]
upscaled_yuv_names = [x[:-4]+'_upscaled.yuv' for x in csv_df['yuv']]
content_list = [f.split('_')[2] for f in upscaled_yuv_names]


def msssim(frame1, frame2, method='product'):

    extend_mode = 'constant'
    avg_window = np.array(gen_gauss_window(5, 1.5))
    K_1 = 0.01
    K_2 = 0.03
    level = 5
    weight1 = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    weight2 = weight1.copy()
    weight2 /= np.sum(weight2)

    downsample_filter = np.ones(2, dtype=np.float32)/2.0

    im1 = frame1.astype(np.float32)
    im2 = frame2.astype(np.float32)

    overall_mssim1 = []
    overall_mssim2 = []
    for i in range(level):
      mssim_array, ssim_map_array, mcs_array, cs_map_array = ssim_full(im1, im2, K_1 = K_1, K_2 = K_2, avg_window = avg_window)
      mssim_array = mssim_array[0]
      ssim_map_array = ssim_map_array[0]
      mcs_array = mcs_array[0]
      cs_map_array = cs_map_array[0]
      filtered_im1 = scipy.ndimage.correlate1d(im1, downsample_filter, 0)
      filtered_im1 = scipy.ndimage.correlate1d(filtered_im1, downsample_filter, 1)
      filtered_im1 = filtered_im1[1:, 1:]

      filtered_im2 = scipy.ndimage.correlate1d(im2, downsample_filter, 0)
      filtered_im2 = scipy.ndimage.correlate1d(filtered_im2, downsample_filter, 1)
      filtered_im2 = filtered_im2[1:, 1:]

      im1 = filtered_im1[::2, ::2]
      im2 = filtered_im2[::2, ::2]

      if i != level-1:
        overall_mssim1.append(mcs_array**weight1[i])
        overall_mssim2.append(mcs_array*weight2[i])

    if method == "product":
      overall_mssim = np.product(overall_mssim1) * mssim_array
    else:
      overall_mssim = np.sum(overall_mssim2) + mssim_array

    return overall_mssim

def Y_compute_gnl(Y,nl_method,nl_param):
    Y = Y.astype(np.float32)
    if(nl_method=='nakarushton'):
        Y_transform =  Y/(Y+avg_luminance) #
    elif(nl_method=='one_exp'):
        Y = Y/1023.0
        avg = np.average(Y)
        Y_transform = np.exp(nl_param*(Y-avg))
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
    elif(nl_method=='one_exp'):
        Y = Y/1023.0
        h, w = np.shape(Y)
        avg_window = gen_gauss_window(31//2, 7.0/6.0)
        mu_image = np.zeros((h, w), dtype=np.float32)
        Y= np.array(Y).astype('float32')
        scipy.ndimage.correlate1d(Y, avg_window, 0, mu_image, mode='constant')
        scipy.ndimage.correlate1d(mu_image, avg_window, 1, mu_image, mode='constant')
        Y_transform = np.exp(nl_param*(Y - mu_image))
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


def single_vid_msssim(i):
    dis_video_name = upscaled_yuv_names[i]
    if('ref' in dis_video_name):
        return
    content =content_list[i] 
    fps =fps_list[i] 
    ref_video_name = os.path.join('/mnt/b9f5646b-2c64-4699-8766-c4bba45fb442/fall2021_hdr_upscaled_yuv/4k_ref_'+content+'_upscaled.yuv')
    dis_video = open(os.path.join('/mnt/b9f5646b-2c64-4699-8766-c4bba45fb442/fall2021_hdr_upscaled_yuv',dis_video_name))
    ref_video = open(ref_video_name)

    width,height=int(3840),int(2160)
    msssim_outname = os.path.join('./features/msssim_features/',os.path.splitext(os.path.basename(dis_video_name))[0]+'.z')
    msssim_exp_gnl1_outname = os.path.join('./features/msssim_features_global_m_exp1/',os.path.splitext(os.path.basename(dis_video_name))[0]+'.z')
    msssim_exp_gnl2_outname = os.path.join('./features/msssim_features_global_m_exp2/',os.path.splitext(os.path.basename(dis_video_name))[0]+'.z')
    msssim_exp_lnl1_outname = os.path.join('./features/msssim_features_local_m_exp1/',os.path.splitext(os.path.basename(dis_video_name))[0]+'.z')
    msssim_exp_lnl2_outname = os.path.join('./features/msssim_features_local_m_exp2/',os.path.splitext(os.path.basename(dis_video_name))[0]+'.z')
    msssim_logit_gnl1_outname = os.path.join('./features/msssim_features_global_logit1/',os.path.splitext(os.path.basename(dis_video_name))[0]+'.z')
    msssim_logit_gnl2_outname = os.path.join('./features/msssim_features_global_logit2/',os.path.splitext(os.path.basename(dis_video_name))[0]+'.z')
    msssim_logit_lnl1_outname = os.path.join('./features/msssim_features_local_logit1/',os.path.splitext(os.path.basename(dis_video_name))[0]+'.z')
    msssim_logit_lnl2_outname = os.path.join('./features/msssim_features_local_logit2/',os.path.splitext(os.path.basename(dis_video_name))[0]+'.z')
    msssim_two_exp_gnl1_outname = os.path.join('./features/msssim_features_global_two_exp5e-1/',os.path.splitext(os.path.basename(dis_video_name))[0]+'.z')
    msssim_two_exp_gnl2_outname = os.path.join('./features/msssim_features_global_two_exp2/',os.path.splitext(os.path.basename(dis_video_name))[0]+'.z')
    msssim_two_exp_gnl3_outname = os.path.join('./features/msssim_features_global_two_exp5/',os.path.splitext(os.path.basename(dis_video_name))[0]+'.z')
    msssim_two_exp_lnl1_outname = os.path.join('./features/msssim_features_local_two_exp5e-1/',os.path.splitext(os.path.basename(dis_video_name))[0]+'.z')
    msssim_two_exp_lnl2_outname = os.path.join('./features/msssim_features_local_two_exp2/',os.path.splitext(os.path.basename(dis_video_name))[0]+'.z')
    msssim_two_exp_lnl3_outname = os.path.join('./features/msssim_features_local_two_exp5/',os.path.splitext(os.path.basename(dis_video_name))[0]+'.z')
    if(os.path.exists(msssim_exp_gnl1_outname)):
        return

    if(os.path.exists(os.path.dirname(msssim_outname))==False):
        os.mkdir(os.path.dirname(msssim_outname))
    if(os.path.exists(os.path.dirname(msssim_exp_gnl1_outname))==False):
        os.mkdir(os.path.dirname(msssim_exp_gnl1_outname))
    if(os.path.exists(os.path.dirname(msssim_exp_gnl2_outname))==False):
        os.mkdir(os.path.dirname(msssim_exp_gnl2_outname))
    if(os.path.exists(os.path.dirname(msssim_exp_lnl1_outname))==False):
        os.mkdir(os.path.dirname(msssim_exp_lnl1_outname))
    if(os.path.exists(os.path.dirname(msssim_exp_lnl2_outname))==False):
        os.mkdir(os.path.dirname(msssim_exp_lnl2_outname))

    if(os.path.exists(os.path.dirname(msssim_logit_gnl1_outname))==False):
        os.mkdir(os.path.dirname(msssim_logit_gnl1_outname))
    if(os.path.exists(os.path.dirname(msssim_logit_gnl2_outname))==False):
        os.mkdir(os.path.dirname(msssim_logit_gnl2_outname))
    if(os.path.exists(os.path.dirname(msssim_logit_lnl1_outname))==False):
        os.mkdir(os.path.dirname(msssim_logit_lnl1_outname))
    if(os.path.exists(os.path.dirname(msssim_logit_lnl2_outname))==False):
        os.mkdir(os.path.dirname(msssim_logit_lnl2_outname))

    if(os.path.exists(os.path.dirname(msssim_two_exp_gnl1_outname))==False):
        os.mkdir(os.path.dirname(msssim_two_exp_gnl1_outname))
    if(os.path.exists(os.path.dirname(msssim_two_exp_gnl2_outname))==False):
        os.mkdir(os.path.dirname(msssim_two_exp_gnl2_outname))
    if(os.path.exists(os.path.dirname(msssim_two_exp_gnl3_outname))==False):
        os.mkdir(os.path.dirname(msssim_two_exp_gnl3_outname))
    if(os.path.exists(os.path.dirname(msssim_two_exp_lnl1_outname))==False):
        os.mkdir(os.path.dirname(msssim_two_exp_lnl1_outname))
    if(os.path.exists(os.path.dirname(msssim_two_exp_lnl2_outname))==False):
        os.mkdir(os.path.dirname(msssim_two_exp_lnl2_outname))
    if(os.path.exists(os.path.dirname(msssim_two_exp_lnl3_outname))==False):
        os.mkdir(os.path.dirname(msssim_two_exp_lnl3_outname))

    print(ref_video_name,dis_video_name,height,width,fps,msssim_exp_gnl1_outname)
    msssim_list = []
    msssim_exp_lnl1_list = []
    msssim_exp_lnl2_list = []
    msssim_exp_gnl1_list = []
    msssim_exp_gnl2_list = []
    msssim_exp_lnl2_list = []
    msssim_logit_gnl1_list = []
    msssim_logit_gnl2_list = []
    msssim_logit_lnl1_list = []
    msssim_logit_lnl2_list = []
    msssim_two_exp_gnl1_list = []
    msssim_two_exp_gnl2_list = []
    msssim_two_exp_gnl3_list = []
    msssim_two_exp_lnl1_list = []
    msssim_two_exp_lnl2_list = []
    msssim_two_exp_lnl3_list = []


    for framenum in range(framenos_list[i]):
        try:
            ref_y,_,_ =hdr_yuv_read(ref_video,framenum,height,width)

            #exp local
            ref_y_lnl1 = Y_compute_lnl(ref_y,nl_method='exp',nl_param=1) 
            ref_y_lnl2 = Y_compute_lnl(ref_y,nl_method='exp',nl_param=2) 

            # logit lnl
            ref_y_logit_lnl1 = Y_compute_lnl(ref_y,nl_method='logit',nl_param=1) 
            ref_y_logit_lnl2 = Y_compute_lnl(ref_y,nl_method='logit',nl_param=2) 

            # one_exp lnl
            ref_y_one_exp_lnl1a = Y_compute_lnl(ref_y,nl_method='one_exp',nl_param=0.5) 
            ref_y_one_exp_lnl2a = Y_compute_lnl(ref_y,nl_method='one_exp',nl_param=2) 
            ref_y_one_exp_lnl3a = Y_compute_lnl(ref_y,nl_method='one_exp',nl_param=5) 

            ref_y_one_exp_lnl1b = Y_compute_lnl(ref_y,nl_method='one_exp',nl_param=-0.5) 
            ref_y_one_exp_lnl2b = Y_compute_lnl(ref_y,nl_method='one_exp',nl_param=-2) 
            ref_y_one_exp_lnl3b = Y_compute_lnl(ref_y,nl_method='one_exp',nl_param=-5) 

            # exp gnl
            ref_y_gnl1 = Y_compute_gnl(ref_y,nl_method='exp',nl_param=1) 
            ref_y_gnl2 = Y_compute_gnl(ref_y,nl_method='exp',nl_param=2) 

            # logit gnl
            ref_y_gnl_logit1 = Y_compute_gnl(ref_y,nl_method='logit',nl_param=1) 
            ref_y_gnl_logit2 = Y_compute_gnl(ref_y,nl_method='logit',nl_param=2) 

            # one_exp gnl
            ref_y_one_exp_gnl1a = Y_compute_gnl(ref_y,nl_method='one_exp',nl_param=0.5) 
            ref_y_one_exp_gnl2a = Y_compute_gnl(ref_y,nl_method='one_exp',nl_param=2) 
            ref_y_one_exp_gnl2a = Y_compute_gnl(ref_y,nl_method='one_exp',nl_param=5) 

            ref_y_one_exp_gnl1b = Y_compute_gnl(ref_y,nl_method='one_exp',nl_param=-0.5) 
            ref_y_one_exp_gnl2b = Y_compute_gnl(ref_y,nl_method='one_exp',nl_param=-2) 
            ref_y_one_exp_gnl3b = Y_compute_gnl(ref_y,nl_method='one_exp',nl_param=-5) 


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

            # one_exp lnl
            dis_y_one_exp_lnl1a = Y_compute_lnl(dis_y,nl_method='one_exp',nl_param=0.5) 
            dis_y_one_exp_lnl2a = Y_compute_lnl(dis_y,nl_method='one_exp',nl_param=2) 
            dis_y_one_exp_lnl3a = Y_compute_lnl(dis_y,nl_method='one_exp',nl_param=5) 

            dis_y_one_exp_lnl1b = Y_compute_lnl(dis_y,nl_method='one_exp',nl_param=-0.5) 
            dis_y_one_exp_lnl2b = Y_compute_lnl(dis_y,nl_method='one_exp',nl_param=-2) 
            dis_y_one_exp_lnl3b = Y_compute_lnl(dis_y,nl_method='one_exp',nl_param=-5) 
            # one_exp gnl
            dis_y_one_exp_gnl1a = Y_compute_gnl(dis_y,nl_method='one_exp',nl_param=0.5) 
            dis_y_one_exp_gnl2a = Y_compute_gnl(dis_y,nl_method='one_exp',nl_param=2) 
            dis_y_one_exp_gnl3a = Y_compute_lnl(dis_y,nl_method='one_exp',nl_param=5) 

            dis_y_one_exp_gnl1b = Y_compute_gnl(dis_y,nl_method='one_exp',nl_param=-0.5) 
            dis_y_one_exp_gnl2b = Y_compute_gnl(dis_y,nl_method='one_exp',nl_param=-2) 
            dis_y_one_exp_gnl3b = Y_compute_lnl(dis_y,nl_method='one_exp',nl_param=-5) 


        except Exception as e:
            print(e)
            if(len(msssim_list)):
                dump(msssim_list,msssim_outname)
                dump(msssim_exp_lnl1_list,msssim_exp_lnl1_outname)
                dump(msssim_exp_lnl2_list,msssim_exp_lnl2_outname)
                dump(msssim_exp_gnl1_list,msssim_exp_gnl1_outname)
                dump(msssim_exp_gnl2_list,msssim_exp_gnl2_outname)
                dump(msssim_logit_gnl1_list,msssim_logit_gnl1_outname)
                dump(msssim_logit_gnl2_list,msssim_logit_gnl2_outname)
                dump(msssim_logit_lnl1_list,msssim_logit_lnl1_outname)
                dump(msssim_logit_lnl2_list,msssim_logit_lnl2_outname)

                dump(msssim_two_exp_gnl1_list,msssim_two_exp_gnl1_outname)
                dump(msssim_two_exp_gnl2_list,msssim_two_exp_gnl2_outname)
                dump(msssim_two_exp_gnl3_list,msssim_two_exp_gnl3_outname)
                dump(msssim_two_exp_lnl1_list,msssim_two_exp_lnl1_outname)
                dump(msssim_two_exp_lnl2_list,msssim_two_exp_lnl2_outname)
                dump(msssim_two_exp_lnl3_list,msssim_two_exp_lnl3_outname)
            break
        try:
            msssim_val = msssim(ref_y,dis_y)
            msssim_exp_lnl1 = msssim(ref_y_lnl1,dis_y_lnl1)
            msssim_exp_lnl2 = msssim(ref_y_lnl2,dis_y_lnl2)

            msssim_exp_gnl1 = msssim(ref_y_gnl1,dis_y_gnl1)
            msssim_exp_gnl2 = msssim(ref_y_gnl2,dis_y_gnl2)
            
            msssim_logit_gnl1 = msssim(ref_y_gnl_logit1,dis_y_gnl_logit1)
            msssim_logit_gnl2 = msssim(ref_y_gnl_logit2,dis_y_gnl_logit2)
            
            msssim_logit_lnl1 = msssim(ref_y_logit_lnl1,dis_y_lnl_logit1)
            msssim_logit_lnl2 = msssim(ref_y_logit_lnl2,dis_y_lnl_logit2)

            msssim_one_exp_gnl1a = msssim(ref_y_gnl_one_exp1a,dis_y_gnl_one_exp1a)
            msssim_one_exp_gnl2a = msssim(ref_y_gnl_one_exp2a,dis_y_gnl_one_exp2a)
            msssim_one_exp_gnl3a = msssim(ref_y_gnl_one_exp3a,dis_y_gnl_one_exp3a)
            
            msssim_one_exp_lnl1a = msssim(ref_y_one_exp_lnl1a,dis_y_lnl_one_exp1a)
            msssim_one_exp_lnl2a = msssim(ref_y_one_exp_lnl2a,dis_y_lnl_one_exp2a)
            msssim_one_exp_lnl3a = msssim(ref_y_one_exp_lnl3a,dis_y_lnl_one_exp3a)

            msssim_one_exp_gnl1b = msssim(ref_y_gnl_one_exp1b,dis_y_gnl_one_exp1b)
            msssim_one_exp_gnl2b = msssim(ref_y_gnl_one_exp2b,dis_y_gnl_one_exp2b)
            msssim_one_exp_gnl3b = msssim(ref_y_gnl_one_exp3b,dis_y_gnl_one_exp3b)
            
            msssim_one_exp_lnl1b = msssim(ref_y_one_exp_lnl1b,dis_y_lnl_one_exp1b)
            msssim_one_exp_lnl2b = msssim(ref_y_one_exp_lnl2b,dis_y_lnl_one_exp2b)
            msssim_one_exp_lnl3b = msssim(ref_y_one_exp_lnl3b,dis_y_lnl_one_exp3b)

            msssim_two_exp_gnl1 = np.concatenate((msssim_one_exp_gnl1a,msssim_one_exp_gnl1b))
            msssim_two_exp_gnl2 = np.concatenate((msssim_one_exp_gnl2a,msssim_one_exp_gnl2b))
            msssim_two_exp_gnl3 = np.concatenate((msssim_one_exp_gnl3a,msssim_one_exp_gnl3b))

            msssim_two_exp_lnl1 = np.concatenate((msssim_one_exp_lnl1a,msssim_one_exp_lnl1b))
            msssim_two_exp_lnl2 = np.concatenate((msssim_one_exp_lnl2a,msssim_one_exp_lnl2b))
            msssim_two_exp_lnl3 = np.concatenate((msssim_one_exp_lnl3a,msssim_one_exp_lnl3b))

            msssim_list.append(msssim_val)
            msssim_exp_lnl1_list.append(msssim_exp_lnl1)
            msssim_exp_lnl2_list.append(msssim_exp_lnl2)
            msssim_exp_gnl1_list.append(msssim_exp_gnl1)
            msssim_exp_gnl2_list.append(msssim_exp_gnl2)
            msssim_logit_lnl1_list.append(msssim_logit_lnl1)
            msssim_logit_lnl2_list.append(msssim_logit_lnl2)
            msssim_logit_gnl1_list.append(msssim_logit_gnl1)
            msssim_logit_gnl2_list.append(msssim_logit_gnl2)

            msssim_two_exp_lnl1_list.append(msssim_two_exp_lnl1)
            msssim_two_exp_lnl2_list.append(msssim_two_exp_lnl2)
            msssim_two_exp_lnl3_list.append(msssim_two_exp_lnl3)
            msssim_two_exp_gnl1_list.append(msssim_two_exp_gnl1)
            msssim_two_exp_gnl2_list.append(msssim_two_exp_gnl2)
            msssim_two_exp_gnl3_list.append(msssim_two_exp_gnl3)
        except:
            msssim_logit_lnl1_list = []
            break
    if(len(msssim_logit_lnl1_list)):
        dump(msssim_list,msssim_outname)
        dump(msssim_exp_lnl1_list,msssim_exp_lnl1_outname)
        dump(msssim_exp_lnl2_list,msssim_exp_lnl2_outname)
        dump(msssim_exp_gnl1_list,msssim_exp_gnl1_outname)
        dump(msssim_exp_gnl2_list,msssim_exp_gnl2_outname)
        dump(msssim_logit_gnl1_list,msssim_logit_gnl1_outname)
        dump(msssim_logit_gnl2_list,msssim_logit_gnl2_outname)
        dump(msssim_logit_lnl1_list,msssim_logit_lnl1_outname)
        dump(msssim_logit_lnl2_list,msssim_logit_lnl2_outname)

        dump(msssim_two_exp_gnl1_list,msssim_two_exp_gnl1_outname)
        dump(msssim_two_exp_gnl2_list,msssim_two_exp_gnl2_outname)
        dump(msssim_two_exp_gnl3_list,msssim_two_exp_gnl3_outname)
        dump(msssim_two_exp_lnl1_list,msssim_two_exp_lnl1_outname)
        dump(msssim_two_exp_lnl2_list,msssim_two_exp_lnl2_outname)
        dump(msssim_two_exp_lnl3_list,msssim_two_exp_lnl3_outname)




    #msssim_command = ['./run_msssim.sh',ref_video,dis_vid,msssim_outname,dis_fps]
    #try:
    #subprocess.check_call(msssim_command)
    #subprocess.check_call(psnr_command)
    #except:
    #    return
    return

Parallel(n_jobs=30)(delayed(single_vid_msssim)(i) for i in range(len(upscaled_yuv_names)))
