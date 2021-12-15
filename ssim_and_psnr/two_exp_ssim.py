import numpy as np
from skimage.metrics import structural_similarity as ssim
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

def gen_gauss_window(lw, sigma):
    sd = np.float32(sigma)
    lw = int(lw)
    weights = [0.0] * (2 * lw + 1)
    weights[lw] = 1.0
    sum = 1.0
    sd *= sd
    for ii in range(1, lw + 1):
        tmp = np.exp(-0.5 * np.float32(ii * ii) / sd)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
        sum += 2.0 * tmp
    for ii in range(2 * lw + 1):
        weights[ii] /= sum
    return weights

def Y_compute_gnl(Y,nl_method,nl_param):
    Y = Y.astype(np.float32)
    if(nl_method=='one_exp'):
        Y = Y/1023.0
        avg = np.average(Y)
        Y_transform = np.exp(nl_param*(Y-avg))
    elif(nl_method=='nakarushton'):
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

    if(nl_method=='one_exp'):
        h, w = np.shape(Y)
        Y=  Y/1023.0
        avg_window = gen_gauss_window(31//2, 7.0/6.0)
        mu_image = np.zeros((h, w), dtype=np.float32)
        Y= np.array(Y).astype('float32')
        scipy.ndimage.correlate1d(Y, avg_window, 0, mu_image, mode='constant')
        scipy.ndimage.correlate1d(mu_image, avg_window, 1, mu_image, mode='constant')
        Y_transform = np.exp(nl_param*(Y- mu_image))
    elif(nl_method=='logit'):
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


def single_vid_ssim(i):
    dis_video_name = upscaled_yuv_names[i]
    if('ref' in dis_video_name):
        return
    content =content_list[i] 
    fps =fps_list[i] 
    ref_video_name = os.path.join('/media/josh-admin/seagate/fall2021_hdr_upscaled_yuv/4k_ref_'+content+'_upscaled.yuv')
    dis_video = open(os.path.join('/media/josh-admin/seagate/fall2021_hdr_upscaled_yuv',dis_video_name))
    ref_video = open(ref_video_name)

    width,height=int(3840),int(2160)
    ssim_exp_gnl1_outname = os.path.join('./features/ssim_features_global_t_exp5e-1/',os.path.splitext(os.path.basename(dis_video_name))[0]+'.z')
    ssim_exp_gnl2_outname = os.path.join('./features/ssim_features_global_t_exp2/',os.path.splitext(os.path.basename(dis_video_name))[0]+'.z')
    ssim_exp_gnl3_outname = os.path.join('./features/ssim_features_global_t_exp5/',os.path.splitext(os.path.basename(dis_video_name))[0]+'.z')

    ssim_exp_lnl1_outname = os.path.join('./features/ssim_features_local_t_exp5e-1/',os.path.splitext(os.path.basename(dis_video_name))[0]+'.z')
    ssim_exp_lnl2_outname = os.path.join('./features/ssim_features_local_t_exp2/',os.path.splitext(os.path.basename(dis_video_name))[0]+'.z')
    ssim_exp_lnl3_outname = os.path.join('./features/ssim_features_local_t_exp5/',os.path.splitext(os.path.basename(dis_video_name))[0]+'.z')

    if(os.path.exists(os.path.dirname(ssim_exp_gnl1_outname))==False):
        os.mkdir(os.path.dirname(ssim_exp_gnl1_outname))
    if(os.path.exists(os.path.dirname(ssim_exp_gnl2_outname))==False):
        os.mkdir(os.path.dirname(ssim_exp_gnl2_outname))
    if(os.path.exists(os.path.dirname(ssim_exp_gnl3_outname))==False):
        os.mkdir(os.path.dirname(ssim_exp_gnl3_outname))

    if(os.path.exists(os.path.dirname(ssim_exp_lnl1_outname))==False):
        os.mkdir(os.path.dirname(ssim_exp_lnl1_outname))
    if(os.path.exists(os.path.dirname(ssim_exp_lnl2_outname))==False):
        os.mkdir(os.path.dirname(ssim_exp_lnl2_outname))
    if(os.path.exists(os.path.dirname(ssim_exp_lnl3_outname))==False):
        os.mkdir(os.path.dirname(ssim_exp_lnl3_outname))

    if(os.path.exists(ssim_exp_gnl1_outname)):
        return
    print(ref_video_name,dis_video_name,height,width,fps,ssim_exp_gnl1_outname)
    ssim_exp_lnl1_list = []
    ssim_exp_lnl2_list = []
    ssim_exp_lnl3_list = []
    ssim_exp_gnl1_list = []
    ssim_exp_gnl2_list = []
    ssim_exp_gnl3_list = []


    for framenum in range(framenos_list[i]):
        try:
            ref_y,_,_ =hdr_yuv_read(ref_video,framenum,height,width)

            #exp local
            ref_y_lnl1a = Y_compute_lnl(ref_y,nl_method='one_exp',nl_param=0.5) 
            ref_y_lnl2a = Y_compute_lnl(ref_y,nl_method='one_exp',nl_param=2) 
            ref_y_lnl3a = Y_compute_lnl(ref_y,nl_method='one_exp',nl_param=5) 


            ref_y_gnl1a = Y_compute_gnl(ref_y,nl_method='one_exp',nl_param=0.5) 
            ref_y_gnl2a = Y_compute_gnl(ref_y,nl_method='one_exp',nl_param=2) 
            ref_y_gnl3a = Y_compute_gnl(ref_y,nl_method='one_exp',nl_param=5) 

            ref_y_lnl1b = Y_compute_lnl(ref_y,nl_method='one_exp',nl_param=-0.5) 
            ref_y_lnl2b = Y_compute_lnl(ref_y,nl_method='one_exp',nl_param=-2) 
            ref_y_lnl3b = Y_compute_lnl(ref_y,nl_method='one_exp',nl_param=-5) 

            ref_y_gnl1b = Y_compute_gnl(ref_y,nl_method='one_exp',nl_param=-0.5) 
            ref_y_gnl2b = Y_compute_gnl(ref_y,nl_method='one_exp',nl_param=-2) 
            ref_y_gnl3b = Y_compute_gnl(ref_y,nl_method='one_exp',nl_param=-5) 


            dis_y,_,_ =hdr_yuv_read(dis_video,framenum,height,width) 

            dis_y_lnl1a = Y_compute_lnl(dis_y,nl_method='one_exp',nl_param=0.5) 
            dis_y_lnl2a = Y_compute_lnl(dis_y,nl_method='one_exp',nl_param=2) 
            dis_y_lnl3a = Y_compute_lnl(dis_y,nl_method='one_exp',nl_param=5) 


            dis_y_gnl1a = Y_compute_gnl(dis_y,nl_method='one_exp',nl_param=0.5) 
            dis_y_gnl2a = Y_compute_gnl(dis_y,nl_method='one_exp',nl_param=2) 
            dis_y_gnl3a = Y_compute_gnl(dis_y,nl_method='one_exp',nl_param=5) 

            dis_y_lnl1b = Y_compute_lnl(dis_y,nl_method='one_exp',nl_param=-0.5) 
            dis_y_lnl2b = Y_compute_lnl(dis_y,nl_method='one_exp',nl_param=-2) 
            dis_y_lnl3b = Y_compute_lnl(dis_y,nl_method='one_exp',nl_param=-5) 

            dis_y_gnl1b = Y_compute_gnl(dis_y,nl_method='one_exp',nl_param=-0.5) 
            dis_y_gnl2b = Y_compute_gnl(dis_y,nl_method='one_exp',nl_param=-2) 
            dis_y_gnl3b = Y_compute_gnl(dis_y,nl_method='one_exp',nl_param=-5) 




        except Exception as e:
            print(e)
            if(len(ssim_exp_gnl1_list)):
                dump(ssim_exp_lnl1_list,ssim_exp_lnl1_outname)
                dump(ssim_exp_lnl2_list,ssim_exp_lnl2_outname)
                dump(ssim_exp_lnl3_list,ssim_exp_lnl3_outname)

                dump(ssim_exp_gnl1_list,ssim_exp_gnl1_outname)
                dump(ssim_exp_gnl2_list,ssim_exp_gnl2_outname)
                dump(ssim_exp_gnl3_list,ssim_exp_gnl3_outname)
            break
        ssim_exp_lnl1a = ssim(ref_y_lnl1a,dis_y_lnl1a)
        ssim_exp_lnl2a = ssim(ref_y_lnl2a,dis_y_lnl2a)
        ssim_exp_lnl3a = ssim(ref_y_lnl3a,dis_y_lnl3a)

        ssim_exp_gnl1a = ssim(ref_y_gnl1a,dis_y_gnl1a)
        ssim_exp_gnl2a = ssim(ref_y_gnl2a,dis_y_gnl2a)
        ssim_exp_gnl3a = ssim(ref_y_gnl3a,dis_y_gnl3a)

        ssim_exp_lnl1b = ssim(ref_y_lnl1b,dis_y_lnl1b) 
        ssim_exp_lnl2b = ssim(ref_y_lnl2b,dis_y_lnl2b) 
        ssim_exp_lnl3b = ssim(ref_y_lnl3b,dis_y_lnl3b) 

        ssim_exp_gnl1b = ssim(ref_y_gnl1b,dis_y_gnl1b)
        ssim_exp_gnl2b = ssim(ref_y_gnl2b,dis_y_gnl2b)
        ssim_exp_gnl3b = ssim(ref_y_gnl3b,dis_y_gnl3b)
        

        ssim_exp_lnl1_list.append([ssim_exp_lnl1a, ssim_exp_lnl1b]) 
        ssim_exp_lnl2_list.append([ssim_exp_lnl2a, ssim_exp_lnl2b])
        ssim_exp_lnl3_list.append([ssim_exp_lnl3a, ssim_exp_lnl3b])

        ssim_exp_gnl1_list.append([ssim_exp_gnl1a, ssim_exp_gnl1b])
        ssim_exp_gnl2_list.append([ssim_exp_gnl2a, ssim_exp_gnl2b])
        ssim_exp_gnl3_list.append([ssim_exp_gnl3a, ssim_exp_gnl3b])

    if(len(ssim_exp_lnl1_list)):

        dump(ssim_exp_lnl1_list,ssim_exp_lnl1_outname)
        dump(ssim_exp_lnl2_list,ssim_exp_lnl2_outname)
        dump(ssim_exp_lnl3_list,ssim_exp_lnl3_outname)

        dump(ssim_exp_gnl1_list,ssim_exp_gnl1_outname)
        dump(ssim_exp_gnl2_list,ssim_exp_gnl2_outname)
        dump(ssim_exp_gnl3_list,ssim_exp_gnl3_outname)




#    #ssim_command = ['./run_ssim.sh',ref_video,dis_vid,ssim_outname,dis_fps]
#    #try:
#    #subprocess.check_call(ssim_command)
#    #subprocess.check_call(ssim_command)
#    #except:
#    #    return
    return

Parallel(n_jobs=60)(delayed(single_vid_ssim)(i) for i in range(len(upscaled_yuv_names)))
