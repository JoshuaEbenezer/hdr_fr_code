import numpy as np
import time
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

    if(nl_method=='two_exp'):
        avg = np.average(Y)
        Y_transform = np.exp(nl_param*(Y-avg)),np.exp(-nl_param*(Y-avg))
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

    if(nl_method=='two_exp'):
        h, w = np.shape(Y)
        avg_window = gen_gauss_window(31//2, 7.0/6.0)
        mu_image = np.zeros((h, w), dtype=np.float32)
        Y = np.array(Y).astype('float32')
        scipy.ndimage.correlate1d(Y, avg_window, 0, mu_image, mode='constant')
        scipy.ndimage.correlate1d(mu_image, avg_window, 1, mu_image, mode='constant')
        y1 = np.exp(nl_param* (Y- mu_image))
        y2 = np.exp(-nl_param*(Y- mu_image))
        Y_transform = y1,y2

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

def compute_speed(ref, ref_next, dis, dis_next, \
                             window):
    blk = 5;
    sigma_nsq = 0.1;
    times_to_down_size = 4; 

    #resize all frames
    for i in range(times_to_down_size):
        ref = np.array(cv2.resize(ref, None, fx=0.5, fy=0.5, \
                         interpolation=cv2.INTER_AREA),dtype=np.float32)
        ref_next = np.array(cv2.resize(ref_next, None, fx=0.5, fy=0.5, \
                              interpolation=cv2.INTER_AREA),dtype=np.float32)
        dis = np.array(cv2.resize(dis, None, fx=0.5, fy=0.5, \
                         interpolation=cv2.INTER_AREA),dtype=np.float32)
        dis_next = np.array(cv2.resize(dis_next, None, fx=0.5, fy=0.5, \
                              interpolation=cv2.INTER_AREA),dtype=np.float32)
    
    # calculate local averages    
    h, w = ref.shape
    mu_ref = np.zeros((h, w), dtype=np.float32)
    mu_dis = np.zeros((h, w), dtype=np.float32)
    
    scipy.ndimage.correlate1d(ref, window, 0, mu_ref, mode='reflect')
    scipy.ndimage.correlate1d(mu_ref, window, 1, mu_ref, mode='reflect')
    
    scipy.ndimage.correlate1d(dis, window, 0, mu_dis, mode='reflect')
    scipy.ndimage.correlate1d(mu_dis, window, 1, mu_dis, mode='reflect')
    
    # estimate local variances and conditional entropies in the spatial
    # domain for ith reference and distorted frames
    ss_ref, q_ref = est_params(ref - mu_ref, blk, sigma_nsq)
    spatial_ref = q_ref*np.log2(1+ss_ref)
    ss_dis, q_dis = est_params(dis - mu_dis, blk, sigma_nsq)
    spatial_dis = q_dis*np.log2(1+ss_dis)
    
    speed_s = np.nanmean(np.abs(spatial_ref.ravel() - spatial_dis.ravel()))
    speed_s_sn = np.abs(np.nanmean(spatial_ref.ravel() - spatial_dis.ravel()))
    
    ## frame differencing
    ref_diff = ref_next - ref;
    dis_diff = dis_next - dis;
    
    ## calculate local averages of frame differences
    mu_ref_diff = np.zeros((h, w), dtype=np.float32)
    mu_dis_diff = np.zeros((h, w), dtype=np.float32)
    
    scipy.ndimage.correlate1d(ref_diff, window, 0, mu_ref_diff, mode='reflect')
    scipy.ndimage.correlate1d(mu_ref_diff, window, 1, mu_ref_diff, mode='reflect')
    
    scipy.ndimage.correlate1d(dis_diff, window, 0, mu_dis_diff, mode='reflect')
    scipy.ndimage.correlate1d(mu_dis_diff, window, 1, mu_dis_diff, mode='reflect')
    
    """ Temporal SpEED
     estimate local variances and conditional entropies in the spatial
     domain for the reference and distorted frame differences """
     
    ss_ref_diff, q_ref = est_params(ref_diff - mu_ref_diff, blk, sigma_nsq)
    temporal_ref = q_ref*np.log2(1+ss_ref)*np.log2(1+ss_ref_diff)
    ss_dis_diff, q_dis = est_params(dis_diff - mu_dis_diff, blk, sigma_nsq)
    temporal_dis = q_dis*np.log2(1+ss_dis)*np.log2(1 + ss_dis_diff)
    
    speed_t = np.nanmean(np.abs(temporal_ref.ravel() - temporal_dis.ravel()));
    speed_t_sn = np.abs(np.nanmean(temporal_ref.ravel() - temporal_dis.ravel()));
    
    return speed_s, speed_s_sn, speed_t, speed_t_sn

def est_params(y, blk, sigma):
    """ 'ss' and 'ent' refer to the local variance parameter and the
        entropy at different locations of the subband
        y is a subband of the decomposition, 'blk' is the block size, 'sigma' is
        the neural noise variance """
    
    sizeim = np.floor(np.array(y.shape)/blk) * blk
    sizeim = sizeim.astype(int)
    y = y[:sizeim[0],:sizeim[1]].T
    
    temp = skimage.util.view_as_windows(np.ascontiguousarray(y), (blk,blk))\
    .reshape(-1,blk*blk).T
    
    cu = np.cov(temp, bias=1).astype(np.float32)
    
    eigval, eigvec = np.linalg.eig(cu)
    Q = np.matrix(eigvec)
    #L = diag(diag(L).*(diag(L)>0))*sum(diag(L))/(sum(diag(L).*(diag(L)>0))+(sum(diag(L).*(diag(L)>0))==0));
    L = np.matrix(np.diag(np.maximum(eigval, 0)))
    
    cu = Q*L*Q.T
    temp = skimage.util.view_as_blocks(np.ascontiguousarray(y), (blk,blk))\
    .reshape(-1,blk*blk).T
    
    L,Q = np.linalg.eigh(cu.astype(np.float64))
    L = L.astype(np.float32)
    #Estimate local variance parameters
    if np.max(L) > 0:
        ss = scipy.linalg.solve(cu, temp)
        ss = np.sum(ss*temp, axis=0)/(blk*blk)
        ss = ss.reshape((int(sizeim[1]/blk), int(sizeim[0]/blk))).T
    else:
        ss = np.zeros((sizeim/blk).astype(int),dtype=np.float32)
    
    L = L[L>0]
    
    #Compute entropy
    ent = np.zeros_like(ss, dtype=np.float32)
    for u in range(len(L)):
        ent += np.log2(ss*L[u]+sigma) + np.log(2*math.pi*np.exp(1));
        
    return ss, ent

def fread(fid, nelements, dtype):
     if dtype is str:
         dt = np.uint8  # WARNING: assuming 8-bit ASCII for np.str!
     else:
         dt = dtype

     data_array = np.fromfile(fid, dt, nelements)
     data_array.shape = (nelements, 1)

     return data_array


def single_vid_speed(i):
    dis_video_name = upscaled_yuv_names[i]
    if('ref' in dis_video_name):
        return
    content =content_list[i] 
    fps =fps_list[i] 
    ref_video_name = os.path.join('/media/josh-admin/seagate/fall2021_hdr_upscaled_yuv/4k_ref_'+content+'_upscaled.yuv')
    dis_video = open(os.path.join('/media/josh-admin/seagate/fall2021_hdr_upscaled_yuv',dis_video_name))

    ref_video = open(ref_video_name)

    width,height=int(3840),int(2160)
    speed_two_exp_lnl1_outname = os.path.join('./features/speed_features_local_two_exp1/',os.path.splitext(os.path.basename(dis_video_name))[0]+'.z')
    speed_two_exp_lnl2_outname = os.path.join('./features/speed_features_local_two_exp2/',os.path.splitext(os.path.basename(dis_video_name))[0]+'.z')
    speed_two_exp_lnl3_outname = os.path.join('./features/speed_features_local_two_exp3/',os.path.splitext(os.path.basename(dis_video_name))[0]+'.z')

    speed_two_exp_gnl1_outname = os.path.join('./features/speed_features_local_two_exp1/',os.path.splitext(os.path.basename(dis_video_name))[0]+'.z')
    speed_two_exp_gnl2_outname = os.path.join('./features/speed_features_local_two_exp2/',os.path.splitext(os.path.basename(dis_video_name))[0]+'.z')
    speed_two_exp_gnl3_outname = os.path.join('./features/speed_features_local_two_exp3/',os.path.splitext(os.path.basename(dis_video_name))[0]+'.z')

    if(os.path.exists(os.path.dirname(speed_two_exp_lnl1_outname))==False):
        os.mkdir(os.path.dirname(speed_two_exp_lnl1_outname))
    if(os.path.exists(os.path.dirname(speed_two_exp_lnl2_outname))==False):
        os.mkdir(os.path.dirname(speed_two_exp_lnl2_outname))
    if(os.path.exists(os.path.dirname(speed_two_exp_lnl3_outname))==False):
        os.mkdir(os.path.dirname(speed_two_exp_lnl3_outname))

    if(os.path.exists(os.path.dirname(speed_two_exp_gnl1_outname))==False):
        os.mkdir(os.path.dirname(speed_two_exp_gnl1_outname))
    if(os.path.exists(os.path.dirname(speed_two_exp_gnl2_outname))==False):
        os.mkdir(os.path.dirname(speed_two_exp_gnl2_outname))
    if(os.path.exists(os.path.dirname(speed_two_exp_gnl3_outname))==False):
        os.mkdir(os.path.dirname(speed_two_exp_gnl3_outname))

    if(os.path.exists(speed_two_exp_lnl1_outname)):
        return
    print(ref_video_name,dis_video_name,height,width,fps,speed_two_exp_lnl1_outname)
    speed_two_exp_lnl1_list = []
    speed_two_exp_lnl2_list = []
    speed_two_exp_lnl3_list = []

    speed_two_exp_gnl1_list = []
    speed_two_exp_gnl2_list = []
    speed_two_exp_gnl3_list = []

    avg_window = gen_gauss_window(3, 7.0/6.0)

    for framenum in range(framenos_list[i]-1):
        try:
            if(framenum==0):
                ref_y,_,_ =hdr_yuv_read(ref_video,framenum,height,width)
                ref_y = ref_y/1023.0


                # two_exp lnl
                ref_y_two_exp_lnl1a,ref_y_two_exp_lnl1b  = Y_compute_lnl(ref_y,nl_method='two_exp',nl_param=0.5) 
                ref_y_two_exp_lnl2a,ref_y_two_exp_lnl2b  = Y_compute_lnl(ref_y,nl_method='two_exp',nl_param=2) 
                ref_y_two_exp_lnl3a,ref_y_two_exp_lnl3b  = Y_compute_lnl(ref_y,nl_method='two_exp',nl_param=5) 



                # two exp gnl
                ref_y_two_exp_gnl1a,ref_y_two_exp_gnl1b  = Y_compute_gnl(ref_y,nl_method='two_exp',nl_param=0.5) 
                ref_y_two_exp_gnl2a,ref_y_two_exp_gnl2b  = Y_compute_gnl(ref_y,nl_method='two_exp',nl_param=2) 
                ref_y_two_exp_gnl3a,ref_y_two_exp_gnl3b  = Y_compute_gnl(ref_y,nl_method='two_exp',nl_param=5) 


                dis_y,_,_ =hdr_yuv_read(dis_video,framenum,height,width) 
                dis_y = dis_y/1023.0

                # two_exp lnl
                dis_y_two_exp_lnl1a,dis_y_two_exp_lnl1b  = Y_compute_lnl(dis_y,nl_method='two_exp',nl_param=0.5) 
                dis_y_two_exp_lnl2a,dis_y_two_exp_lnl2b  = Y_compute_lnl(dis_y,nl_method='two_exp',nl_param=2) 
                dis_y_two_exp_lnl3a,dis_y_two_exp_lnl3b  = Y_compute_lnl(dis_y,nl_method='two_exp',nl_param=5) 

    #            # two exp gnl
                dis_y_two_exp_gnl1a,dis_y_two_exp_gnl1b  = Y_compute_gnl(dis_y,nl_method='two_exp',nl_param=0.5) 
                dis_y_two_exp_gnl2a,dis_y_two_exp_gnl2b  = Y_compute_gnl(dis_y,nl_method='two_exp',nl_param=2) 
                dis_y_two_exp_gnl3a,dis_y_two_exp_gnl3b  = Y_compute_gnl(dis_y,nl_method='two_exp',nl_param=5) 

            ref_y_next,_,_ = hdr_yuv_read(ref_video,framenum+1,height,width) 
            ref_y_next = ref_y_next/1023.0

            # two exp lnl
            ref_y_lnl_two_exp1_nexta,ref_y_lnl_two_exp1_nextb = Y_compute_lnl(ref_y_next,nl_method='two_exp',nl_param=0.5) 
            ref_y_lnl_two_exp2_nexta,ref_y_lnl_two_exp2_nextb  = Y_compute_lnl(ref_y_next,nl_method='two_exp',nl_param=2) 
            ref_y_lnl_two_exp3_nexta,ref_y_lnl_two_exp3_nextb  = Y_compute_lnl(ref_y_next,nl_method='two_exp',nl_param=5) 
            

            # two_exp gnl
            ref_y_gnl_two_exp1_nexta,ref_y_gnl_two_exp1_nextb = Y_compute_gnl(ref_y_next,nl_method='two_exp',nl_param=0.5) 
            ref_y_gnl_two_exp2_nexta,ref_y_gnl_two_exp2_nextb  = Y_compute_gnl(ref_y_next,nl_method='two_exp',nl_param=2) 
            ref_y_gnl_two_exp3_nexta,ref_y_gnl_two_exp3_nextb  = Y_compute_gnl(ref_y_next,nl_method='two_exp',nl_param=5) 

            del ref_y_next

            dis_y_next,_,_ = hdr_yuv_read(dis_video,framenum+1,height,width) 
            dis_y_next = dis_y_next/1023.0
#            # two exp lnl
            dis_y_lnl_two_exp1_nexta,dis_y_lnl_two_exp1_nextb = Y_compute_lnl(dis_y_next,nl_method='two_exp',nl_param=0.5) 
            dis_y_lnl_two_exp2_nexta,dis_y_lnl_two_exp2_nextb  = Y_compute_lnl(dis_y_next,nl_method='two_exp',nl_param=2) 
            dis_y_lnl_two_exp3_nexta,dis_y_lnl_two_exp3_nextb  = Y_compute_lnl(dis_y_next,nl_method='two_exp',nl_param=5) 
            

            # two_exp gnl
            dis_y_gnl_two_exp1_nexta,dis_y_gnl_two_exp1_nextb = Y_compute_gnl(dis_y_next,nl_method='two_exp',nl_param=0.5) 
            dis_y_gnl_two_exp2_nexta,dis_y_gnl_two_exp2_nextb  = Y_compute_gnl(dis_y_next,nl_method='two_exp',nl_param=2) 
            dis_y_gnl_two_exp3_nexta,dis_y_gnl_two_exp3_nextb  = Y_compute_gnl(dis_y_next,nl_method='two_exp',nl_param=5) 

            del dis_y_next

#
#            print('distorted vid stats')
#            print(np.amin(dis_y_two_exp_lnl1a))
#            print(np.amin(dis_y_two_exp_lnl2a))
#            print(np.amin(dis_y_two_exp_lnl3a))
#            print(np.amin(dis_y_two_exp_gnl1a))
#            print(np.amin(dis_y_two_exp_gnl2a))
#            print(np.amin(dis_y_two_exp_gnl3a))
#
#            print(np.amax(dis_y_two_exp_lnl1a))
#            print(np.amax(dis_y_two_exp_lnl2a))
#            print(np.amax(dis_y_two_exp_lnl3a))
#            print(np.amax(dis_y_two_exp_gnl1a))
#            print(np.amax(dis_y_two_exp_gnl2a))
#            print(np.amax(dis_y_two_exp_gnl3a))
#


            speed_two_exp_lnl1a = compute_speed(ref_y_two_exp_lnl1a,ref_y_lnl_two_exp1_nexta,dis_y_two_exp_lnl1a,dis_y_lnl_two_exp1_nexta,avg_window)
            speed_two_exp_lnl2a = compute_speed(ref_y_two_exp_lnl2a,ref_y_lnl_two_exp2_nexta,dis_y_two_exp_lnl2a,dis_y_lnl_two_exp2_nexta,avg_window)
            speed_two_exp_lnl3a = compute_speed(ref_y_two_exp_lnl3a,ref_y_lnl_two_exp3_nexta,dis_y_two_exp_lnl3a,dis_y_lnl_two_exp3_nexta,avg_window)

            speed_two_exp_lnl1b = compute_speed(ref_y_two_exp_lnl1b,ref_y_lnl_two_exp1_nextb,dis_y_two_exp_lnl1b,dis_y_lnl_two_exp1_nextb,avg_window)
            speed_two_exp_lnl2b = compute_speed(ref_y_two_exp_lnl2b,ref_y_lnl_two_exp2_nextb,dis_y_two_exp_lnl2b,dis_y_lnl_two_exp2_nextb,avg_window)
            speed_two_exp_lnl3b = compute_speed(ref_y_two_exp_lnl3b,ref_y_lnl_two_exp3_nextb,dis_y_two_exp_lnl3b,dis_y_lnl_two_exp3_nextb,avg_window)

            speed_two_exp_gnl1a = compute_speed(ref_y_two_exp_gnl1a,ref_y_gnl_two_exp1_nexta,dis_y_two_exp_gnl1a,dis_y_gnl_two_exp1_nexta,avg_window)
            speed_two_exp_gnl2a = compute_speed(ref_y_two_exp_gnl2a,ref_y_gnl_two_exp2_nexta,dis_y_two_exp_gnl2a,dis_y_gnl_two_exp2_nexta,avg_window)
            speed_two_exp_gnl3a = compute_speed(ref_y_two_exp_gnl3a,ref_y_gnl_two_exp3_nexta,dis_y_two_exp_gnl3a,dis_y_gnl_two_exp3_nexta,avg_window)

            speed_two_exp_gnl1b = compute_speed(ref_y_two_exp_gnl1b,ref_y_gnl_two_exp1_nextb,dis_y_two_exp_gnl1b,dis_y_gnl_two_exp1_nextb,avg_window)
            speed_two_exp_gnl2b = compute_speed(ref_y_two_exp_gnl2b,ref_y_gnl_two_exp2_nextb,dis_y_two_exp_gnl2b,dis_y_gnl_two_exp2_nextb,avg_window)
            speed_two_exp_gnl3b = compute_speed(ref_y_two_exp_gnl3b,ref_y_gnl_two_exp3_nextb,dis_y_two_exp_gnl3b,dis_y_gnl_two_exp3_nextb,avg_window)

            speed_two_exp_lnl1_list.append(np.concatenate((speed_two_exp_lnl1a,speed_two_exp_lnl1b)))
            speed_two_exp_lnl2_list.append(np.concatenate((speed_two_exp_lnl2a,speed_two_exp_lnl2b)))
            speed_two_exp_lnl3_list.append(np.concatenate((speed_two_exp_lnl3a,speed_two_exp_lnl3b)))

            speed_two_exp_gnl1_list.append(np.concatenate((speed_two_exp_gnl1a,speed_two_exp_gnl1b)))
            speed_two_exp_gnl2_list.append(np.concatenate((speed_two_exp_gnl2a,speed_two_exp_gnl2b)))
            speed_two_exp_gnl3_list.append(np.concatenate((speed_two_exp_gnl3a,speed_two_exp_gnl3b)))

            ref_y_two_exp_lnl1a,ref_y_two_exp_lnl1b  =  ref_y_lnl_two_exp1_nexta,ref_y_lnl_two_exp1_nextb
            ref_y_two_exp_lnl2a,ref_y_two_exp_lnl2b  =  ref_y_lnl_two_exp2_nexta,ref_y_lnl_two_exp2_nextb
            ref_y_two_exp_lnl3a,ref_y_two_exp_lnl3b  =  ref_y_lnl_two_exp3_nexta,ref_y_lnl_two_exp3_nextb
                                                       
                                                                                                        
                                                       # two_exp gnl
            ref_y_two_exp_gnl1a,ref_y_two_exp_gnl1b  = ref_y_gnl_two_exp1_nexta,ref_y_gnl_two_exp1_nextb
            ref_y_two_exp_gnl2a,ref_y_two_exp_gnl2b  = ref_y_gnl_two_exp2_nexta,ref_y_gnl_two_exp2_nextb
            ref_y_two_exp_gnl3a,ref_y_two_exp_gnl3b  = ref_y_gnl_two_exp3_nexta,ref_y_gnl_two_exp3_nextb

            dis_y_two_exp_lnl1a,dis_y_two_exp_lnl1b  =dis_y_lnl_two_exp1_nexta,dis_y_lnl_two_exp1_nextb
            dis_y_two_exp_lnl2a,dis_y_two_exp_lnl2b  =dis_y_lnl_two_exp2_nexta,dis_y_lnl_two_exp2_nextb
            dis_y_two_exp_lnl3a,dis_y_two_exp_lnl3b  =dis_y_lnl_two_exp3_nexta,dis_y_lnl_two_exp3_nextb
                                                      
             # two exp gnl                                                                             
            dis_y_two_exp_gnl1a,dis_y_two_exp_gnl1b  =dis_y_gnl_two_exp1_nexta,dis_y_gnl_two_exp1_nextb
            dis_y_two_exp_gnl2a,dis_y_two_exp_gnl2b  =dis_y_gnl_two_exp2_nexta,dis_y_gnl_two_exp2_nextb
            dis_y_two_exp_gnl3a,dis_y_two_exp_gnl3b  =dis_y_gnl_two_exp3_nexta,dis_y_gnl_two_exp3_nextb
                                                      








        except Exception as e:
            print(e)
            if(len(speed_two_exp_lnl1_list)):
                dump(speed_two_exp_lnl1_list,speed_two_exp_lnl1_outname)
                dump(speed_two_exp_lnl2_list,speed_two_exp_lnl2_outname)
                dump(speed_two_exp_lnl3_list,speed_two_exp_lnl3_outname)

                dump(speed_two_exp_gnl1_list,speed_two_exp_gnl1_outname)
                dump(speed_two_exp_gnl2_list,speed_two_exp_gnl2_outname)
                dump(speed_two_exp_gnl3_list,speed_two_exp_gnl3_outname)
            break
    if(len(speed_two_exp_lnl1_list)):
        dump(speed_two_exp_lnl1_list,speed_two_exp_lnl1_outname)
        dump(speed_two_exp_lnl2_list,speed_two_exp_lnl2_outname)
        dump(speed_two_exp_lnl3_list,speed_two_exp_lnl3_outname)

        dump(speed_two_exp_gnl1_list,speed_two_exp_gnl1_outname)
        dump(speed_two_exp_gnl2_list,speed_two_exp_gnl2_outname)
        dump(speed_two_exp_gnl3_list,speed_two_exp_gnl3_outname)




    #speed_command = ['./run_speed.sh',ref_video,dis_vid,speed_outname,dis_fps]
    #try:
    #subprocess.check_call(speed_command)
    #subprocess.check_call(psnr_command)
    #except:
    #    return
    return

#single_vid_speed(40)
Parallel(n_jobs=10)(delayed(single_vid_speed)(i) for i in range(len(upscaled_yuv_names)))
