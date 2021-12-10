import numpy as np
import os
import glob
import subprocess
from joblib import Parallel,delayed,dump
import scipy.ndimage
import pandas as pd
import scipy.linalg
from strred_utils import *
from hdr_utils import hdr_yuv_read

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
        Y = Y/1023.0
        avg = np.average(Y)
        Y_transform = np.exp(nl_param*(Y-avg)),np.exp(-nl_param*(Y-avg))
    elif(nl_method=='nakarushton'):
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

    if(nl_method=='two_exp'):
        Y = Y/1023.0
        h, w = np.shape(Y)
        avg_window = gen_gauss_window(31//2, 7.0/6.0)
        mu_image = np.zeros((h, w), dtype=np.float32)
        Y = np.array(Y).astype('float32')
        scipy.ndimage.correlate1d(Y, avg_window, 0, mu_image, mode='constant')
        scipy.ndimage.correlate1d(mu_image, avg_window, 1, mu_image, mode='constant')
        y1 = np.exp(nl_param* (Y- mu_image))
        y2 = np.exp(-nl_param*(Y- mu_image))
        Y_transform = y1,y2

    elif(nl_method=='one_exp'):
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

def est_params(frame, blk, sigma_nn):
    h, w = frame.shape
    sizeim = np.floor(np.array(frame.shape)/blk) * blk
    sizeim = sizeim.astype(np.int)

    frame = frame[:sizeim[0], :sizeim[1]]

    #paired_products
    temp = []
    for u in range(blk):
      for v in range(blk):
        temp.append(np.ravel(frame[v:(sizeim[0]-(blk-v)+1), u:(sizeim[1]-(blk-u)+1)]))
    temp = np.array(temp).astype(np.float32)

    cov_mat = np.cov(temp, bias=1).astype(np.float32)

    # force PSD
    eigval, eigvec = np.linalg.eig(cov_mat)
    Q = np.matrix(eigvec)
    xdiag = np.matrix(np.diag(np.maximum(eigval, 0)))
    cov_mat = Q*xdiag*Q.T

    temp = []
    for u in range(blk):
      for v in range(blk):
        temp.append(np.ravel(frame[v::blk, u::blk]))
    temp = np.array(temp).astype(np.float32)

    # float32 vs float64 difference between python2 and python3
    # avoiding this problem with quick cast to float64
    V,d = scipy.linalg.eigh(cov_mat.astype(np.float64))
    V = V.astype(np.float32)

    # Estimate local variance
    sizeim_reduced = (sizeim/blk).astype(np.int)
    ss = np.zeros((sizeim_reduced[0], sizeim_reduced[1]), dtype=np.float32)
    if np.max(V) > 0:
      # avoid the matrix inverse for extra strred/accuracy
      ss = scipy.linalg.solve(cov_mat, temp)
      ss = np.sum(np.multiply(ss, temp) / (blk**2), axis=0)
      ss = ss.reshape(sizeim_reduced)

    V = V[V>0]

    # Compute entropy
    ent = np.zeros_like(ss, dtype=np.float32)
    for u in range(V.shape[0]):
      ent += np.log2(ss * V[u] + sigma_nn) + np.log(2*np.pi*np.exp(1))


    return ss, ent


def extract_info(frame1, frame2):
    blk = 3
    sigma_nsq = 0.1
    sigma_nsqt = 0.1

    model = SpatialSteerablePyramid(height=6)
    y1 = model.extractSingleBand(frame1, filtfile="sp5Filters", band=0, level=4)
    y2 = model.extractSingleBand(frame2, filtfile="sp5Filters", band=0, level=4)

    ydiff = y1 - y2

    ss, q = est_params(y1, blk, sigma_nsq)
    ssdiff, qdiff = est_params(ydiff, blk, sigma_nsqt)


    spatial = np.multiply(q, np.log2(1 + ss))
    temporal = np.multiply(qdiff, np.multiply(np.log2(1 + ss), np.log2(1 + ssdiff)))

    return spatial, temporal
def compute_strred(refFrame1,refFrame2,disFrame1,disFrame2):
    """Computes Spatio-Temporal Reduced Reference Entropic Differencing (ST-RRED) Index. [#f1]_

    Both video inputs are compared over frame differences, with quality determined by
    differences in the entropy per subband.

    Parameters
    ----------
    referenceVideoData : ndarray
        Reference video, ndarray of dimension (T, M, N, C), (T, M, N), (M, N, C), or (M, N),
        where T is the number of frames, M is the height, N is width,
        and C is number of channels. Here C is only allowed to be 1.

    distortedVideoData : ndarray
        Distorted video, ndarray of dimension (T, M, N, C), (T, M, N), (M, N, C), or (M, N),
        where T is the number of frames, M is the height, N is width,
        and C is number of channels. Here C is only allowed to be 1.

    Returns
    -------
    strred_array : ndarray
        The ST-RRED results, ndarray of dimension ((T-1)/2, 4), where T
        is the number of frames.  Each row holds spatial score, temporal score,
        reduced reference spatial score, and reduced reference temporal score.

    strred : float
        The final ST-RRED score if all blocks are averaged after comparing
        reference and distorted data. This is close to full-reference.

    strredssn : float
        The final ST-RRED score if all blocks are averaged before comparing
        reference and distorted data. This is the reduced reference score.

    References
    ----------

    .. [#f1] R. Soundararajan and A. C. Bovik, "Video Quality Assessment by Reduced Reference Spatio-temporal Entropic Differencing," IEEE Transactions on Circuits and Systems for Video Technology, April 2013.

    """


    rreds = []
    rredt = []

    rredssn = []
    rredtsn = []


    spatialRef, temporalRef = extract_info(refFrame1, refFrame2)
    spatialDis, temporalDis = extract_info(disFrame1, disFrame2)

    rreds.append(np.mean(np.abs(spatialRef - spatialDis)))
    rredt.append(np.mean(np.abs(temporalRef - temporalDis)))


    rreds = np.array(rreds)
    rredt = np.array(rredt)

    srred = np.mean(rreds)
    trred = np.mean(rredt)
    strred = srred * trred

    return strred
def fread(fid, nelements, dtype):
     if dtype is str:
         dt = np.uint8  # WARNING: assuming 8-bit ASCII for np.str!
     else:
         dt = dtype

     data_array = np.fromfile(fid, dt, nelements)
     data_array.shape = (nelements, 1)

     return data_array

def y4mFileRead(filePath,width, height,startFrame):
#    """Cut the YUV file at startFrame position for numFrame frames"""
    oneFrameNumBytes = int(width*height*1.5)
    with open(filePath, 'r+b') as file1:

        # header info
        line1 = file1.readline()

        # string of FRAME 
        line2 = file1.readline()

        frameByteOffset = len(line1)+(len(line2)+oneFrameNumBytes) * startFrame

        # each frame begins with the 5 bytes 'FRAME' followed by some zero or more characters"
        bytesToRead = oneFrameNumBytes + len(line2)

        file1.seek(frameByteOffset)
        y1 = fread(file1,height*width,np.uint8)
        y = np.reshape(y1,(height,width))
        return np.expand_dims(y,2)

dis_metadata_csv = pd.read_csv("/home/josh-admin/hdr/qa/hdr_chipqa/fall2021_yuv_rw_info.csv")
print([i for i in dis_metadata_csv["yuv"]])
framenos_list = dis_metadata_csv["framenos"]

outfolder = './hdr_rred_features'

def single_vid_strred(i):
    dis_basename = dis_metadata_csv['yuv'].iloc[i][:-4]+'_upscaled.yuv'
    if('ref' in dis_basename):
        return
    dis_vid = os.path.join("/media/josh-admin/seagate/fall2021_hdr_upscaled_yuv",dis_basename)
    print(dis_vid)
    if(os.path.exists(dis_vid)==False):
        print('does not exist')
        return
    dis_fps = str(dis_metadata_csv['fps'].iloc[i])

    content = dis_basename.split('_')[2]
    ref_basename = '4k_ref_'+content+'_upscaled.yuv'
    ref_fps = dis_fps
    ref_filename = os.path.join("/media/josh-admin/seagate/fall2021_hdr_upscaled_yuv/",ref_basename)

    width,height=int(3840),int(2160)

    strred_two_exp_lnl1_outname = os.path.join('./hdr_rred_features/strred_features_local_two_exp1/',os.path.splitext(os.path.basename(dis_basename))[0]+'.z')
    strred_two_exp_lnl2_outname = os.path.join('./hdr_rred_features/strred_features_local_two_exp2/',os.path.splitext(os.path.basename(dis_basename))[0]+'.z')
    strred_two_exp_lnl3_outname = os.path.join('./hdr_rred_features/strred_features_local_two_exp3/',os.path.splitext(os.path.basename(dis_basename))[0]+'.z')

    strred_two_exp_gnl1_outname = os.path.join('./hdr_rred_features/strred_features_local_two_exp1/',os.path.splitext(os.path.basename(dis_basename))[0]+'.z')
    strred_two_exp_gnl2_outname = os.path.join('./hdr_rred_features/strred_features_local_two_exp2/',os.path.splitext(os.path.basename(dis_basename))[0]+'.z')
    strred_two_exp_gnl3_outname = os.path.join('./hdr_rred_features/strred_features_local_two_exp3/',os.path.splitext(os.path.basename(dis_basename))[0]+'.z')

    strred_two_exp_lnl1_list = []
    strred_two_exp_lnl2_list = []
    strred_two_exp_lnl3_list = []

    strred_two_exp_gnl1_list = []
    strred_two_exp_gnl2_list = []
    strred_two_exp_gnl3_list = []

    if(os.path.exists(os.path.dirname(strred_two_exp_lnl1_outname))==False):
        os.mkdir(os.path.dirname(strred_two_exp_lnl1_outname))
    if(os.path.exists(os.path.dirname(strred_two_exp_lnl2_outname))==False):
        os.mkdir(os.path.dirname(strred_two_exp_lnl2_outname))
    if(os.path.exists(os.path.dirname(strred_two_exp_lnl3_outname))==False):
        os.mkdir(os.path.dirname(strred_two_exp_lnl3_outname))

    if(os.path.exists(os.path.dirname(strred_two_exp_gnl1_outname))==False):
        os.mkdir(os.path.dirname(strred_two_exp_gnl1_outname))
    if(os.path.exists(os.path.dirname(strred_two_exp_gnl2_outname))==False):
        os.mkdir(os.path.dirname(strred_two_exp_gnl2_outname))
    if(os.path.exists(os.path.dirname(strred_two_exp_gnl3_outname))==False):
        os.mkdir(os.path.dirname(strred_two_exp_gnl3_outname))

    print(ref_filename,dis_vid,height,width,dis_fps,strred_two_exp_gnl1_outname)
    frame_num= 0 
    if(os.path.exists(strred_two_exp_gnl1_outname)):
        return
    print(ref_filename,dis_vid,height,width,dis_fps,strred_two_exp_gnl1_outname)

    ref_video = open(ref_filename)
    dis_video = open(dis_vid)
    for framenum in range(framenos_list[i]):
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



#


            strred_two_exp_lnl1a = compute_strred(ref_y_two_exp_lnl1a,ref_y_lnl_two_exp1_nexta,dis_y_two_exp_lnl1a,dis_y_lnl_two_exp1_nexta)
            strred_two_exp_lnl2a = compute_strred(ref_y_two_exp_lnl2a,ref_y_lnl_two_exp2_nexta,dis_y_two_exp_lnl2a,dis_y_lnl_two_exp2_nexta)
            strred_two_exp_lnl3a = compute_strred(ref_y_two_exp_lnl3a,ref_y_lnl_two_exp3_nexta,dis_y_two_exp_lnl3a,dis_y_lnl_two_exp3_nexta)

            strred_two_exp_lnl1b = compute_strred(ref_y_two_exp_lnl1b,ref_y_lnl_two_exp1_nextb,dis_y_two_exp_lnl1b,dis_y_lnl_two_exp1_nextb)
            strred_two_exp_lnl2b = compute_strred(ref_y_two_exp_lnl2b,ref_y_lnl_two_exp2_nextb,dis_y_two_exp_lnl2b,dis_y_lnl_two_exp2_nextb)
            strred_two_exp_lnl3b = compute_strred(ref_y_two_exp_lnl3b,ref_y_lnl_two_exp3_nextb,dis_y_two_exp_lnl3b,dis_y_lnl_two_exp3_nextb)

            strred_two_exp_gnl1a = compute_strred(ref_y_two_exp_gnl1a,ref_y_gnl_two_exp1_nexta,dis_y_two_exp_gnl1a,dis_y_gnl_two_exp1_nexta)
            strred_two_exp_gnl2a = compute_strred(ref_y_two_exp_gnl2a,ref_y_gnl_two_exp2_nexta,dis_y_two_exp_gnl2a,dis_y_gnl_two_exp2_nexta)
            strred_two_exp_gnl3a = compute_strred(ref_y_two_exp_gnl3a,ref_y_gnl_two_exp3_nexta,dis_y_two_exp_gnl3a,dis_y_gnl_two_exp3_nexta)

            strred_two_exp_gnl1b = compute_strred(ref_y_two_exp_gnl1b,ref_y_gnl_two_exp1_nextb,dis_y_two_exp_gnl1b,dis_y_gnl_two_exp1_nextb)
            strred_two_exp_gnl2b = compute_strred(ref_y_two_exp_gnl2b,ref_y_gnl_two_exp2_nextb,dis_y_two_exp_gnl2b,dis_y_gnl_two_exp2_nextb)
            strred_two_exp_gnl3b = compute_strred(ref_y_two_exp_gnl3b,ref_y_gnl_two_exp3_nextb,dis_y_two_exp_gnl3b,dis_y_gnl_two_exp3_nextb)

            strred_two_exp_lnl1_list.append(np.concatenate((strred_two_exp_lnl1a,strred_two_exp_lnl1b)))
            strred_two_exp_lnl2_list.append(np.concatenate((strred_two_exp_lnl2a,strred_two_exp_lnl2b)))
            strred_two_exp_lnl3_list.append(np.concatenate((strred_two_exp_lnl3a,strred_two_exp_lnl3b)))

            strred_two_exp_gnl1_list.append(np.concatenate((strred_two_exp_gnl1a,strred_two_exp_gnl1b)))
            strred_two_exp_gnl2_list.append(np.concatenate((strred_two_exp_gnl2a,strred_two_exp_gnl2b)))
            strred_two_exp_gnl3_list.append(np.concatenate((strred_two_exp_gnl3a,strred_two_exp_gnl3b)))

            ref_y = ref_y_next
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
            if(len(strred_two_exp_lnl1_list)):
                dump(strred_two_exp_lnl1_list,strred_two_exp_lnl1_outname)
                dump(strred_two_exp_lnl2_list,strred_two_exp_lnl2_outname)
                dump(strred_two_exp_lnl3_list,strred_two_exp_lnl3_outname)

                dump(strred_two_exp_gnl1_list,strred_two_exp_gnl1_outname)
                dump(strred_two_exp_gnl2_list,strred_two_exp_gnl2_outname)
                dump(strred_two_exp_gnl3_list,strred_two_exp_gnl3_outname)
                break
            if(len(strred_two_exp_lnl1_list)):
                dump(strred_two_exp_lnl1_list,strred_two_exp_lnl1_outname)
                dump(strred_two_exp_lnl2_list,strred_two_exp_lnl2_outname)
                dump(strred_two_exp_lnl3_list,strred_two_exp_lnl3_outname)

                dump(strred_two_exp_gnl1_list,strred_two_exp_gnl1_outname)
                dump(strred_two_exp_gnl2_list,strred_two_exp_gnl2_outname)
                dump(strred_two_exp_gnl3_list,strred_two_exp_gnl3_outname)





    #strred_command = ['./run_strred.sh',ref_video,dis_vid,strred_outname,dis_fps]
    #try:
    #subprocess.check_call(strred_command)
    #subprocess.check_call(psnr_command)
    #except:
    #    return
    return

Parallel(n_jobs=10)(delayed(single_vid_strred)(i) for i in range(len(dis_metadata_csv)))
#for i in range(len(dis_metadata_csv)):
#    single_vid_strred(i)
