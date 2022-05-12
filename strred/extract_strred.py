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
      # avoid the matrix inverse for extra speed/accuracy
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
def compute_strred(referenceVideoData, distortedVideoData):
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


    referenceVideoData = np.asarray(referenceVideoData)
    distortedVideoData = np.asarray(distortedVideoData)
    assert(referenceVideoData.shape == distortedVideoData.shape)

    T, M, N= referenceVideoData.shape


#    referenceVideoData = referenceVideoData[:, :, :, 0]
#    distortedVideoData = distortedVideoData[:, :, :, 0]

    rreds = []
    rredt = []

    rredssn = []
    rredtsn = []

    for i in range(0, T-1, 2):
      refFrame1 = referenceVideoData[i,:,:].astype(np.float32)
      refFrame2 = referenceVideoData[i+1,:,:].astype(np.float32)

      disFrame1 = distortedVideoData[i,:,:].astype(np.float32)
      disFrame2 = distortedVideoData[i+1,:,:].astype(np.float32)

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

def single_vid_strred(i):
    dis_basename = dis_metadata_csv['yuv'].iloc[i]
    ref_basename = dis_metadata_csv['ref'].iloc[i]
    if dis_basename == ref_basename:
        return
    dis_vid = os.path.join("/mnt/31393986-51f4-4175-8683-85582af93b23/videos/HDR_2022_SPRING_yuv",dis_basename)
    print(dis_vid)
    if(os.path.exists(dis_vid)==False):
        print('does not exist')
        raise
    strred_outname = os.path.join('./hdr_rred_features/strred_features/',os.path.splitext(os.path.basename(dis_vid))[0]+'.z')
    if os.path.exists(strred_outname):
        return 
    dis_fps = 25
    ref_filename = os.path.join("/mnt/31393986-51f4-4175-8683-85582af93b23/videos/HDR_2022_SPRING_yuv",ref_basename)

    width,height=int(3840),int(2160)
    if not os.path.exists('./hdr_rred_features/strred_features/'):
        os.makedirs('./hdr_rred_features/strred_features/')
    if(os.path.exists(strred_outname)):
        return
    print(ref_filename,dis_vid,height,width,dis_fps,strred_outname)
    strred_list= []

    frame_num= 0 
    strred_outname = os.path.join('./hdr_rred_features/strred_features/',os.path.splitext(os.path.basename(dis_vid))[0]+'.z')

    strred_list = []
    
    ref_video = open(ref_filename)
    dis_video = open(dis_vid)
    for framenum in range(framenos_list[i]):
        try:
            ref_y,_,_ =hdr_yuv_read(ref_video,framenum,height,width)

           
            ref_y_next,_,_ =hdr_yuv_read(ref_video,framenum+1,height,width)



            dis_y,_,_ =hdr_yuv_read(dis_video,framenum,height,width) 

            dis_y_next,_,_ =hdr_yuv_read(dis_video,framenum+1,height,width) 



        except Exception as e:
            print(e)
            print(framenum, ' frames read')

           
            break
        strred_val = compute_strred([ref_y,ref_y_next],[dis_y,dis_y_next])
  
        strred_list.append(strred_val)
    dump(strred_list,strred_outname)

    return



dis_metadata_csv = pd.read_csv("/home/zs5397/code/hdr_fr_code/spring2022_yuv_info.csv")
print([i for i in dis_metadata_csv["yuv"]])
framenos_list = dis_metadata_csv["framenos"]

outfolder = './hdr_rred_features'

Parallel(n_jobs=1)(delayed(single_vid_strred)(i) for i in range(len(dis_metadata_csv)))
#for i in range(len(dis_metadata_csv)):
#    single_vid_strred(i)
