import numpy as np
import os
import glob
import cv2
from joblib import Parallel, delayed, dump
import scipy.ndimage
import pandas as pd
from skvideo.utils.mscn import gen_gauss_window
import skimage.util
import math
from hdr_utils import hdr_yuv_read
import argparse
import numpy as np
from scipy import ndimage
from transform_frame import TransformFrame
from strred.strred_utils import SpatialSteerablePyramid

parser = argparse.ArgumentParser(
    description='Compute strred for a set of videos')

parser.add_argument('--framenos_list', help='Upscaled video names')
parser.add_argument('--output_pth', help='Output path')
parser.add_argument('--nonlinear_method', help='Nonlinear method')
parser.add_argument('--nonlinear_param', help='Nonlinear param', type=float)
parser.add_argument('--patch_size', help='Patch size', default=31, type=int)
parser.add_argument('--type', help='Type of nonlinearity',
                    default='local', type=str)
args = parser.parse_args()

nltrans = TransformFrame(args.nonlinear_method, args.nonlinear_param,
                         nl_type=args.type, patch_size=args.patch_size)


def est_params(frame, blk, sigma_nn):
    h, w = frame.shape
    sizeim = np.floor(np.array(frame.shape)/blk) * blk
    sizeim = sizeim.astype(np.int32)

    frame = frame[:sizeim[0], :sizeim[1]]

    # paired_products
    temp = []
    for u in range(blk):
        for v in range(blk):
            temp.append(
                np.ravel(frame[v:(sizeim[0]-(blk-v)+1), u:(sizeim[1]-(blk-u)+1)]))
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
    V, d = scipy.linalg.eigh(cov_mat.astype(np.float64))
    V = V.astype(np.float32)

    # Estimate local variance
    sizeim_reduced = (sizeim/blk).astype(np.int32)
    ss = np.zeros((sizeim_reduced[0], sizeim_reduced[1]), dtype=np.float32)
    if np.max(V) > 0:
        # avoid the matrix inverse for extra speed/accuracy
        ss = scipy.linalg.solve(cov_mat, temp)
        ss = np.sum(np.multiply(ss, temp) / (blk**2), axis=0)
        ss = ss.reshape(sizeim_reduced)

    V = V[V > 0]

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
    spatial_levels = []
    temporal_levels = []
    for level in range(5):
        y1 = model.extractSingleBand(
            frame1, filtfile="sp5Filters", band=0, level=level)
        y2 = model.extractSingleBand(
            frame2, filtfile="sp5Filters", band=0, level=level)

        ydiff = y1 - y2

        ss, q = est_params(y1, blk, sigma_nsq)
        ssdiff, qdiff = est_params(ydiff, blk, sigma_nsqt)

        spatial = np.multiply(q, np.log2(1 + ss))
        temporal = np.multiply(qdiff, np.multiply(
            np.log2(1 + ss), np.log2(1 + ssdiff)))
        spatial_levels.append(spatial)
        temporal_levels.append(temporal)
    return spatial_levels, temporal_levels


def strred_features(referenceVideoData, distortedVideoData):
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

    T, M, N = referenceVideoData.shape


#    referenceVideoData = referenceVideoData[:, :, :, 0]
#    distortedVideoData = distortedVideoData[:, :, :, 0]

    rreds = []
    rredt = []

    rredssn = []
    rredtsn = []

    for i in range(0, T-1, 2):
        refFrame1 = referenceVideoData[i, :, :].astype(np.float32)
        refFrame2 = referenceVideoData[i+1, :, :].astype(np.float32)

        disFrame1 = distortedVideoData[i, :, :].astype(np.float32)
        disFrame2 = distortedVideoData[i+1, :, :].astype(np.float32)

        spatialRef, temporalRef = extract_info(refFrame1, refFrame2)
        spatialDis, temporalDis = extract_info(disFrame1, disFrame2)

        rreds += [np.mean(np.abs(spatialRef[i] - spatialDis[i]))
                  for i in range(len(spatialRef))]
        rredt += [np.mean(np.abs(temporalRef[i] - temporalDis[i]))
                  for i in range(len(temporalDis))]

    rreds = np.array(rreds)
    rredt = np.array(rredt)

    strred = rreds * rredt

    return strred


def single_vid_strred(i):
    dis_video_name = upscaled_yuv_names[i]
    ref_video_name = os.path.join(
        '/mnt/31393986-51f4-4175-8683-85582af93b23/videos/HDR_2022_SPRING_yuv_update/', ref_names[i])
    ssim_outname = os.path.join(output_pth, os.path.splitext(
        os.path.basename(dis_video_name))[0]+'.csv')
    if dis_video_name == ref_names[i]:
        return
    if os.path.exists(ssim_outname):
        print('Found ', ssim_outname)
        return
    fps = 1
    dis_video = open(os.path.join(
        '/mnt/31393986-51f4-4175-8683-85582af93b23/videos/HDR_2022_SPRING_yuv_update/', upscaled_yuv_names[i]))

    ref_video = open(ref_video_name)

    width, height = int(3840), int(2160)

    print(ref_video_name, dis_video_name, height, width, fps)

    print(ref_video_name, dis_video_name, height, width, fps)
    feats_list = []

    ref_y_prev, _, _ = hdr_yuv_read(ref_video, 0, height, width)
    dis_y_prev, _, _ = hdr_yuv_read(dis_video, 0, height, width)

    if args.nonlinear_method.lower() != 'none':
        ref_y_prev = nltrans.transform_frame(ref_y_prev)
        dis_y_prev = nltrans.transform_frame(dis_y_prev)
    for framenum in range(1, framenos_list[i], 20):
        print(framenum)
        ref_y, _, _ = hdr_yuv_read(ref_video, framenum, height, width)
        dis_y, _, _ = hdr_yuv_read(dis_video, framenum, height, width)

        if args.nonlinear_method.lower() != 'none':
            ref_y = nltrans.transform_frame(ref_y)
            dis_y = nltrans.transform_frame(dis_y)
        feats = strred_features([ref_y_prev, ref_y], [dis_y_prev, dis_y])
        feats_list.append(feats)
    vid_feats = np.array(feats_list)
    df_one = pd.DataFrame(vid_feats.mean(axis=0)).transpose()

    df_one['video'] = dis_video_name
    return df_one


# csv_file = '/home/zs5397/code/hdr_fr_code/spring2022_yuv_info.csv'
csv_file = args.framenos_list
csv_df = pd.read_csv(csv_file)
files = csv_df["yuv"]
fps_list = 25
framenos_list = csv_df["framenos"]
upscaled_yuv_names = csv_df['yuv']
ref_names = csv_df['ref']
output_pth = os.path.join(args.output_pth, 'strred')

if not os.path.exists(output_pth):
    os.makedirs(output_pth)
r = Parallel(n_jobs=80)(delayed(single_vid_strred)(i)
                       for i in range(0, len(upscaled_yuv_names)))
features = pd.concat(r)
features.to_csv(
    f'{output_pth}/strred_{args.nonlinear_method}_{args.type}_{args.nonlinear_param}_{args.patch_size}_.csv')
