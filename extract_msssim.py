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
from skimage.metrics._structural_similarity import structural_similarity_features as ssim_features
import pdb

parser = argparse.ArgumentParser(
    description='Compute MS-SSIM for a set of videos')

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


def msssim(frame1, frame2, method='product'):
    feats = []
    level = 3

    downsample_filter = np.ones(2, dtype=np.float32)/2.0

    im1 = frame1.astype(np.float32)
    im2 = frame2.astype(np.float32)

    for i in range(level):
        pdb.set_trace()
        feat_one_scale = ssim_features(im1, im2)
        feats += feat_one_scale
        filtered_im1 = scipy.ndimage.correlate1d(im1, downsample_filter, 0)
        filtered_im1 = scipy.ndimage.correlate1d(
            filtered_im1, downsample_filter, 1)
        filtered_im1 = filtered_im1[1:, 1:]

        filtered_im2 = scipy.ndimage.correlate1d(im2, downsample_filter, 0)
        filtered_im2 = scipy.ndimage.correlate1d(
            filtered_im2, downsample_filter, 1)
        filtered_im2 = filtered_im2[1:, 1:]

        im1 = filtered_im1[::2, ::2]
        im2 = filtered_im2[::2, ::2]
        pdb.set_trace()
    return feats


def single_vid_msssim(i):
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
    ssim_list = []

    for framenum in range(0, framenos_list[i], 25):
        print(framenum)
        ref_y, _, _ = hdr_yuv_read(ref_video, framenum, height, width)
        dis_y, _, _ = hdr_yuv_read(dis_video, framenum, height, width)

        if args.nonlinear_method.lower() != 'none':
            ref_y = nltrans.transform_frame(ref_y)
            dis_y = nltrans.transform_frame(dis_y)
        feats = msssim(ref_y, dis_y)
        ssim_list.append(feats)
        pdb.set_trace()
    vid_feats = np.array(ssim_list)
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
output_pth = os.path.join(args.output_pth, 'msssim')

if not os.path.exists(output_pth):
    os.makedirs(output_pth)
r = Parallel(n_jobs=1)(delayed(single_vid_msssim)(i)
                       for i in range(0, len(upscaled_yuv_names)))
features = pd.concat(r)
features.to_csv(
    f'{output_pth}/msssim_{args.nonlinear_method}_{args.type}_{args.nonlinear_param}_{args.patch_size}_.csv')
