import numpy as np


import os
import glob
import cv2
from joblib import Parallel, delayed, dump
import scipy.ndimage
import pandas as pd
import skimage.util
import math
from hdr_utils import hdr_yuv_read
from transform_frame import TransformFrame
import argparse
import numpy as np
from scipy import ndimage
from skvideo.measure import psnr
parser = argparse.ArgumentParser(
    description='Compute psnr for a set of videos')

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

# def psnr_feature(im_x,im_y):
#     win_rows, win_cols = 3, 3
#     win_mean_x = ndimage.uniform_filter(im_x, (win_rows, win_cols),mode = 'mirror')
#     win_sqr_mean_x  = ndimage.uniform_filter(im_x**2, (win_rows, win_cols),mode = 'mirror')
#     win_var_x = win_sqr_mean_x - win_mean_x**2


def single_vid_psnr(i):
    dis_video_name = upscaled_yuv_names[i]
    ref_video_name = os.path.join(
        '/mnt/31393986-51f4-4175-8683-85582af93b23/videos/HDR_2022_SPRING_yuv_update', ref_names[i])
    psnr_outname = os.path.join(output_pth, os.path.splitext(
        os.path.basename(dis_video_name))[0]+'.csv')
    if dis_video_name == ref_names[i]:
        return
    if os.path.exists(psnr_outname):
        print('Found ', psnr_outname)
        return
    fps = 1
    dis_video = open(os.path.join(
        '/mnt/31393986-51f4-4175-8683-85582af93b23/videos/HDR_2022_SPRING_yuv_update', upscaled_yuv_names[i]))

    ref_video = open(ref_video_name)

    width, height = int(3840), int(2160)

    print(ref_video_name, dis_video_name, height, width, fps)

    print(ref_video_name, dis_video_name, height, width, fps)
    psnr_list = []

    for framenum in range(0, int(framenos_list[i]), 25):
        print(framenum)
        try:
            ref_y, _, _ = hdr_yuv_read(ref_video, framenum, height, width)
            dis_y, _, _ = hdr_yuv_read(dis_video, framenum, height, width)
        except:
            print('Error reading frame ', framenum, dis_video)
            break

        # ref_y = ref_y.astype(np.float32)
        # dis_y = dis_y.astype(np.float32)
        if args.nonlinear_method.lower() != 'none':
            ref_y = nltrans.transform_frame(ref_y)
            dis_y = nltrans.transform_frame(dis_y)
        psnr_score = psnr(ref_y, dis_y)
        psnr_list.append([psnr_score])
    vid_feats = np.array(psnr_list)
    df_one = pd.DataFrame(vid_feats.mean(axis=0)).transpose()

    df_one['video'] = dis_video_name
    return df_one


# csv_file = '/home/zs5397/code/hdr_fr_code/spring2022_yuv_info.csv'
csv_file = args.framenos_list
csv_df = pd.read_csv(csv_file)
csv_df.fillna(250, inplace=True)
files = csv_df["yuv"]
fps_list = 25
framenos_list = csv_df["framenos"]
upscaled_yuv_names = csv_df['yuv']
ref_names = csv_df['ref']
output_pth = os.path.join(args.output_pth, 'psnr')

if not os.path.exists(output_pth):
    os.makedirs(output_pth)
# r = Parallel(n_jobs=1)(delayed(single_vid_psnr)(i)
#                        for i in range(0, len(upscaled_yuv_names)))
r = Parallel(n_jobs=50)(delayed(single_vid_psnr)(i)
                        for i in range(0, len(upscaled_yuv_names)))
features = pd.concat(r)
features.to_csv(
    f'{output_pth}/psnr_{args.nonlinear_method}_{args.type}_{args.nonlinear_param}_{args.patch_size}_.csv')
