import argparse
import subprocess
import joblib
import os
from GREED_feat import greed_feat
import numpy as np
import time
import os
import glob
from joblib import Parallel, delayed, dump
import pandas as pd
from os.path import join


def greed_single_vid(i):

    ref_video_name = os.path.join(
        '/mnt/31393986-51f4-4175-8683-85582af93b23/videos/HDR_2022_SPRING_yuv_update/', ref_names[i])
    dis_video = os.path.join(
        '/mnt/31393986-51f4-4175-8683-85582af93b23/videos/HDR_2022_SPRING_yuv_update/', upscaled_yuv_names[i])
    tmpcsv = f'./tmp/{os.path.basename(dis_video)}.csv'
    if os.path.exists(tmpcsv):
        return
    height = 2160
    width = 3840

    ref_fps = 25
    bit_depth = 10
    if bit_depth == 8:
        pix_format = 'yuv420p'
    else:
        pix_format = 'yuv420p10le'

    dist_fps = ref_fps  # frame rate of distorted sequence
    try:
        GREED_feat = greed_feat(dis_video, ref_video_name, dist_fps, ref_fps,
                                'bior22', height, width, bit_depth, 'lnl', 0, 'local')
        df_one = pd.DataFrame(GREED_feat).transpose()
        df_one['video'] = os.path.basename(dis_video)
        df_one.to_csv(f'./tmp_dog_60-90/{os.path.basename(dis_video)}.csv')
        return df_one
    except Exception as e:
        print('error',e)
        return None


csv_file = '/home/zs5397/code/hdr_fr_code/spring2022_yuv_info.csv'
csv_df = pd.read_csv(csv_file)
files = csv_df["yuv"]
framenos_list = csv_df["framenos"]
upscaled_yuv_names = csv_df['yuv']
ref_names = csv_df['ref']

output_pth = '/media/zaixi/zaixi_nas/HDRproject/feats/fr_evaluate_HDRAQ_correct/greed'
if not os.path.exists(output_pth):
    os.makedirs(output_pth)
r = Parallel(n_jobs=1)(delayed(greed_single_vid)(i)
                       for i in range(len(upscaled_yuv_names)))
df = pd.concat(r)
df.to_csv(join(output_pth, 'greed_none_local_-0.5_31_.csv'))
