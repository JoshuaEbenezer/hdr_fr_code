import pandas as pd
import subprocess
import numpy as np
import os
import glob
from joblib import Parallel,delayed

dis_metadata_csv = pd.read_csv("/home/josh-admin/hdr/qa/hdr_chipqa/fall2021_yuv_rw_info.csv")
print([i for i in dis_metadata_csv["yuv"]])

outfolder = './hdr_greed_features'

def greed_single_vid(i):
    dis_basename = dis_metadata_csv['yuv'].iloc[i][:-4]+'_upscaled.yuv'
    dis_filename = os.path.join("/media/josh-admin/seagate/fall2021_hdr_upscaled_yuv",dis_basename)
    print(dis_filename)

    dis_fps = str(dis_metadata_csv['fps'].iloc[i])

    content = dis_basename.split('_')[2]
    orig_basename = '4k_ref_'+content+'_upscaled.yuv'
    orig_fps = dis_fps
     # for LIVE ETRI, the distorted version's FPS was made to match the original by interpolation
#    dis_fps = orig_fps
    orig_filename = os.path.join("/media/josh-admin/seagate/fall2021_hdr_upscaled_yuv/",orig_basename)
    print(orig_filename,orig_fps)
    if not (os.path.exists(orig_filename)):
        print(orig_filename, ' does not exist')
    if not (os.path.exists(dis_filename)):
        print(dis_filename,' does not exist')
    subprocess.check_call(['./run_greed.sh',orig_filename,dis_filename,orig_fps,dis_fps,str(2160),str(3840),str(10),outfolder])
Parallel(n_jobs=-1)(delayed(greed_single_vid)(i) for i in range(len(dis_metadata_csv)))
#for i in range(len(dis_metadata_csv)):
#    greed_single_vid(i)
