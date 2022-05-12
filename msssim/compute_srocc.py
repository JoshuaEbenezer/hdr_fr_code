import numpy as np
from joblib import load
import pandas as pd
import os
import glob
from scipy.stats import spearmanr,pearsonr
from scipy.optimize import curve_fit
import glob

def logistic(t, b1, b2, b3, b4):
    a = b1 # ymax
    b = b2 # ymin
    c = b3 # xmean
    s = b4
    yhat = (a-b)/(1+ np.exp(-((t-c)/abs(s)))) + b
    return yhat


def results(all_preds,all_dmos):
    all_preds = np.asarray(all_preds)
    print(np.max(all_preds),np.min(all_preds))
    all_preds[np.isnan(all_preds)]=0
    all_dmos = np.asarray(all_dmos)
    [[b1, b2, b3, b4], _] = curve_fit(logistic,
                                          all_preds, all_dmos, p0=0.5*np.ones((4,)), maxfev=20000)

    preds_fitted = logistic(all_preds,b1, b2, b3, b4)
    preds_srocc = spearmanr(preds_fitted,all_dmos)
    preds_lcc = pearsonr(preds_fitted,all_dmos)
    preds_rmse = np.sqrt(np.mean((preds_fitted-all_dmos)**2))
    print('SROCC:')
    print(preds_srocc[0])
    print('LCC:')
    print(preds_lcc[0])
    print('RMSE:')
    print(preds_rmse)
    print(len(all_preds),' videos were read')
    return

feature_folders = ['./features/msssim_features/']#   glob.glob(os.path.join('./features/*'))

for folder in feature_folders:
    print(os.path.basename(folder))
    filenames = glob.glob(os.path.join(folder,'*.z'))
    score_df = pd.read_csv('/home/zs5397/code/hdr_fr_code/Spring_2022_score.csv')


    all_psnr = []
    all_dmos = []
    upscaled_names =list(score_df["video"])
    for infile in filenames:
        vid_name= os.path.splitext(os.path.basename(infile))[0]+'.yuv'
        vid_index = upscaled_names.index(vid_name)

        dmos = score_df["sureal_DMOS"].iloc[vid_index]
        psnr = np.mean(load(infile))
        all_psnr.append(psnr)
        all_dmos.append(dmos)

    results(all_psnr,all_dmos)



