import numpy as np
from scipy.io import loadmat
import sys
import pandas as pd
import os
from joblib import load,dump
from scipy.stats import spearmanr,pearsonr
from scipy.optimize import curve_fit
import glob

#sys.stdout = open("vdp_srcc_results.txt",'a')

feature_files= glob.glob('./hdrvdp2_sep_features/*')
def results(all_preds,all_dmos):
    all_preds = np.asarray(all_preds)
    print(np.max(all_preds),np.min(all_preds))
    all_preds[np.isnan(all_preds)]=0
    all_dmos = np.asarray(all_dmos)
    [[b0, b1, b2, b3, b4], _] = curve_fit(lambda t, b0, b1, b2, b3, b4: b0 * (0.5 - 1.0/(1 + np.exp(b1*(t - b2))) + b3 * t + b4),
                                          all_preds, all_dmos, p0=0.5*np.ones((5,)), maxfev=20000)

    preds_fitted = b0 * (0.5 - 1.0/(1 + np.exp(b1*(all_preds - b2))) + b3 * all_preds+ b4)
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
all_vdp = []
all_dmos = []
score_df = pd.read_csv('/home/josh/hdr/fall21_score_analysis/sureal_dark_mos_and_dmos.csv')


upscaled_names =list(score_df["video"])
for f in feature_files:
    print(f)
    if('ref' in f):
        continue
    vid_name= os.path.splitext(os.path.basename(f))[0]
    vid_index = upscaled_names.index(vid_name)
    dmos = score_df["dark_dmos"].iloc[vid_index]

    vdp = loadmat(f)['Q'][0][0]
    print(vdp)
    all_vdp.append(vdp)
    all_dmos.append(dmos)


results(all_vdp,all_dmos)

#sys.stdout.close()
