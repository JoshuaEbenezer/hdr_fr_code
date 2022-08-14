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

feature_files= glob.glob('/home/zs5397/code/hdr_fr_code/hdrvdp/features/hdrvdp2_features/*')

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
                                          all_preds, all_dmos, p0=[0.1,0.2,0.3,60], maxfev=20000)

    preds_fitted = logistic(all_preds,b1, b2, b3, b4)
    preds_srocc = spearmanr(preds_fitted,all_dmos)
    preds_lcc = pearsonr(all_preds,all_dmos)
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
score_df = pd.read_csv('/home/zs5397/code/hdr_fr_code/Spring_2022_score.csv')
pred = pd.read_csv('/home/zs5397/code/hdr_fr_code/hdrvdp/vdp3.csv')

upscaled_names =list(score_df["video"])
pred = pred.merge(score_df[['video','sureal_DMOS']],on='video')

results(pred['scores'],pred['sureal_DMOS'])

#sys.stdout.close()
