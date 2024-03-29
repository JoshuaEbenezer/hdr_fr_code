import numpy as np
import sys
import pandas as pd
import os
from joblib import load,dump
from scipy.stats import spearmanr,pearsonr
from scipy.optimize import curve_fit
import glob

#sys.stdout = open("speed_srcc_results.txt",'a')

folders = glob.glob('./features/speed/')

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
print(folders)
for folder in folders:
    base = os.path.basename(folder)
    print(base)
    filenames = glob.glob(os.path.join(folder,'*.z'))
    all_speed = []
    all_vspeed = []
    all_dmos = []
    score_df = pd.read_csv('/home/zs5397/code/hdr_fr_code/Spring_2022_score.csv')
    out_folder = os.path.join('./feature_means/',base) 
    if(os.path.exists(out_folder)==False):
        os.makedirs(out_folder)


    upscaled_names =list(score_df["video"])
    for f in filenames:
        if('mu1000000' in f):
            continue
        vid_name= os.path.splitext(os.path.basename(f))[0]+'.yuv'
        vid_index = upscaled_names.index(vid_name)
        dmos = score_df["sureal_DMOS"].iloc[vid_index]
        outname= os.path.join(out_folder,vid_name+'.z')
        if(os.path.exists(outname)):
            X = load(outname)
            speed = X['speed']
            v_speed = X['v_speed']
        else:

            speed_list = load(f)
            speed = np.mean([s[0] for s in speed_list])
            v_speed =np.mean([s[0]*s[2] for s in speed_list]) 
            X = {'speed':speed,'v_speed':v_speed}
            dump(X,outname)
        all_speed.append(speed)
        all_vspeed.append(v_speed)

        all_dmos.append(dmos)


    results(all_speed,all_dmos)
    results(all_vspeed,all_dmos)

#sys.stdout.close()
