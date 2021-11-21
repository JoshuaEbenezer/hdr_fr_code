import numpy as np
import pandas as pd
import os
from joblib import load,dump
from scipy.stats import spearmanr,pearsonr
from scipy.optimize import curve_fit
import glob

filenames = glob.glob('./speed_features_local_m_exp1/*.z')
all_speed = []
all_vspeed = []
all_dmos = []
score_df = pd.read_csv('/home/josh/hdr/fall21_score_analysis/fall21_mos_and_dmos_rawavg.csv')
out_folder = './feature_means/speed_features_local_m_exp1_mean'
if(os.path.exists(out_folder)==False):
    os.mkdir(out_folder)

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
    preds_rmse = np.sqrt(np.mean(preds_fitted-all_dmos)**2)
    print('SROCC:')
    print(preds_srocc[0])
    print('LCC:')
    print(preds_lcc[0])
    print('RMSE:')
    print(preds_rmse)
    print(len(all_preds),' videos were read')

upscaled_names =[v+'_upscaled' for v in score_df["video"]]
for f in filenames:
    if('ref' in f):
        continue
    vid_name= os.path.splitext(os.path.basename(f))[0]
    vid_index = upscaled_names.index(vid_name)
    dmos = score_df["dark_dmos"].iloc[vid_index]
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

