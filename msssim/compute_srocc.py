import numpy as np
from joblib import load
import pandas as pd
import os
import glob
from scipy.stats import spearmanr,pearsonr
from scipy.optimize import curve_fit
import glob

def results(all_preds,all_dmos):
    all_preds = np.asarray(all_preds)
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

feature_folders = ['./features/msssim_features/']#   glob.glob(os.path.join('./features/*'))

for folder in feature_folders:
    print(os.path.basename(folder))
    filenames = glob.glob(os.path.join(folder,'*.z'))
    score_df = pd.read_csv('/Users/joshua/code/hdr/fall21_score_analysis/sureal_dark_mos_and_dmos.csv')


    all_psnr = []
    all_dmos = []
    upscaled_names = [v+'_upscaled' for v in score_df['video']]
    for infile in filenames:
        vid_name= os.path.splitext(os.path.basename(infile))[0]

        dmos = score_df['dark_dmos'].iloc[upscaled_names.index(vid_name)]
        psnr = np.mean(load(infile))
        all_psnr.append(psnr)
        all_dmos.append(dmos)

    results(all_psnr,all_dmos)



