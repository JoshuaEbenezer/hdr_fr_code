from training import *
import glob
import pandas as pd
import numpy as np
import sklearn
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import itertools
# from joblib import Parallel, delayed
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from os.path import join
from mpi4py import MPI
import sys
import re

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


# The input path for the feature files
feats_pth_root = '/media/zaixi/zaixi_nas/HDRproject/feats/fr_evaluate_HDRAQ_correct'
# The path to the input score file
score_file = '/home/zs5397/code/training/Spring_2022_score.csv'
# The output path for prediction csv file. i.e. the predicted scores for each video. It's not strictly defined but rather just the average prediction of each train-test split.
pred_csvpth = './predicts/greed/'
# Output path for scatterplot.
plot_pth = './plots/fr'
os.makedirs(pred_csvpth, exist_ok=True)
os.makedirs(plot_pth, exist_ok=True)
N_times = 100
df_score = pd.read_csv(score_file, index_col=0)
counter = 0
parameters_localexp = [0.5, 5.0]
configs = []

for method in ['msssim']:
    configs.append([False, method, 0, 0])
    for par1 in parameters_localexp:
        for par2 in parameters_localexp:
            configs.append([True, method, par1, par2])


print('number of options ', len(configs))
localexpp1s = []
localexpp2s = []
methods = []
sroccs = []
plccs = []
rmses = []
allres = []
for cfig_index in range(rank, len(configs), size):
    print(configs[cfig_index])
    nonlinear, method, par1, par2 = configs[cfig_index]

    feature = pd.read_csv(
        join(feats_pth_root, method, f'{method}_none_local_-0.5_31_.csv'), index_col=0)
    # get the local transformed features
    if nonlinear:

        nonlinear_feat1 = pd.read_csv(
            join(feats_pth_root, method, f'{method}_texp_local_{par1}_31_.csv'), index_col=0)
        nonlinear_feat2 = pd.read_csv(
            join(feats_pth_root, method, f'{method}_texp_local_{par2}_31_.csv'), index_col=0)
        local_exp_feats = nonlinear_feat1.merge(
            nonlinear_feat2, on='video')
        feature = feature.merge(local_exp_feats, on='video')

    # drop videos that have Epl in it.
    feature = feature[~feature['video'].str.contains('Epl')]

    print(feature.shape)
    feature['content'] = feature['video'].map(lambda x: x.split('_')[0])
    # feature['video'] = feature['video'].map(lambda x: x[:-4])
    feature = feature.merge(df_score[['video', 'sureal_DMOS']])
    feature = feature.rename({'sureal_DMOS': 'score'}, axis=1)
    r = []
    for times in range(N_times):
        r_eachtime = train_for_srocc_svr(feature)
        r.append(r_eachtime)
    plotname = os.path.join(
        plot_pth, f'par_{par1}_{par2}_{method}_{nonlinear}_scatter.jpg')

    if not os.path.exists(os.path.dirname(plotname)):
        os.makedirs(os.path.dirname(plotname))
    srocc, plcc, rmse, pred = unpack_and_plot(
        r, plotname, feature, get_pred=True)
    pred_csvfile = join(
        pred_csvpth, f'par_{par1}_{par2}_{method}_{nonlinear}.csv')
    print(srocc)
    pred.to_csv(pred_csvfile)
    plccs.append(plcc)
    rmses.append(rmse)
    sroccs.append(srocc)
    methods.append(method)
    localexpp1s.append(par1)
    localexpp2s.append(par2)
    res = pd.DataFrame({'localexpp1s': localexpp1s, 'localexpp2s': localexpp2s,
                       'method': methods, 'srocc': sroccs, 'plcc': plccs, 'rmse': rmses})
    allres.append(res)

print("process {} send data to root...".format(rank))
recv_data = comm.gather(res, root=0)
if rank == 0:
    print("process {} gather all data ...".format(rank))
    df = pd.concat(recv_data)
    df.to_csv('eval_frssim_msssim3.csv')
