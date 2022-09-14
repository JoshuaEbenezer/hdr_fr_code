from email import contentmanager
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
from sklearn.metrics import mean_squared_error
import itertools
from joblib import Parallel, delayed
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import joblib
import matplotlib.pyplot as plt
import sys
import pickle


def split_content(df, test_size=0.2):
    contents = df['content']
    contents = contents.unique()
    c_train, c_test = train_test_split(contents, test_size=test_size)
    train = df[df['content'].map(lambda x: x in c_train)]
    test = df[df['content'].map(lambda x: x in c_test)]
    return train, test


def generatekfs(dataframe):
    dataframe = dataframe.sort_values(by='content')
    kfolds = sklearn.model_selection.KFold(5)
    conts = dataframe['content'].unique()
    sp = kfolds.split(conts)

    crossvalidationlist = []
    for i in range(kfolds.get_n_splits(conts)):
        live_train, live_test = next(sp)

        live_train_ind = list(
            dataframe.index[dataframe['content'].map(lambda x: x in conts[live_train])])
        live_test_ind = list(
            dataframe.index[dataframe['content'].map(lambda x: x in conts[live_test])])

        crossvalidationlist.append([live_train_ind, live_test_ind])
    return dataframe, crossvalidationlist


def predict(df, svr_pth, scaler_pth):
    X_test = df.drop(['video'], axis=1)
    scaler = StandardScaler()
    svrfile = open(svr_pth, 'rb')
    grid_svr = pickle.load(svrfile)
    svrfile.close()
    scalerfile = open(scaler_pth, 'rb')
    scaler = pickle.load(scalerfile)
    scalerfile.close()

    X_test = scaler.transform(X_test)
    predict = grid_svr.predict(X_test)
    df['pred'] = predict
    return df


def train_for_srocc_svr(df, split='content'):
    if split == 'content':

        train, test = split_content(df)
        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)
        df, kfs = generatekfs(train)

    elif split == 'banding_vs_all':
        # dfbanding =
        df_banding = df[df['is_AQ_test']]
        df_banding = df_banding[df_banding['is_AQ_test']]
        train, test = split_content(df_banding)
        train = pd.concat([train, df[~df['is_AQ_test']]])
        train = train.drop('is_AQ_test', axis=1)
        test = test.drop('is_AQ_test', axis=1)
        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)
        df, kfs = generatekfs(train)

    X_train = df.drop(['video', 'score', 'content'], axis=1)
    y_train = df['score']
    X_test = test.drop(['video', 'score', 'content'], axis=1)
    y_test = test['score']
    scaler = StandardScaler()

    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
#     param_grid = [
#   {'C': np.logspace(-7,2,10), 'kernel': ['linear'],'gamma':np.logspace(-8,2,10,base = 2)},
#   {'C': np.logspace(-7,2,10), 'gamma': np.logspace(3,12,10,base=2), 'kernel': ['rbf']},
#  ]
    param_grid = [
        {'C': np.logspace(-7, 4, 5, base=2),
         'kernel': ['linear', 'rbf'], 'gamma':np.logspace(-1, 5, 5, base=2)},
    ]
    grid_svr = GridSearchCV(SVR(), param_grid=param_grid, cv=kfs, n_jobs=1)

    grid_svr.fit(X_train, y_train)

    preds = grid_svr.predict(X_test)
    srocc_test = spearmanr(preds, y_test)
    plcc_test = pearsonr(preds, y_test)
    rmse_test = mean_squared_error(preds, y_test, squared=False)
    existing_models = len(os.listdir('./models/svr'))
    svrfile = open(f"./models/svr/model_{existing_models}.pkl", "wb")
    pickle.dump(grid_svr, svrfile)
    svrfile.close()
    scalerfile = open(f"./models/scaler/model_{existing_models}.pkl", "wb")
    pickle.dump(scaler, scalerfile)
    scalerfile.close()
    # print(srocc_test[0])
    # sys.stdout.flush()
    test['pred'] = preds
    return srocc_test[0], plcc_test[0], rmse_test, test


def unpack_and_plot(r, plotname, feats, get_pred=False):
    sroccs = [x[0] for x in r]
    plccs = [x[1] for x in r]
    rmses = [x[2] for x in r]
    res = [x[3] for x in r[1:]]
    pred = r[0][3]
    pred = pred[['video', 'pred']]
    for each_time in res:
        pred = pred.merge(
            each_time[['video', 'pred']], how='outer', on='video')

    ave = pred.mean(axis=1, numeric_only=True)
    pred.loc[:, 'pred'] = ave
    pred = pred.merge(feats[['video', 'score']], on='video')
    plt.scatter(pred['score'], pred['pred'])
    plt.savefig(plotname)
    plt.cla()

    if get_pred:
        return np.median(sroccs), np.median(plccs), np.median(rmses), pred[['video', 'pred', 'score']]
    else:
        return np.median(sroccs), np.median(plccs), np.median(res)


def combine_feats(files):
    allfeat = []
    for i in range(len(files)):
        feats_one = pd.read_csv(files[i])
        # feats_one.drop('Unnamed: 0',axis = 1)
        allfeat.append(feats_one)
    feats = pd.concat(allfeat)
    return feats.drop('Unnamed: 0', axis=1)


def combine_feats_josh(files):

    allfeat = []
    names = []
    for i in range(len(files)):

        feats_one = joblib.load(files[i])['features']
        allfeat.append(feats_one)
        vname = os.path.basename(files[i])[:-11]+'.mp4'
        # print(vname)
        names.append(vname)
    feats = pd.DataFrame(np.stack(allfeat))
    feats['video'] = names
    return feats


if __name__ == '__main__':

    feats = pd.read_csv(
        '/home/zaixi/code/HDRproject/training/ictcp_feats.csv', index_col=0)
    scores = pd.read_csv(
        '/home/zaixi/code/HDRproject/training/HDR_score_4k.csv')
    feats = feats.merge(
        scores[['video', 'content', 'score']], on='video', how='inner')

    N_times = 10

    rocc = []
    r = Parallel(n_jobs=-1)(delayed(train_for_srocc_svr)(feats)
                            for t in range(N_times))
