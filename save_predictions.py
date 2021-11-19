import numpy as np
import pandas as pd
import os
from joblib import load,Parallel,delayed
from scipy.stats import spearmanr,pearsonr
from scipy.optimize import curve_fit
import glob
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import json

def results(all_preds,all_dmos):
    all_preds = np.asarray(all_preds)
    print(np.max(all_preds),np.min(all_preds))
    all_preds[np.isnan(all_preds)]=0
    all_dmos = np.asarray(all_dmos)
    try:
        [[b0, b1, b2, b3, b4], _] = curve_fit(lambda t, b0, b1, b2, b3, b4: b0 * (0.5 - 1.0/(1 + np.exp(b1*(t - b2))) + b3 * t + b4),
                                              all_preds, all_dmos, p0=0.5*np.ones((5,)), maxfev=20000)

        preds_fitted = b0 * (0.5 - 1.0/(1 + np.exp(b1*(all_preds - b2))) + b3 * all_preds+ b4)
    except Exception as e:
        print(e)
        preds_fitted =all_preds
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
    return preds_fitted

def find(lst, a):
    return [i for i, x in enumerate(lst) if x==a]
def unique_scores(test_zips):
    scores = []
    names =[]
    preds =[]
    for v in test_zips:
        print(v)
        for l in v:
            names.append(l[0])
            scores.append(l[1])
            preds.append(l[2])
    print(names)
    print(scores)

    nset = set(names)
    print(len(names))
    print(len(nset))
    print(nset)
    nscores = []
    npreds = []
    nlist = []
    for n in nset:
        indices = find(names,n)
        nlist.append(n)
        nscores.append(np.mean([scores[i] for i in indices]))
        npreds.append(np.mean([preds[i] for i in indices]))
    print(nscores,npreds)
    return nlist,nscores,npreds


scores_df = pd.read_csv('/data/PV_VQA_Study/code/score_cleanup_code/lbvfr_dmos_from_raw_avg_mos.csv')
video_names = scores_df['video']
scores = scores_df['dmos']
#scores_df['content']=[ i[-9:] for i in scores_df['video'] ]
print(len(scores_df['content'].unique()))
srocc_list = []
test_zips = []
VMAF =True  

def trainval_split(trainval_content,r):
    train,val= train_test_split(trainval_content,test_size=0.2,random_state=r)
    train_features = []
    train_indices = []
    val_features = []
    train_scores = []
    val_scores = []
#    feature_folder= "/home/ubuntu/bitstream_mode3_p1204_3/features/p1204_etri_features"

    train_names = []
    val_names = []
    for i,vid in enumerate(video_names):
        if('SRC' in vid):
            continue
        if(VMAF):
            feature_folder= './vmaf/vmaf_features_PR/'
            featfile_name = vid+'.json'

            json_f = os.path.join(feature_folder,featfile_name)
            score = scores[i]

            feature_list = []
            with open(json_f) as f:
                json_data = json.load(f)
    #            print(json_data)
                pool_metrics = json_data['pooled_metrics']
                for key in pool_metrics.keys():
                    if(key=='vmaf'):
                        continue
                    feature_list.append(pool_metrics[key]['mean'])
            feature = np.asarray(feature_list,dtype=np.float32)
        else: #P1204.3
            feature_folder = './p1204/p1204_lbvfr_features'
            featfile_name = vid+'.z'
            try:
                feat_file = load(os.path.join(feature_folder,featfile_name))
                score = scores[i]
            except:
                print(featfile_name, ' was not found')
                continue

            feature = feat_file['features']

        feature = np.nan_to_num(feature)
        #        if(np.isnan(feature).any()):
#            print(vid)
        if(scores_df.loc[i]['content'] in train):
            train_features.append(feature)
            train_scores.append(score)
            train_indices.append(i)
            train_names.append(scores_df.loc[i]['video'])

        elif(scores_df.loc[i]['content'] in val):
            val_features.append(feature)
            val_scores.append(score)
            val_names.append(scores_df.loc[i]['video'])
#    print('Train set')
#    print(len(train_names))
#    print('Validation set')
#    print(len(val_names))
    return np.asarray(train_features),train_scores,np.asarray(val_features),val_scores,train,val_names

def single_split(trainval_content,cv_index,gamma,C):

    train_features,train_scores,val_features,val_scores,_,_ = trainval_split(trainval_content,cv_index)
    clf = svm.SVR(gamma=gamma,C=C)
    scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
    #scaler = StandardScaler()
    X_train = scaler.fit_transform(train_features)
    X_test = scaler.transform(val_features)
    clf.fit(X_train,train_scores)
    return clf.score(X_test,val_scores)
def grid_search(gamma_list,C_list,trainval_content):
    best_score = -100
    best_C = C_list[0]
    best_gamma = gamma_list[0]
    for gamma in gamma_list:
        for C in C_list:
            cv_score = Parallel(n_jobs=-1)(delayed(single_split)(trainval_content,cv_index,gamma,C) for cv_index in range(5))
            avg_cv_score = np.average(cv_score)
            if(avg_cv_score>best_score):
                best_score = avg_cv_score
                best_C = C
                best_gamma = gamma
    return best_C,best_gamma

def train_test(r):
    train_features,train_scores,test_features,test_scores,trainval_content,test_names = trainval_split(scores_df['content'].unique(),r)
    best_C,best_gamma = grid_search(np.logspace(-7,2,10),np.logspace(1,10,10,base=2),trainval_content)

    scaler = MinMaxScaler(feature_range=(-1,1))
    scaler.fit(train_features)
    X_train = scaler.transform(train_features)
    X_test = scaler.transform(test_features)
    best_svr =SVR(gamma=best_gamma,C=best_C)
    best_svr.fit(X_train,train_scores)
    preds = best_svr.predict(X_test)
    preds_fitted = results(preds,test_scores)
    test_zip = list(zip(test_names,test_scores,preds_fitted))
    return test_zip
test_zips = Parallel(n_jobs=-1,verbose=0)(delayed(train_test)(i) for i in range(100))

nlist,nscores,npreds = unique_scores(test_zips)

from joblib import dump

X={'names':nlist,'preds':npreds,'dmos':nscores}
dump(X,'./vmaf/vmaf_rawavg_scores.z')
