from scipy.io import loadmat
import glob
import pandas as pd
feature_files= glob.glob('/media/zaixi/zaixi_nas/HDRproject/feats/fr_evaluate_HDRLIVE_correct/hdrvdp/*.mat')
all_feats = []
for ref in feature_files:
    ref_mat = loadmat(ref)
    feats = pd.DataFrame(ref_mat['featMap'][0][0][0])
    feats = feats.loc[ feats.sum(axis=1) != 0, :]
    feats = feats.mean(axis=0)
    feats = pd.DataFrame(feats).transpose()
    feats['video'] = ref.split('/')[-1][:-4]
    all_feats.append(feats)
all_feats = pd.concat(all_feats)
all_feats.to_csv('/media/zaixi/zaixi_nas/HDRproject/feats/fr_evaluate_HDRLIVE_correct/hdrvdp/hdrvdp_none_local_-0.5_31_.csv')