import glob
import os
import pandas as pd

files = glob.glob('tmp/*.csv')

df = pd.concat((pd.read_csv(f, index_col=0) for f in files), ignore_index=True)
print(df)
df.to_csv('/media/zaixi/zaixi_nas/HDRproject/feats/fr_evaluate_HDRAQ_correct/greed/greed_none_local_-0.5_31_.csv')
