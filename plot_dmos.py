import pandas as pd
from joblib import load
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob

scores_df = pd.read_csv('../score_cleanup_code/lbvfr_dmos_from_raw_avg_mos.csv')

codec = sorted(scores_df['codec'].unique())
content = sorted(scores_df['content'].unique())
fr = scores_df['fr'].unique()
res = sorted(scores_df['res'].unique())
bitrate = sorted(scores_df['bitrate'].unique(),key=int)


def plot_by_col(col_vals,col_name):
    scores = []
    for val in col_vals:
        scores.append(scores_df[scores_df[col_name]==val].dmos.values)
    plt.boxplot(scores,labels=col_vals)
    plt.xlabel(col_name)
    plt.ylabel('MOS')
    plt.savefig('./plots/rawavgmos_HFR_v_SFR_'+codec+'_DMOS/'+val+'.png')
    plt.close()

#plot_by_col(content,'content')
#plot_by_col(res,'res')
#plot_by_col(codec,'codec')
#plot_by_col(bitrate,'bitrate')
#plot_by_col(fr,'fr')
#

def sort_by_x(x,y):
    new_x, new_y = zip(*sorted(zip(x, y)))
    return new_x,new_y
def plot_by_fr(col_vals,col_name,codec,scores_df,preds_df):
    
    for val in col_vals:
        hfr_bitrates = scores_df[(scores_df[col_name]==val) & (scores_df['fr']=='HFR')& (scores_df['codec']==codec)].bitrate.values
        sfr_bitrates = scores_df[(scores_df[col_name]==val) & (scores_df['fr']=='SFR')& (scores_df['codec']==codec)].bitrate.values
        if(len(sfr_bitrates)<6 or len(hfr_bitrates)<6):
            continue
        hfr_scores = scores_df[(scores_df[col_name]==val) & (scores_df['fr']=='HFR')& (scores_df['codec']==codec)].dmos.values
        sfr_scores = scores_df[(scores_df[col_name]==val) & (scores_df['fr']=='SFR')& (scores_df['codec']==codec)].dmos.values
        sfr_bitrates,sfr_scores = sort_by_x(sfr_bitrates,sfr_scores)
        print(sfr_bitrates)
        hfr_bitrates,hfr_scores = sort_by_x(hfr_bitrates,hfr_scores)
        plt.figure()
        plt.plot(hfr_bitrates,hfr_scores,'g+',color='green',linestyle='dashed')
        #for i,txt in enumerate(res_vals):
        #    plt.annotate(txt, (hfr_bitrates[i], hfr_scores[i]))
        plt.plot(sfr_bitrates,sfr_scores,'r+',color='red',linestyle='dashed')

        hfr_bitrates = preds_df[(preds_df[col_name]==val) & (preds_df['fr']=='HFR')& (preds_df['codec']==codec)].bitrate.values
        sfr_bitrates = preds_df[(preds_df[col_name]==val) & (preds_df['fr']=='SFR')& (preds_df['codec']==codec)].bitrate.values
        if(len(sfr_bitrates)<6 or len(hfr_bitrates)<6):
            plt.close()
            continue
        #if(len(sfr_bitrates)<6 or len(hfr_bitrates)<6):
        #    continue
        hfr_scores = preds_df[(preds_df[col_name]==val) & (preds_df['fr']=='HFR')& (preds_df['codec']==codec)].preds.values
        sfr_scores = preds_df[(preds_df[col_name]==val) & (preds_df['fr']=='SFR')& (preds_df['codec']==codec)].preds.values
        sfr_bitrates,sfr_scores = sort_by_x(sfr_bitrates,sfr_scores)
        print(sfr_bitrates)
        hfr_bitrates,hfr_scores = sort_by_x(hfr_bitrates,hfr_scores)
        plt.plot(hfr_bitrates,hfr_scores,'b+',color='blue',linestyle='dashed')
        #for i,txt in enumerate(res_vals):
        #    plt.annotate(txt, (hfr_bitrates[i], hfr_scores[i]))
        plt.plot(sfr_bitrates,sfr_scores,'c+',color='cyan',linestyle='dashed')
        #for i,txt in enumerate(res_vals):
        #    plt.annotate(txt, (sfr_bitrates[i], sfr_scores[i]))
        plt.legend([codec+" HFR",codec+ "SFR",'P1204 '+codec+" HFR",'P1204 '+ codec+ "SFR"])
        plt.xlabel('Bitrate (kbps)')
        plt.ylabel('DMOS')
        plt.title(val)
        plt.savefig('./plots/rawavgmos_HFR_v_SFR_'+codec+'_vmaf_preds/'+val+'.png')
        plt.close()
#        plt.show()




def res_from_content(content):
    if(content=='EPLDay' or content=='EPLNight' ):
        res = '3840x2160'
    elif(content=='TNFF' or content=='TNFNFL' or content=='USOpen'):
        res  = '1280x720'
    elif(content=='Cricket1' or content=='Cricket2'):
        res = '1440x1080'
    return res

def fps_from_content(content,fr):
    if(content=='EPLDay' or content=='EPLNight' or content=='Cricket1' or content=='Cricket2' or content=='USOpen'):
        if(fr=='HFR'):
            fps = 50
        else:
            fps = 25
    elif(content=='TNFF' or content=='TNFNFL'):
        if(fr=='HFR'):
            fps = 59.94
        else:
            fps = 29.97
    return fps

def expand_res_name(res_shorthand,content):
    print(res_shorthand)
    if(res_shorthand=='720p'):
        resolution='1280x720'
    elif(res_shorthand == '540p'):
        resolution='960x540'
    elif(res_shorthand == '396p'):
        resolution='704x396'
    elif(res_shorthand == '288p'):
        resolution='512x288'
    elif(res_shorthand=='SRC'):
        resolution = res_from_content(content)
    return resolution

pred_dict = load('./vmaf/vmaf_scores.z')
preds_df = pd.DataFrame.from_dict(pred_dict)
video_names = preds_df['names']

content = []
res = []
bitrate = []
fps = []
codec = []
begin_times = []


for vid in video_names:
    split_name = vid.split('_')
    print(split_name)
    content.append(split_name[0]+'_'+split_name[5])
    codec.append(split_name[1])
    res_shorthand = split_name[3]
    res.append(expand_res_name(res_shorthand,split_name[0]))
    bitrate.append(int(split_name[4][:-1]))
    begin_times.append(split_name[5][:-4])
    fr = split_name[2]
    fps.append(fr) 


preds_df['content'] = content
preds_df['fr'] = fps
preds_df['codec']=codec
preds_df['resolution']=res
preds_df['bitrate'] = bitrate
preds_df['begin_time']=begin_times
print(preds_df)
print(scores_df)
print(scores_df['codec'])
plot_by_fr(content,'content','HEVC',scores_df,preds_df)
#plot_by_fr(content,'content','AVC')
#res_fr_scores = []
#res_fr_bitrates = []
#names = []
#for res_val in res:
#    for frame_rate in fr:
#        res_fr_scores.append(np.mean(scores_df[(scores_df['res']==res_val) & (scores_df['fr']==frame_rate)].dmos.values))
#        res_fr_bitrates.append(np.mean(scores_df[(scores_df['res']==res_val) & (scores_df['fr']==frame_rate)].bitrate.values))
#        names.append(res_val+frame_rate)
#res_fr_bitrates,res_fr_scores = sort_by_x(res_fr_bitrates,res_fr_scores)
#fig = plt.figure(figsize=(32.0, 5.0))
#plt.plot(res_fr_bitrates,res_fr_scores,'b+',color='blue',linestyle='dashed')
#
#plt.xticks(res_fr_bitrates,labels=names)
#plt.savefig('./plots/res_fr_dmos.png')
#plt.close()
