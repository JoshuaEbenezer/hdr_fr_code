from models.vbliinds.vbliinds import vbliinds
import argparse,glob
from os.path import join,basename,exists
import os
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument('--vid_path', type=str, default='test.mp4', \
                    help='Path to video', metavar='')
parser.add_argument('--bit_depth', type=str, default='8', \
                    help='8, or 16', metavar='')
parser.add_argument('--model', type=str, default='8', \
                    help='vbliinds,videval', metavar='')

args = parser.parse_args()

        if args.model.lower() == 'hdrvdp1':
            cmd = f'matlab -r "addpath(\'./models/VIDEVAL/include/\');calc_VIDEVAL_feats(\'{add_upscale(vname)}\',{fps},{f_count});exit;" '
        elif args.model.lower() == 'hdrvdp2':
            cmd = f'matlab -r "addpath(\'./models/RAPIQUE/include/\');calc_RAPIQUE_features(\'{add_upscale(vname)}\',3840,2160,{fps},{f_count},512.0,resnet50,\'avg_pool\',1);exit;" '
        elif args.model.lower() == 'hdrvdp3':
            cmd = f'matlab -r "addpath(\'./hdrvdp-3.0.6/\');hdrvdp3(\'{add_upscale(vname)}\',[3840,2160],{fps},{f_count});exit;" '

        print(cmd)
        os.system(cmd)

hdrvdp3( task, test, reference, color_encoding, pixels_per_degree, options )
