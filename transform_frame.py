import numpy as np
from skimage.metrics import structural_similarity as ssim
import os
import glob
import cv2
from joblib import Parallel,delayed,dump
import scipy.ndimage
import pandas as pd
import skimage.util
import math


def gen_gauss_window(lw, sigma):
    sd = np.float32(sigma)
    lw = int(lw)
    weights = [0.0] * (2 * lw + 1)
    weights[lw] = 1.0
    sum = 1.0
    sd *= sd
    for ii in range(1, lw + 1):
        tmp = np.exp(-0.5 * np.float32(ii * ii) / sd)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
        sum += 2.0 * tmp
    for ii in range(2 * lw + 1):
        weights[ii] /= sum
    return weights

class TransformFrame:
    def __init__ (self,nl_method,nl_param,nl_type='local',patch_size=31):
        self.patch_size = patch_size
        self.nl_method = nl_method
        self.nl_param = nl_param
        self.nl_type = nl_type

    def transform_frame(self,frame):
        if(self.nl_type=='local'):
            Y_transform = self.Y_compute_lnl(frame)
        elif(self.nl_type=='global'):
            Y_transform = self.Y_compute_gnl(frame)
        return Y_transform
        
    def Y_compute_gnl(self,Y):
        Y = Y.astype(np.float32)
        if(self.nl_method=='exp'):
            delta = self.nl_param
            Y = -4+(Y-np.amin(Y))* 8/(1e-3+np.amax(Y)-np.amin(Y))
            Y_transform =  np.exp(np.abs(Y)**delta)-1
            Y_transform[Y<0] = -Y_transform[Y<0]
        elif(self.nl_method=='texp'):
            if Y.max()>1:
                Y=Y/Y.max()
            Y = -4+(Y-np.amin(Y))* 8/(1e-3+np.amax(Y)-np.amin(Y))
            Y_transform =  np.exp(np.abs(Y)**delta)-1
            Y_transform[Y<0] = -Y_transform[Y<0]
        
    def Y_compute_lnl(self,Y):
        Y = Y.astype(np.float32)
        patch_size = self.patch_size
        if(self.nl_method=='exp'):
            maxY = scipy.ndimage.maximum_filter(Y,size=(patch_size,patch_size))
            minY = scipy.ndimage.minimum_filter(Y,size=(patch_size,patch_size))
            delta = self.nl_param
            Y = -4+(Y-minY)* 8/(1e-3+maxY-minY)
            Y_transform =  np.exp(np.abs(Y)**delta)-1
            Y_transform[Y<0] = -Y_transform[Y<0]
        elif self.nl_method == 'texp':
            assert len(np.shape(Y)) == 2
            h, w = np.shape(Y)
            if Y.max()>1:
                Y=Y/Y.max()
            avg_window = gen_gauss_window(patch_size//2, 7.0/6.0)
            mu_image = np.zeros((h, w), dtype=np.float32)
            Y = np.array(Y).astype('float32')
            scipy.ndimage.correlate1d(Y, avg_window, 0, mu_image, mode='constant')
            scipy.ndimage.correlate1d(mu_image, avg_window, 1, mu_image, mode='constant')
            Y_transform = np.exp(self.nl_param*(Y - mu_image))
        return Y_transform
            

