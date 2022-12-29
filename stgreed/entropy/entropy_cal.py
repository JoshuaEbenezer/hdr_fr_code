import numpy as np
from entropy.yuvRead import yuvRead_frame
import os
from entropy.entropy_params import est_params_ggd_temporal
from entropy.entropy_params import est_params_ggd
import scipy.signal
from skvideo.utils.mscn import gen_gauss_window
import scipy.ndimage
from skimage.filters import rank
from skimage.morphology import disk

import pdb


def compute_nl(Y, nl_method, nl_param, domain):
    h, w = np.shape(Y)
    Y = Y.astype(np.float32)
    if nl_method == 'none':
        return Y
    if(domain == 'global'):

        if(nl_method == 'one_exp'):
            Y = Y/1023.0
            avg = np.average(Y)
            Y_transform = np.exp(nl_param*(Y-avg))
        elif(nl_method == 'nakarushton'):
            Y_transform = Y/(Y+avg_luminance)
        elif(nl_method == 'sigmoid'):
            Y_transform = 1/(1+(np.exp(-(1e-3*(Y-avg_luminance)))))
        elif(nl_method == 'logit'):
            delta = nl_param
            Y_scaled = -0.99+1.98*(Y-np.amin(Y))/(1e-3+np.amax(Y)-np.amin(Y))
            Y_transform = np.log((1+(Y_scaled)**delta)/(1-(Y_scaled)**delta))
            if(delta % 2 == 0):
                Y_transform[Y < 0] = -Y_transform[Y < 0]
        elif(nl_method == 'exp'):
            delta = nl_param
            Y = -4+(Y-np.amin(Y)) * 8/(1e-3+np.amax(Y)-np.amin(Y))
            Y_transform = np.exp(np.abs(Y)**delta)-1
            Y_transform[Y < 0] = -Y_transform[Y < 0]
        elif(nl_method == 'custom'):
            Y = -0.99+(Y-np.amin(Y)) * 1.98/(1e-3+np.amax(Y)-np.amin(Y))
            Y_transform = transform(Y, 5)
    elif(domain == 'local'):
        if(nl_method == 'one_exp'):
            Y = Y/1023.0

            avg_window = gen_gauss_window(31//2, 7.0/6.0)
            mu_image = np.zeros((h, w), dtype=np.float32)
            Y = np.array(Y).astype('float32')
            scipy.ndimage.correlate1d(
                Y, avg_window, 0, mu_image, mode='constant')
            scipy.ndimage.correlate1d(
                mu_image, avg_window, 1, mu_image, mode='constant')
            Y_transform = np.exp(nl_param*(Y - mu_image))

        elif(nl_method == 'logit'):
            maxY = scipy.ndimage.maximum_filter(Y, size=(31, 31))
            minY = scipy.ndimage.minimum_filter(Y, size=(31, 31))
            delta = nl_param
            Y_scaled = -0.99+1.98*(Y-minY)/(1e-3+maxY-minY)
            Y_transform = np.log((1+(Y_scaled)**delta)/(1-(Y_scaled)**delta))
            if(delta % 2 == 0):
                Y_transform[Y < 0] = -Y_transform[Y < 0]
        elif(nl_method == 'texp'):
            patch_size = 31
            Y = Y/np.max(Y)
            avg_window = gen_gauss_window(patch_size//2, 7.0/6.0)
            mu_image = np.zeros((h, w), dtype=np.float32)
            image = np.array(Y).astype('float32')
            scipy.ndimage.correlate1d(
                image, avg_window, 0, mu_image, mode='constant')
            scipy.ndimage.correlate1d(mu_image, avg_window, 1,
                                      mu_image, mode='constant')
            Y_transform = np.exp(nl_param*(image - mu_image))

        elif(nl_method == 'lnl'):

            
            if(len(Y.shape)==2):
                Y = np.expand_dims(Y,axis=2)

            maxY = scipy.ndimage.maximum_filter(Y,size=(17,17,1))
            minY = scipy.ndimage.minimum_filter(Y,size=(17,17,1))
            Y_scaled = -1+(Y-minY)* 2/(1e-3+maxY-minY)
            Y_transform =  np.exp(np.abs(Y_scaled)*4)-1
            Y_transform[Y_scaled<0] = -Y_transform[Y_scaled<0]
            Y_transform = np.squeeze(Y_transform)
        elif(nl_method == 'lhe'):
            scale = np.max(Y)
            footprint = 31
            scale_neg = np.min(Y)
            # print(scale_neg)
            coef_1 = (Y-scale_neg)/(scale-scale_neg)
            coef_16 = coef_1*1023
            coef_16 = coef_16.astype(np.uint16)
            footprint = disk(footprint)
            img_eq_ref = rank.equalize(coef_16, selem=footprint)
            Y_transform = img_eq_ref.astype(
                np.float32)/1023*(scale-scale_neg)+scale_neg

        elif(nl_method == 'custom'):
            maxY = scipy.ndimage.maximum_filter(Y, size=(31, 31))
            minY = scipy.ndimage.minimum_filter(Y, size=(31, 31))
            Y = -0.99+(Y-minY) * 1.98/(1e-3+maxY-minY)
            Y_transform = transform(Y, 5)

        elif(nl_method == 'sigmoid'):
            avg_luminance = scipy.ndimage.gaussian_filter(Y, sigma=7.0/6.0)
            Y_transform = 1/(1+(np.exp(-(1e-3*(Y-avg_luminance)))))
    return Y_transform


def compute_MS_transform(image, window, extend_mode='reflect'):
    h, w = image.shape
    mu_image = np.zeros((h, w), dtype=np.float32)
    scipy.ndimage.correlate1d(image, window, 0, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(mu_image, window, 1, mu_image, mode=extend_mode)
    return image - mu_image


def video_process(vid_path, nl_method, nl_param, nl_domain, width, height, bit_depth, gray, T, filt, num_levels, scales):

    # Load WPT filters

    filt_path = 'stgreed/WPT_Filters/' + \
        filt + '_wpt_' + str(num_levels) + '.mat'
    wfun = scipy.io.loadmat(filt_path)
    wfun = wfun['wfun']

    blk = 5
    sigma_nsq = 0.1
    win_len = 7

    entropy = {}
    vid_stream = open(vid_path, 'r')

    for scale_factor in scales:
        print('Processing scale factor: ' +
              str(scale_factor), 'video: ' + vid_path)
        sz = 2**(-scale_factor)
        frame_data = np.zeros((int(height*sz), int(width*sz), T))

        spatial_sig = []
        spatial_ent = []
        for frame_ind in range(T):
            # for frame_ind in range(3):

            Y, _, _ = \
                yuvRead_frame(vid_stream, width, height,
                              frame_ind, bit_depth, gray, sz)
            frame_data[:, :, frame_ind] = compute_nl(
                Y, nl_method, nl_param, nl_domain)
            window = gen_gauss_window((win_len-1)/2, win_len/6)
            MS_frame = compute_MS_transform(
                frame_data[:, :, frame_ind], window)

            spatial_sig_frame, spatial_ent_frame = est_params_ggd(
                MS_frame, blk, sigma_nsq)
            spatial_sig.append(spatial_sig_frame)
            spatial_ent.append(spatial_ent_frame)

        # Wavelet Packet Filtering
        # valid indices for start and end points
        valid_lim = frame_data.shape[2] - wfun.shape[1] + 1
        start_ind = wfun.shape[1]//2 - 1
        dpt_filt = np.zeros((frame_data.shape[0], frame_data.shape[1],
                             2**num_levels - 1, valid_lim))

        for freq in range(wfun.shape[0]):
            dpt_filt[:, :, freq, :] = scipy.ndimage.filters.convolve1d(frame_data,
                                                                       wfun[freq, :], axis=2, mode='constant')[:, :, start_ind:start_ind + valid_lim]

        temporal_sig, temporal_ent = est_params_ggd_temporal(dpt_filt, blk,
                                                             sigma_nsq)

        # convert lists to numpy arrays
        spatial_sig = np.array(spatial_sig)
        spatial_sig[np.isinf(spatial_sig)] = 0

        spatial_ent = np.array(spatial_ent)
        spatial_ent[np.isinf(spatial_ent)] = 0

        temporal_sig = np.array(temporal_sig)
        temporal_sig[np.isinf(temporal_sig)] = 0

        temporal_ent = np.array(temporal_ent)
        temporal_ent[np.isinf(temporal_ent)] = 0

        # calculate rescaled entropy
        spatial_ent_scaled = np.log(1 + spatial_sig**2) * spatial_ent
        temporal_ent_scaled = np.log(1 + temporal_sig**2) * temporal_ent

        # reshape spatial entropy to heightxwidthxnum_frames
        spatial_ent_scaled = spatial_ent_scaled.transpose(1, 2, 0)

        entropy['spatial_scale' +
                str(scale_factor)] = spatial_ent_scaled[:, :, :valid_lim]
        entropy['temporal_scale' + str(scale_factor)] = temporal_ent_scaled

    return entropy
