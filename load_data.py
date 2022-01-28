import numpy as np
import pandas as pd
import scipy.io
import tensorflow as tf
from preprocessing.denoise_signal import Fourier, SVD_denoise, Wavelet, Wavelet_denoise, savitzky_golay
from preprocessing.utils import get_spectrogram
import matplotlib.pyplot as plt

def load_all(opt):
    '''
    For all files, the following item in the variable name indicates:
        DE - drive end accelerometer data
        FE - fan end accelerometer data
        BA - base accelerometer data
        time - time series data
        RPM - rpm during testing

        Fault Diameter: 0.007"
        Motor Load (HP): 0
        Approx. Motor Speed (rpm): 1797
    '''
    num = 124600
    n   = 200

    Normal_0_name    = [[1, 0, 0, 0, 0, 0]]*int(num/n)
    B007_0_name      = [[0, 1, 0, 0, 0, 0]]*int(num/n)
    IR007_0_name     = [[0, 0, 1, 0, 0, 0]]*int(num/n)
    OR007_3_0_name   = [[0, 0, 0, 1, 0, 0]]*int(num/n)
    OR007_6_0_name   = [[0, 0, 0, 0, 1, 0]]*int(num/n)
    OR007_12_0_name  = [[0, 0, 0, 0, 0, 1]]*int(num/n)

    label = np.concatenate((Normal_0_name, B007_0_name, IR007_0_name, OR007_3_0_name, OR007_6_0_name, OR007_12_0_name))

    Normal_0 = scipy.io.loadmat('./data/Normal_0.mat')
    B007_0 = scipy.io.loadmat('./data/B007_0.mat')
    IR007_0 = scipy.io.loadmat('./data/IR007_0.mat')
    OR007_3_0 = scipy.io.loadmat('./data/OR007_3_0.mat')
    OR007_6_0 = scipy.io.loadmat('./data/OR007_6_0.mat')
    OR007_12_0 = scipy.io.loadmat('./data/OR007_12_0.mat')

    # Data 21-----------------------------------------------
    # B007_0 = scipy.io.loadmat('./data/21/B021_0.mat')
    # IR007_0 = scipy.io.loadmat('./data/21/IR021_0.mat')
    # OR007_3_0 = scipy.io.loadmat('./data/21/OR021_3_0.mat')
    # OR007_6_0 = scipy.io.loadmat('./data/21/OR021_6_0.mat')
    # OR007_12_0 = scipy.io.loadmat('./data/21/OR021_12_0.mat')
    # all_labels = {0: 'Normal_0', 1: 'B007_0', 2: 'IR007_0', 3: 'OR007_3_0', 4: 'OR007_6_0', 5: 'OR007_12_0'}

    Normal_0_X097_DE_time   = Normal_0['X097_DE_time'][:num]
    Normal_0_X097_FE_time   = Normal_0['X097_FE_time'][:num]
    B007_0_X122_DE_time     = B007_0['X122_DE_time'][:num]
    B007_0_X122_FE_time     = B007_0['X122_FE_time'][:num]
    IR007_0_X122_DE_time    = IR007_0['X109_DE_time'][:num]
    IR007_0_X122_FE_time    = IR007_0['X109_FE_time'][:num]
    OR007_3_0_X122_DE_time  = OR007_3_0['X148_DE_time'][:num]
    OR007_3_0_X122_FE_time  = OR007_3_0['X148_FE_time'][:num]
    OR007_6_0_X122_DE_time  = OR007_6_0['X135_DE_time'][:num]
    OR007_6_0_X122_FE_time  = OR007_6_0['X135_FE_time'][:num]
    OR007_12_0_X122_DE_time = OR007_12_0['X161_DE_time'][:num]
    OR007_12_0_X122_FE_time = OR007_12_0['X161_FE_time'][:num]

    #data 21--------------------------------------------------------------
    # B007_0_X122_DE_time     = B007_0['X226_DE_time'][:num]
    # B007_0_X122_FE_time     = B007_0['X226_FE_time'][:num]
    # IR007_0_X122_DE_time    = IR007_0['X213_DE_time'][:num]
    # IR007_0_X122_FE_time    = IR007_0['X213_FE_time'][:num]
    # OR007_3_0_X122_DE_time  = OR007_3_0['X250_DE_time'][:num]
    # OR007_3_0_X122_FE_time  = OR007_3_0['X250_FE_time'][:num]
    # OR007_6_0_X122_DE_time  = OR007_6_0['X238_DE_time'][:num]
    # OR007_6_0_X122_FE_time  = OR007_6_0['X238_FE_time'][:num]
    # OR007_12_0_X122_DE_time = OR007_12_0['X262_DE_time'][:num]
    # OR007_12_0_X122_FE_time = OR007_12_0['X262_FE_time'][:num]

    # Normal_0_X097RPM       = Normal_0['X097RPM']
    # B007_0_X122RPM         = B007_0['X226RPM']
    # IR007_0_X122RPM        = IR007_0['X213RPM']
    # OR007_3_0_X122RPM      = OR007_3_0['X250RPM']
    # OR007_6_0_X122RPM      = OR007_6_0['X238RPM']
    # OR007_12_0_X122RPM     = OR007_12_0['X262RPM']


    if opt.denoise == 'DFK':
        Normal_0_X097_DE_time   = Fourier(f=Normal_0_X097_DE_time, num=num, get_result=True, thres=25).reshape(num, 1)
        Normal_0_X097_FE_time   = Fourier(f=Normal_0_X097_FE_time, num=num, get_result=True, thres=85).reshape(num, 1)

        B007_0_X122_DE_time     = Fourier(f=B007_0_X122_DE_time, num=num, get_result=True, thres=25).reshape(num, 1)
        B007_0_X122_FE_time     = Fourier(f=B007_0_X122_FE_time, num=num, get_result=True, thres=85).reshape(num, 1)

        IR007_0_X122_DE_time    = Fourier(f=IR007_0_X122_DE_time, num=num, get_result=True, thres=25).reshape(num, 1)
        IR007_0_X122_FE_time    = Fourier(f=IR007_0_X122_FE_time, num=num, get_result=True, thres=85).reshape(num, 1)

        OR007_3_0_X122_DE_time  = Fourier(f=OR007_3_0_X122_DE_time, num=num, get_result=True, thres=25).reshape(num, 1)
        OR007_3_0_X122_FE_time  = Fourier(f=OR007_3_0_X122_FE_time, num=num, get_result=True, thres=85).reshape(num, 1)

        OR007_6_0_X122_DE_time  = Fourier(f=OR007_6_0_X122_DE_time, num=num, get_result=True, thres=25).reshape(num, 1)
        OR007_6_0_X122_FE_time  = Fourier(f=OR007_6_0_X122_FE_time, num=num, get_result=True, thres=85).reshape(num, 1)

        OR007_12_0_X122_DE_time = Fourier(f=OR007_12_0_X122_DE_time, num=num, get_result=True, thres=25).reshape(num, 1)
        OR007_12_0_X122_FE_time = Fourier(f=OR007_12_0_X122_FE_time, num=num, get_result=True, thres=85).reshape(num, 1)

        DE_time = np.concatenate((Normal_0_X097_DE_time.reshape(int(num/n), n), B007_0_X122_DE_time.reshape(int(num/n), n), IR007_0_X122_DE_time.reshape(int(num/n), n), OR007_3_0_X122_DE_time.reshape(int(num/n), n), OR007_6_0_X122_DE_time.reshape(int(num/n), n), OR007_12_0_X122_DE_time.reshape(int(num/n), n)))
        FE_time = np.concatenate((Normal_0_X097_FE_time.reshape(int(num/n), n), B007_0_X122_FE_time.reshape(int(num/n), n), IR007_0_X122_FE_time.reshape(int(num/n), n), OR007_3_0_X122_FE_time.reshape(int(num/n), n), OR007_6_0_X122_FE_time.reshape(int(num/n), n), OR007_12_0_X122_FE_time.reshape(int(num/n), n)))

        merge_data = np.concatenate((DE_time, FE_time), axis=1)

    if opt.denoise == 'savitzky_golay':
        window_size=15
        Normal_0_X097_DE_time   = savitzky_golay(y=Normal_0_X097_DE_time, window_size=window_size, order=4, range_y=num).reshape(num, 1)
        Normal_0_X097_FE_time   = savitzky_golay(y=Normal_0_X097_FE_time, window_size=window_size, order=4, range_y=num).reshape(num, 1)

        B007_0_X122_DE_time     = savitzky_golay(y=B007_0_X122_DE_time, window_size=window_size, order=4, range_y=num).reshape(num, 1)
        B007_0_X122_FE_time     = savitzky_golay(y=B007_0_X122_FE_time, window_size=window_size, order=4, range_y=num).reshape(num, 1)

        IR007_0_X122_DE_time    = savitzky_golay(y=IR007_0_X122_DE_time, window_size=window_size, order=4, range_y=num).reshape(num, 1)
        IR007_0_X122_FE_time    = savitzky_golay(y=IR007_0_X122_FE_time, window_size=window_size, order=4, range_y=num).reshape(num, 1)

        OR007_3_0_X122_DE_time  = savitzky_golay(y=OR007_3_0_X122_DE_time, window_size=window_size, order=4, range_y=num).reshape(num, 1)
        OR007_3_0_X122_FE_time  = savitzky_golay(y=OR007_3_0_X122_FE_time, window_size=window_size, order=4, range_y=num).reshape(num, 1)

        OR007_6_0_X122_DE_time  = savitzky_golay(y=OR007_6_0_X122_DE_time, window_size=window_size, order=4, range_y=num).reshape(num, 1)
        OR007_6_0_X122_FE_time  = savitzky_golay(y=OR007_6_0_X122_FE_time, window_size=window_size, order=4, range_y=num).reshape(num, 1)

        OR007_12_0_X122_DE_time = savitzky_golay(y=OR007_12_0_X122_DE_time, window_size=window_size, order=4, range_y=num).reshape(num, 1)
        OR007_12_0_X122_FE_time = savitzky_golay(y=OR007_12_0_X122_FE_time, window_size=window_size, order=4, range_y=num).reshape(num, 1)

        DE_time = np.concatenate((Normal_0_X097_DE_time.reshape(int(num/n), n), B007_0_X122_DE_time.reshape(int(num/n), n), IR007_0_X122_DE_time.reshape(int(num/n), n), OR007_3_0_X122_DE_time.reshape(int(num/n), n), OR007_6_0_X122_DE_time.reshape(int(num/n), n), OR007_12_0_X122_DE_time.reshape(int(num/n), n)))
        FE_time = np.concatenate((Normal_0_X097_FE_time.reshape(int(num/n), n), B007_0_X122_FE_time.reshape(int(num/n), n), IR007_0_X122_FE_time.reshape(int(num/n), n), OR007_3_0_X122_FE_time.reshape(int(num/n), n), OR007_6_0_X122_FE_time.reshape(int(num/n), n), OR007_12_0_X122_FE_time.reshape(int(num/n), n)))

        merge_data = np.concatenate((DE_time, FE_time), axis=1)


    if opt.denoise == 'Wavelet_denoise':
        Normal_0_X097_DE_time = Wavelet_denoise(Normal_0_X097_DE_time)
        Normal_0_X097_FE_time = Wavelet_denoise(Normal_0_X097_FE_time)

        B007_0_X122_DE_time = Wavelet_denoise(B007_0_X122_DE_time)
        B007_0_X122_FE_time = Wavelet_denoise(B007_0_X122_FE_time)

        IR007_0_X122_DE_time = Wavelet_denoise(IR007_0_X122_DE_time)
        IR007_0_X122_FE_time = Wavelet_denoise(IR007_0_X122_FE_time)

        OR007_3_0_X122_DE_time = Wavelet_denoise(OR007_3_0_X122_DE_time)
        OR007_3_0_X122_FE_time = Wavelet_denoise(OR007_3_0_X122_FE_time)

        OR007_6_0_X122_DE_time = Wavelet_denoise(OR007_6_0_X122_DE_time)
        OR007_6_0_X122_FE_time = Wavelet_denoise(OR007_6_0_X122_FE_time)

        OR007_12_0_X122_DE_time = Wavelet_denoise(OR007_12_0_X122_DE_time)
        OR007_12_0_X122_FE_time = Wavelet_denoise(OR007_12_0_X122_FE_time)

        DE_time = np.concatenate((Normal_0_X097_DE_time.reshape(int(num/n), n), B007_0_X122_DE_time.reshape(int(num/n), n), IR007_0_X122_DE_time.reshape(int(num/n), n), OR007_3_0_X122_DE_time.reshape(int(num/n), n), OR007_6_0_X122_DE_time.reshape(int(num/n), n), OR007_12_0_X122_DE_time.reshape(int(num/n), n)))
        FE_time = np.concatenate((Normal_0_X097_FE_time.reshape(int(num/n), n), B007_0_X122_FE_time.reshape(int(num/n), n), IR007_0_X122_FE_time.reshape(int(num/n), n), OR007_3_0_X122_FE_time.reshape(int(num/n), n), OR007_6_0_X122_FE_time.reshape(int(num/n), n), OR007_12_0_X122_FE_time.reshape(int(num/n), n)))

        merge_data = np.concatenate((DE_time, FE_time), axis=1)

    if opt.denoise == 'SVD':
        Normal_0_X097_DE_time = SVD_denoise(Normal_0_X097_DE_time)
        Normal_0_X097_FE_time = SVD_denoise(Normal_0_X097_FE_time)

        B007_0_X122_DE_time = SVD_denoise(B007_0_X122_DE_time)
        B007_0_X122_FE_time = SVD_denoise(B007_0_X122_FE_time)

        IR007_0_X122_DE_time = SVD_denoise(IR007_0_X122_DE_time)
        IR007_0_X122_FE_time = SVD_denoise(IR007_0_X122_FE_time)

        OR007_3_0_X122_DE_time = SVD_denoise(OR007_3_0_X122_DE_time)
        OR007_3_0_X122_FE_time = SVD_denoise(OR007_3_0_X122_FE_time)

        OR007_6_0_X122_DE_time = SVD_denoise(OR007_6_0_X122_DE_time)
        OR007_6_0_X122_FE_time = SVD_denoise(OR007_6_0_X122_FE_time)

        OR007_12_0_X122_DE_time = SVD_denoise(OR007_12_0_X122_DE_time)
        OR007_12_0_X122_FE_time = SVD_denoise(OR007_12_0_X122_FE_time)

        DE_time = np.concatenate((Normal_0_X097_DE_time.reshape(int(num/n), n), B007_0_X122_DE_time.reshape(int(num/n), n), IR007_0_X122_DE_time.reshape(int(num/n), n), OR007_3_0_X122_DE_time.reshape(int(num/n), n), OR007_6_0_X122_DE_time.reshape(int(num/n), n), OR007_12_0_X122_DE_time.reshape(int(num/n), n)))
        FE_time = np.concatenate((Normal_0_X097_FE_time.reshape(int(num/n), n), B007_0_X122_FE_time.reshape(int(num/n), n), IR007_0_X122_FE_time.reshape(int(num/n), n), OR007_3_0_X122_FE_time.reshape(int(num/n), n), OR007_6_0_X122_FE_time.reshape(int(num/n), n), OR007_12_0_X122_FE_time.reshape(int(num/n), n)))

        merge_data = np.concatenate((DE_time, FE_time), axis=1)

    if opt.DNN_B:
        DE_time = np.concatenate((Normal_0_X097_DE_time.reshape(int(num/n), n), B007_0_X122_DE_time.reshape(int(num/n), n), IR007_0_X122_DE_time.reshape(int(num/n), n), OR007_3_0_X122_DE_time.reshape(int(num/n), n), OR007_6_0_X122_DE_time.reshape(int(num/n), n), OR007_12_0_X122_DE_time.reshape(int(num/n), n)))
        FE_time = np.concatenate((Normal_0_X097_FE_time.reshape(int(num/n), n), B007_0_X122_FE_time.reshape(int(num/n), n), IR007_0_X122_FE_time.reshape(int(num/n), n), OR007_3_0_X122_FE_time.reshape(int(num/n), n), OR007_6_0_X122_FE_time.reshape(int(num/n), n), OR007_12_0_X122_FE_time.reshape(int(num/n), n)))
        merge_data = np.concatenate((DE_time, FE_time), axis=1)
    elif opt.DNN_A or opt.CNN_A:
        Normal_0_group     = np.concatenate((Normal_0_X097_DE_time.reshape(int(num/n), n), Normal_0_X097_FE_time.reshape(int(num/n), n)), axis=1)
        B007_0_group       = np.concatenate((B007_0_X122_DE_time.reshape(int(num/n), n), B007_0_X122_FE_time.reshape(int(num/n), n)), axis=1)
        IR007_0_group      = np.concatenate((IR007_0_X122_DE_time.reshape(int(num/n), n), IR007_0_X122_FE_time.reshape(int(num/n), n)), axis=1)
        OR007_3_0_group    = np.concatenate((OR007_3_0_X122_DE_time.reshape(int(num/n), n), OR007_3_0_X122_FE_time.reshape(int(num/n), n)), axis=1)
        OR007_6_0_group    = np.concatenate((OR007_6_0_X122_DE_time.reshape(int(num/n), n), OR007_6_0_X122_FE_time.reshape(int(num/n), n)), axis=1)
        OR007_12_0_group   = np.concatenate((OR007_12_0_X122_DE_time.reshape(int(num/n), n), OR007_12_0_X122_FE_time.reshape(int(num/n), n)), axis=1)
        merge_data         = np.concatenate((Normal_0_group, B007_0_group, IR007_0_group, OR007_3_0_group, OR007_6_0_group, OR007_12_0_group))
    elif opt.CNN_B:
        Normal_0_X097_DE_time   = Normal_0_X097_DE_time.reshape((num//200, 200))
        Normal_0_X097_FE_time   = Normal_0_X097_FE_time.reshape((num//200, 200))
        Normal_0_X097 = np.concatenate((Normal_0_X097_DE_time, Normal_0_X097_FE_time), axis=1)

        B007_0_X122_DE_time     = B007_0_X122_DE_time.reshape((num//200, 200))
        B007_0_X122_FE_time     = B007_0_X122_FE_time.reshape((num//200, 200))
        B007_0_X122 = np.concatenate((B007_0_X122_DE_time, B007_0_X122_FE_time), axis=1)

        IR007_0_X122_DE_time    = IR007_0_X122_DE_time.reshape((num//200, 200))
        IR007_0_X122_FE_time    = IR007_0_X122_FE_time.reshape((num//200, 200))
        IR007_0_X122 = np.concatenate((IR007_0_X122_DE_time, IR007_0_X122_FE_time), axis=1)

        OR007_3_0_X122_DE_time  = OR007_3_0_X122_DE_time.reshape((num//200, 200))
        OR007_3_0_X122_FE_time  = OR007_3_0_X122_FE_time.reshape((num//200, 200))
        OR007_3_0_X122 = np.concatenate((OR007_3_0_X122_DE_time, OR007_3_0_X122_FE_time), axis=1)

        OR007_6_0_X122_DE_time  = OR007_6_0_X122_DE_time.reshape((num//200, 200))
        OR007_6_0_X122_FE_time  = OR007_6_0_X122_FE_time.reshape((num//200, 200))
        OR007_6_0_X122 = np.concatenate((OR007_6_0_X122_DE_time, OR007_6_0_X122_FE_time), axis=1)

        OR007_12_0_X122_DE_time = OR007_12_0_X122_DE_time.reshape((num//200, 200))
        OR007_12_0_X122_FE_time = OR007_12_0_X122_FE_time.reshape((num//200, 200))
        OR007_12_0_X122 = np.concatenate((OR007_12_0_X122_DE_time, OR007_12_0_X122_FE_time), axis=1)

        Normal_0_X097 = np.array([get_spectrogram(i) for i in Normal_0_X097])
        B007_0_X122 = np.array([get_spectrogram(i) for i in B007_0_X122])
        IR007_0_X122 = np.array([get_spectrogram(i) for i in IR007_0_X122])
        OR007_3_0_X122 = np.array([get_spectrogram(i) for i in OR007_3_0_X122])
        OR007_6_0_X122 = np.array([get_spectrogram(i) for i in OR007_6_0_X122])
        OR007_12_0_X122 = np.array([get_spectrogram(i) for i in OR007_12_0_X122])

        merge_data = np.concatenate((Normal_0_X097, B007_0_X122, IR007_0_X122, OR007_3_0_X122, OR007_6_0_X122, OR007_12_0_X122))
