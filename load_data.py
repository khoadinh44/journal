import numpy as np
import pandas as pd
import scipy.io
import tensorflow as tf
from preprocessing.denoise_signal import Fourier, SVD_denoise, Wavelet, Wavelet_denoise, savitzky_golay
from preprocessing.utils import get_spectrogram
import matplotlib.pyplot as plt

def one_hot(pos, num_class):
    num = np.zero((1, num_class))
    num[0, num_class] = 1
    return num

def concatenate_data(x):
  data=None
  for idx, i in enumerate(x):
    if idx == 3:
      data = x[i]
    if idx > 3:
      data = np.concatenate((data, x[i]), axis=0)
  return data.reshape((1, -1))

def load_all(opt, data_12k=False, data_48k=False, data_normal=False):
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
    
    if data_normal:
        Normal_0 = concatenate_data(scipy.io.loadmat('./data/normal/Normal_0.mat'))
        Normal_1 = concatenate_data(scipy.io.loadmat('./data/normal/Normal_1.mat'))
        Normal_2 = concatenate_data(scipy.io.loadmat('./data/normal/Normal_2.mat'))
        Normal_3 = concatenate_data(scipy.io.loadmat('./data/normal/Normal_3.mat'))
    
    if data_12k:
        B007_0 = concatenate_data(scipy.io.loadmat('./data/12k/B007_0.mat'))
        B007_0_label = one_hot(0, 60)
        B007_1 = concatenate_data(scipy.io.loadmat('./data/12k/B007_1.mat'))
        B007_1_label = one_hot(1, 60)
        B007_2 = concatenate_data(scipy.io.loadmat('./data/12k/B007_2.mat'))
        B007_2_label = one_hot(2, 60)
        B007_3 = concatenate_data(scipy.io.loadmat('./data/12k/B007_3.mat'))
        B007_3_label = one_hot(3, 60)

        B014_0 = concatenate_data(scipy.io.loadmat('./data/12k/B014_0.mat'))
        B014_0_label = one_hot(4, 60)
        B014_1 = concatenate_data(scipy.io.loadmat('./data/12k/B014_1.mat'))
        B014_1_label = one_hot(5, 60)
        B014_2 = concatenate_data(scipy.io.loadmat('./data/12k/B014_2.mat'))
        B014_2_label = one_hot(6, 60)
        B014_3 = concatenate_data(scipy.io.loadmat('./data/12k/B014_3.mat'))
        B014_3_label = one_hot(7, 60)
        
        B021_0 = concatenate_data(scipy.io.loadmat('./data/12k/B021_0.mat'))
        B021_0_label = one_hot(8, 60)
        B021_1 = concatenate_data(scipy.io.loadmat('./data/12k/B021_1.mat'))
        B021_1_label = one_hot(9, 60)
        B021_2 = concatenate_data(scipy.io.loadmat('./data/12k/B021_2.mat'))
        B021_2_label = one_hot(10, 60)
        B021_3 = concatenate_data(scipy.io.loadmat('./data/12k/B021_3.mat'))
        B021_3_label = one_hot(11, 60)

        B028_0 = concatenate_data(scipy.io.loadmat('./data/12k/B028_0.mat'))
        B028_0_label = one_hot(12, 60)
        B028_1 = concatenate_data(scipy.io.loadmat('./data/12k/B028_1.mat'))
        B028_1_label = one_hot(13, 60)
        B028_2 = concatenate_data(scipy.io.loadmat('./data/12k/B028_2.mat'))
        B028_2_label = one_hot(14, 60)
        B028_3 = concatenate_data(scipy.io.loadmat('./data/12k/B028_3.mat'))
        B028_3_label = one_hot(15, 60)

        IR007_0 = concatenate_data(scipy.io.loadmat('./data/12k/IR007_0.mat'))
        IR007_0_label = one_hot(16, 60)
        IR007_1 = concatenate_data(scipy.io.loadmat('./data/12k/IR007_1.mat'))
        IR007_1_label = one_hot(17, 60)
        IR007_2 = concatenate_data(scipy.io.loadmat('./data/12k/IR007_2.mat'))
        IR007_2_label = one_hot(18, 60)
        IR007_3 = concatenate_data(scipy.io.loadmat('./data/12k/IR007_3.mat'))
        IR007_3_label = one_hot(19, 60)

        IR014_0 = concatenate_data(scipy.io.loadmat('./data/12k/IR014_0.mat'))
        IR014_0_label = one_hot(20, 60)
        IR014_1 = concatenate_data(scipy.io.loadmat('./data/12k/IR014_1.mat'))
        IR014_1_label = one_hot(21, 60)
        IR014_2 = concatenate_data(scipy.io.loadmat('./data/12k/IR014_2.mat'))
        IR014_2_label = one_hot(22, 60)
        IR014_3 = concatenate_data(scipy.io.loadmat('./data/12k/IR014_3.mat'))
        IR014_3_label = one_hot(23, 60)

        IR021_0 = concatenate_data(scipy.io.loadmat('./data/12k/IR021_0.mat'))
        IR021_0_label = one_hot(24, 60)
        IR021_1 = concatenate_data(scipy.io.loadmat('./data/12k/IR021_1.mat'))
        IR021_1_label = one_hot(25, 60)
        IR021_2 = concatenate_data(scipy.io.loadmat('./data/12k/IR021_2.mat'))
        IR021_2_label = one_hot(26, 60)
        IR021_3 = concatenate_data(scipy.io.loadmat('./data/12k/IR021_3.mat'))
        IR021_3_label = one_hot(27, 60)

        IR028_0 = concatenate_data(scipy.io.loadmat('./data/12k/IR028_0.mat'))
        IR028_0_label = one_hot(28, 60)
        IR028_1 = concatenate_data(scipy.io.loadmat('./data/12k/IR028_1.mat'))
        IR028_1_label = one_hot(29, 60)
        IR028_2 = concatenate_data(scipy.io.loadmat('./data/12k/IR028_2.mat'))
        IR028_2_label = one_hot(30, 60)
        IR028_3 = concatenate_data(scipy.io.loadmat('./data/12k/IR028_3.mat'))
        IR028_3_label = one_hot(31, 60)

        OR007_12_0 = concatenate_data(scipy.io.loadmat('./data/12k/OR007@12_0.mat'))
        OR007_12_0_label = one_hot(32, 60)
        OR007_12_1 = concatenate_data(scipy.io.loadmat('./data/12k/OR007@12_1.mat'))
        OR007_12_1_label = one_hot(33, 60)
        OR007_12_2 = concatenate_data(scipy.io.loadmat('./data/12k/OR007@12_2.mat'))
        OR007_12_2_label = one_hot(34, 60)
        OR007_12_3 = concatenate_data(scipy.io.loadmat('./data/12k/OR007@12_3.mat'))
        OR007_12_3_label = one_hot(35, 60)

        OR007_3_0 = concatenate_data(scipy.io.loadmat('./data/12k/OR007@3_0.mat'))
        OR007_3_0_label = one_hot(36, 60)
        OR007_3_1 = concatenate_data(scipy.io.loadmat('./data/12k/OR007@3_1.mat'))
        OR007_3_1_label = one_hot(37, 60)
        OR007_3_2 = concatenate_data(scipy.io.loadmat('./data/12k/OR007@3_2.mat'))
        OR007_3_2_label = one_hot(38, 60)
        OR007_3_3 = concatenate_data(scipy.io.loadmat('./data/12k/OR007@3_3.mat'))
        OR007_3_3_label = one_hot(39, 60)

        OR007_6_0 = concatenate_data(scipy.io.loadmat('./data/12k/OR007@6_0.mat'))
        OR007_6_0_label = one_hot(40, 60)
        OR007_6_1 = concatenate_data(scipy.io.loadmat('./data/12k/OR007@6_1.mat'))
        OR007_6_1_label = one_hot(41, 60)
        OR007_6_2 = concatenate_data(scipy.io.loadmat('./data/12k/OR007@6_2.mat'))
        OR007_6_2_label = one_hot(42, 60)
        OR007_6_3 = concatenate_data(scipy.io.loadmat('./data/12k/OR007@6_3.mat'))
        OR007_6_3_label = one_hot(43, 60)

        OR0014_6_0 = concatenate_data(scipy.io.loadmat('./data/12k/OR014@6_0.mat'))
        OR0014_6_0_label = one_hot(44, 60)
        OR0014_6_1 = concatenate_data(scipy.io.loadmat('./data/12k/OR014@6_1.mat'))
        OR0014_6_1_label = one_hot(45, 60)
        OR0014_6_2 = concatenate_data(scipy.io.loadmat('./data/12k/OR014@6_2.mat'))
        OR0014_6_2_label = one_hot(46, 60)
        OR0014_6_3 = concatenate_data(scipy.io.loadmat('./data/12k/OR014@6_3.mat'))
        OR0014_6_3_label = one_hot(47, 60)

        OR0021_6_0 = concatenate_data(scipy.io.loadmat('./data/12k/OR021@6_0.mat'))
        OR0021_6_0_label = one_hot(48, 60)
        OR0021_6_1 = concatenate_data(scipy.io.loadmat('./data/12k/OR021@6_1.mat'))
        OR0021_6_1_label = one_hot(49, 60)
        OR0021_6_2 = concatenate_data(scipy.io.loadmat('./data/12k/OR021@6_2.mat'))
        OR0021_6_2_label = one_hot(50, 60)
        OR0021_6_3 = concatenate_data(scipy.io.loadmat('./data/12k/OR021@6_3.mat'))
        OR0021_6_3_label = one_hot(51, 60)

        OR0021_3_0 = concatenate_data(scipy.io.loadmat('./data/12k/OR021@3_0.mat'))
        OR0021_3_0_label = one_hot(52, 60)
        OR0021_3_1 = concatenate_data(scipy.io.loadmat('./data/12k/OR021@3_1.mat'))
        OR0021_3_1_label = one_hot(53, 60)
        OR0021_3_2 = concatenate_data(scipy.io.loadmat('./data/12k/OR021@3_2.mat'))
        OR0021_3_2_label = one_hot(54, 60)
        OR0021_3_3 = concatenate_data(scipy.io.loadmat('./data/12k/OR021@3_3.mat'))
        OR0021_3_3_label = one_hot(55, 60)

        OR0021_12_0 = concatenate_data(scipy.io.loadmat('./data/12k/OR021@12_0.mat'))
        OR0021_12_0_label = one_hot(56, 60)
        OR0021_12_1 = concatenate_data(scipy.io.loadmat('./data/12k/OR021@12_1.mat'))
        OR0021_12_1_label = one_hot(57, 60)
        OR0021_12_2 = concatenate_data(scipy.io.loadmat('./data/12k/OR021@12_2.mat'))
        OR0021_12_2_label = one_hot(58, 60)
        OR0021_12_3 = concatenate_data(scipy.io.loadmat('./data/12k/OR021@12_3.mat'))
        OR0021_12_3_label = one_hot(59, 60)
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

    if opt.use_DNN_B:
        DE_time = np.concatenate((Normal_0_X097_DE_time.reshape(int(num/n), n), B007_0_X122_DE_time.reshape(int(num/n), n), IR007_0_X122_DE_time.reshape(int(num/n), n), OR007_3_0_X122_DE_time.reshape(int(num/n), n), OR007_6_0_X122_DE_time.reshape(int(num/n), n), OR007_12_0_X122_DE_time.reshape(int(num/n), n)))
        FE_time = np.concatenate((Normal_0_X097_FE_time.reshape(int(num/n), n), B007_0_X122_FE_time.reshape(int(num/n), n), IR007_0_X122_FE_time.reshape(int(num/n), n), OR007_3_0_X122_FE_time.reshape(int(num/n), n), OR007_6_0_X122_FE_time.reshape(int(num/n), n), OR007_12_0_X122_FE_time.reshape(int(num/n), n)))
        merge_data = np.concatenate((DE_time, FE_time), axis=1)
    elif opt.use_DNN_A or opt.use_CNN_A or opt.use_CNN_C:
        Normal_0_group     = np.concatenate((Normal_0_X097_DE_time.reshape(int(num/n), n), Normal_0_X097_FE_time.reshape(int(num/n), n)), axis=1)
        B007_0_group       = np.concatenate((B007_0_X122_DE_time.reshape(int(num/n), n), B007_0_X122_FE_time.reshape(int(num/n), n)), axis=1)
        IR007_0_group      = np.concatenate((IR007_0_X122_DE_time.reshape(int(num/n), n), IR007_0_X122_FE_time.reshape(int(num/n), n)), axis=1)
        OR007_3_0_group    = np.concatenate((OR007_3_0_X122_DE_time.reshape(int(num/n), n), OR007_3_0_X122_FE_time.reshape(int(num/n), n)), axis=1)
        OR007_6_0_group    = np.concatenate((OR007_6_0_X122_DE_time.reshape(int(num/n), n), OR007_6_0_X122_FE_time.reshape(int(num/n), n)), axis=1)
        OR007_12_0_group   = np.concatenate((OR007_12_0_X122_DE_time.reshape(int(num/n), n), OR007_12_0_X122_FE_time.reshape(int(num/n), n)), axis=1)
        merge_data         = np.concatenate((Normal_0_group, B007_0_group, IR007_0_group, OR007_3_0_group, OR007_6_0_group, OR007_12_0_group))
    elif opt.use_CNN_B:
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
    return merge_data, label
