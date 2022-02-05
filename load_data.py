import numpy as np
import pandas as pd
import scipy.io
import tensorflow as tf
from preprocessing.denoise_signal import Fourier, SVD_denoise, Wavelet, Wavelet_denoise, savitzky_golay
from preprocessing.utils import get_spectrogram, one_hot
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


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
Normal_0_train, Normal_0_test = concatenate_data(scipy.io.loadmat('./data/normal/Normal_0.mat'), scale=StandardScaler())
Normal_0_label = one_hot(0, 64)
Normal_1_train, Normal_1_test = concatenate_data(scipy.io.loadmat('./data/normal/Normal_1.mat'), scale=StandardScaler())
Normal_0_label = one_hot(1, 64)
Normal_2_train, Normal_2_test = concatenate_data(scipy.io.loadmat('./data/normal/Normal_2.mat'), scale=StandardScaler())
Normal_0_label = one_hot(2, 64)
Normal_3_train, Normal_3_test = concatenate_data(scipy.io.loadmat('./data/normal/Normal_3.mat'), scale=StandardScaler())
Normal_0_label = one_hot(3, 64)

B007_0_train, B007_0_test = concatenate_data(scipy.io.loadmat('./data/12k/B007_0.mat'), scale=StandardScaler())
B007_0_label = one_hot(4, 64)
B007_1_train, B007_1_test = concatenate_data(scipy.io.loadmat('./data/12k/B007_1.mat'), scale=StandardScaler())
B007_1_label = one_hot(5, 64)
B007_2_train, B007_2_test = concatenate_data(scipy.io.loadmat('./data/12k/B007_2.mat'), scale=StandardScaler())
B007_2_label = one_hot(6, 64)
B007_3_train, B007_3_test = concatenate_data(scipy.io.loadmat('./data/12k/B007_3.mat'), scale=StandardScaler())
B007_3_label = one_hot(7, 64)

B014_0_train, B014_0_test = concatenate_data(scipy.io.loadmat('./data/12k/B014_0.mat'), scale=StandardScaler())
B014_0_label = one_hot(8, 64)
B014_1_train, B014_1_test = concatenate_data(scipy.io.loadmat('./data/12k/B014_1.mat'), scale=StandardScaler())
B014_1_label = one_hot(9, 64)
B014_2_train, B014_2_test = concatenate_data(scipy.io.loadmat('./data/12k/B014_2.mat'), scale=StandardScaler())
B014_2_label = one_hot(10, 64)
B014_3_train, B014_3_test = concatenate_data(scipy.io.loadmat('./data/12k/B014_3.mat'), scale=StandardScaler())
B014_3_label = one_hot(11, 64)

B021_0_train, B021_0_test = concatenate_data(scipy.io.loadmat('./data/12k/B021_0.mat'), scale=StandardScaler())
B021_0_label = one_hot(12, 64)
B021_1_train, B021_1_test = concatenate_data(scipy.io.loadmat('./data/12k/B021_1.mat'), scale=StandardScaler())
B021_1_label = one_hot(13, 64)
B021_2_train, B021_2_test = concatenate_data(scipy.io.loadmat('./data/12k/B021_2.mat'), scale=StandardScaler())
B021_2_label = one_hot(14, 64)
B021_3_train, B021_3_test = concatenate_data(scipy.io.loadmat('./data/12k/B021_3.mat'), scale=StandardScaler())
B021_3_label = one_hot(15, 64)

B028_0_train, B028_0_test = concatenate_data(scipy.io.loadmat('./data/12k/B028_0.mat'), scale=StandardScaler())
B028_0_label = one_hot(16, 64)
B028_1_train, B028_1_test = concatenate_data(scipy.io.loadmat('./data/12k/B028_1.mat'), scale=StandardScaler())
B028_1_label = one_hot(17, 64)
B028_2_train, B028_2_test = concatenate_data(scipy.io.loadmat('./data/12k/B028_2.mat'), scale=StandardScaler())
B028_2_label = one_hot(18, 64)
B028_3_train, B028_3_test = concatenate_data(scipy.io.loadmat('./data/12k/B028_3.mat'), scale=StandardScaler())
B028_3_label = one_hot(19, 64)

IR007_0_train, IR007_0_test = concatenate_data(scipy.io.loadmat('./data/12k/IR007_0.mat'), scale=StandardScaler())
IR007_0_label = one_hot(20, 64)
IR007_1_train, IR007_1_test = concatenate_data(scipy.io.loadmat('./data/12k/IR007_1.mat'), scale=StandardScaler())
IR007_1_label = one_hot(21, 64)
IR007_2_train, IR007_2_test = concatenate_data(scipy.io.loadmat('./data/12k/IR007_2.mat'), scale=StandardScaler())
IR007_2_label = one_hot(22, 64)
IR007_3_train, IR007_3_test = concatenate_data(scipy.io.loadmat('./data/12k/IR007_3.mat'), scale=StandardScaler())
IR007_3_label = one_hot(23, 64)

IR014_0_train, IR014_0_test = concatenate_data(scipy.io.loadmat('./data/12k/IR014_0.mat'), scale=StandardScaler())
IR014_0_label = one_hot(24, 64)
IR014_1_train, IR014_1_test = concatenate_data(scipy.io.loadmat('./data/12k/IR014_1.mat'), scale=StandardScaler())
IR014_1_label = one_hot(25, 64)
IR014_2_train, IR014_2_test = concatenate_data(scipy.io.loadmat('./data/12k/IR014_2.mat'), scale=StandardScaler())
IR014_2_label = one_hot(26, 64)
IR014_3_train, IR014_3_test = concatenate_data(scipy.io.loadmat('./data/12k/IR014_3.mat'), scale=StandardScaler())
IR014_3_label = one_hot(27, 64)

IR021_0_train, IR021_0_test = concatenate_data(scipy.io.loadmat('./data/12k/IR021_0.mat'), scale=StandardScaler())
IR021_0_label = one_hot(28, 64)
IR021_1_train, IR021_1_test = concatenate_data(scipy.io.loadmat('./data/12k/IR021_1.mat'), scale=StandardScaler())
IR021_1_label = one_hot(29, 64)
IR021_2_train, IR021_2_test = concatenate_data(scipy.io.loadmat('./data/12k/IR021_2.mat'), scale=StandardScaler())
IR021_2_label = one_hot(30, 64)
IR021_3_train, IR021_3_test = concatenate_data(scipy.io.loadmat('./data/12k/IR021_3.mat'), scale=StandardScaler())
IR021_3_label = one_hot(31, 64)

IR028_0_train, IR028_0_test = concatenate_data(scipy.io.loadmat('./data/12k/IR028_0.mat'), scale=StandardScaler())
IR028_0_label = one_hot(32, 64)
IR028_1_train, IR028_1_test = concatenate_data(scipy.io.loadmat('./data/12k/IR028_1.mat'), scale=StandardScaler())
IR028_1_label = one_hot(33, 64)
IR028_2_train, IR028_2_test = concatenate_data(scipy.io.loadmat('./data/12k/IR028_2.mat'), scale=StandardScaler())
IR028_2_label = one_hot(34, 64)
IR028_3_train, IR028_3_test = concatenate_data(scipy.io.loadmat('./data/12k/IR028_3.mat'), scale=StandardScaler())
IR028_3_label = one_hot(35, 64)

OR007_12_0_train, OR007_12_0_test = concatenate_data(scipy.io.loadmat('./data/12k/OR007@12_0.mat'), scale=StandardScaler())
OR007_12_0_label = one_hot(36, 64)
OR007_12_1_train, OR007_12_1_test = concatenate_data(scipy.io.loadmat('./data/12k/OR007@12_1.mat'), scale=StandardScaler())
OR007_12_1_label = one_hot(37, 64)
OR007_12_2_train, OR007_12_2_test = concatenate_data(scipy.io.loadmat('./data/12k/OR007@12_2.mat'), scale=StandardScaler())
OR007_12_2_label = one_hot(38, 64)
OR007_12_3_train, OR007_12_3_test = concatenate_data(scipy.io.loadmat('./data/12k/OR007@12_3.mat'), scale=StandardScaler())
OR007_12_3_label = one_hot(39, 64)

OR007_3_0_train, OR007_3_0_test = concatenate_data(scipy.io.loadmat('./data/12k/OR007@3_0.mat'), scale=StandardScaler())
OR007_3_0_label = one_hot(40, 64)
OR007_3_1_train, OR007_3_1_test = concatenate_data(scipy.io.loadmat('./data/12k/OR007@3_1.mat'), scale=StandardScaler())
OR007_3_1_label = one_hot(41, 64)
OR007_3_2_train, OR007_3_2_test = concatenate_data(scipy.io.loadmat('./data/12k/OR007@3_2.mat'), scale=StandardScaler())
OR007_3_2_label = one_hot(42, 64)
OR007_3_3_train, OR007_3_3_test = concatenate_data(scipy.io.loadmat('./data/12k/OR007@3_3.mat'), scale=StandardScaler())
OR007_3_3_label = one_hot(43, 64)

OR007_6_0_train, OR007_6_0_test = concatenate_data(scipy.io.loadmat('./data/12k/OR007@6_0.mat'), scale=StandardScaler())
OR007_6_0_label = one_hot(44, 64)
OR007_6_1_train, OR007_6_1_test = concatenate_data(scipy.io.loadmat('./data/12k/OR007@6_1.mat'), scale=StandardScaler())
OR007_6_1_label = one_hot(45, 64)
OR007_6_2_train, OR007_6_2_test = concatenate_data(scipy.io.loadmat('./data/12k/OR007@6_2.mat'), scale=StandardScaler())
OR007_6_2_label = one_hot(46, 64)
OR007_6_3_train, OR007_6_3_test = concatenate_data(scipy.io.loadmat('./data/12k/OR007@6_3.mat'), scale=StandardScaler())
OR007_6_3_label = one_hot(47, 64)

OR0014_6_0_train, OR0014_6_0_test = concatenate_data(scipy.io.loadmat('./data/12k/OR014@6_0.mat'), scale=StandardScaler())
OR0014_6_0_label = one_hot(48, 64)
OR0014_6_1_train, OR0014_6_1_test = concatenate_data(scipy.io.loadmat('./data/12k/OR014@6_1.mat'), scale=StandardScaler())
OR0014_6_1_label = one_hot(49, 64)
OR0014_6_2_train, OR0014_6_2_test = concatenate_data(scipy.io.loadmat('./data/12k/OR014@6_2.mat'), scale=StandardScaler())
OR0014_6_2_label = one_hot(50, 64)
OR0014_6_3_train, OR0014_6_3_test = concatenate_data(scipy.io.loadmat('./data/12k/OR014@6_3.mat'), scale=StandardScaler())
OR0014_6_3_label = one_hot(51, 64)

OR0021_6_0_train, OR0021_6_0_test = concatenate_data(scipy.io.loadmat('./data/12k/OR021@6_0.mat'), scale=StandardScaler())
OR0021_6_0_label = one_hot(52, 64)
OR0021_6_1_train, OR0021_6_1_test = concatenate_data(scipy.io.loadmat('./data/12k/OR021@6_1.mat'), scale=StandardScaler())
OR0021_6_1_label = one_hot(53, 64)
OR0021_6_2_train, OR0021_6_2_test = concatenate_data(scipy.io.loadmat('./data/12k/OR021@6_2.mat'), scale=StandardScaler())
OR0021_6_2_label = one_hot(54, 64)
OR0021_6_3_train, OR0021_6_3_test = concatenate_data(scipy.io.loadmat('./data/12k/OR021@6_3.mat'), scale=StandardScaler())
OR0021_6_3_label = one_hot(55, 64)

OR0021_3_0_train, OR0021_3_0_test = concatenate_data(scipy.io.loadmat('./data/12k/OR021@3_0.mat'), scale=StandardScaler())
OR0021_3_0_label = one_hot(56, 64)
OR0021_3_1_train, OR0021_3_1_test = concatenate_data(scipy.io.loadmat('./data/12k/OR021@3_1.mat'), scale=StandardScaler())
OR0021_3_1_label = one_hot(57, 64)
OR0021_3_2_train, OR0021_3_2_test = concatenate_data(scipy.io.loadmat('./data/12k/OR021@3_2.mat'), scale=StandardScaler())
OR0021_3_2_label = one_hot(58, 64)
OR0021_3_3_train, OR0021_3_3_test = concatenate_data(scipy.io.loadmat('./data/12k/OR021@3_3.mat'), scale=StandardScaler())
OR0021_3_3_label = one_hot(59, 64)

OR0021_12_0_train, OR0021_12_0_test = concatenate_data(scipy.io.loadmat('./data/12k/OR021@12_0.mat'), scale=StandardScaler())
OR0021_12_0_label = one_hot(60, 64)
OR0021_12_1_train, OR0021_12_1_test = concatenate_data(scipy.io.loadmat('./data/12k/OR021@12_1.mat'), scale=StandardScaler())
OR0021_12_1_label = one_hot(61, 64)
OR0021_12_2_train, OR0021_12_2_test = concatenate_data(scipy.io.loadmat('./data/12k/OR021@12_2.mat'), scale=StandardScaler())
OR0021_12_2_label = one_hot(62, 64)
OR0021_12_3_train, OR0021_12_3_test = concatenate_data(scipy.io.loadmat('./data/12k/OR021@12_3.mat'), scale=StandardScaler())
OR0021_12_3_label = one_hot(63, 64)


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
