import numpy as np
import pandas as pd
import scipy.io
import tensorflow as tf
from preprocessing.denoise_signal import Fourier, SVD_denoise, Wavelet, Wavelet_denoise, savitzky_golay
from preprocessing.utils import get_spectrogram, one_hot, concatenate_data
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

scaler = None

data_normal = True
data_12k    = True
data_48k    = False
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


if data_normal:
  Normal_0 = concatenate_data(scipy.io.loadmat('./data/normal/Normal_0.mat'), scale=scaler)
  Normal_0_label = one_hot(0, 64)
  Normal_1 = concatenate_data(scipy.io.loadmat('./data/normal/Normal_1.mat'), scale=scaler)
  Normal_1_label = one_hot(1, 64)
  Normal_2 = concatenate_data(scipy.io.loadmat('./data/normal/Normal_2.mat'), scale=scaler)
  Normal_2_label = one_hot(2, 64)
  Normal_3 = concatenate_data(scipy.io.loadmat('./data/normal/Normal_3.mat'), scale=scaler)
  Normal_3_label = one_hot(3, 64)

if data_12k:
  B007_0 = concatenate_data(scipy.io.loadmat('./data/12k/B007_0.mat'), scale=scaler)
  B007_0_label = one_hot(4, 64)
  B007_1 = concatenate_data(scipy.io.loadmat('./data/12k/B007_1.mat'), scale=scaler)
  B007_1_label = one_hot(5, 64)
  B007_2 = concatenate_data(scipy.io.loadmat('./data/12k/B007_2.mat'), scale=scaler)
  B007_2_label = one_hot(6, 64)
  B007_3 = concatenate_data(scipy.io.loadmat('./data/12k/B007_3.mat'), scale=scaler)
  B007_3_label = one_hot(7, 64)

  B014_0 = concatenate_data(scipy.io.loadmat('./data/12k/B014_0.mat'), scale=scaler)
  B014_0_label = one_hot(8, 64)
  B014_1 = concatenate_data(scipy.io.loadmat('./data/12k/B014_1.mat'), scale=scaler)
  B014_1_label = one_hot(9, 64)
  B014_2 = concatenate_data(scipy.io.loadmat('./data/12k/B014_2.mat'), scale=scaler)
  B014_2_label = one_hot(10, 64)
  B014_3= concatenate_data(scipy.io.loadmat('./data/12k/B014_3.mat'), scale=scaler)
  B014_3_label = one_hot(11, 64)

  B021_0 = concatenate_data(scipy.io.loadmat('./data/12k/B021_0.mat'), scale=scaler)
  B021_0_label = one_hot(12, 64)
  B021_1 = concatenate_data(scipy.io.loadmat('./data/12k/B021_1.mat'), scale=scaler)
  B021_1_label = one_hot(13, 64)
  B021_2 = concatenate_data(scipy.io.loadmat('./data/12k/B021_2.mat'), scale=scaler)
  B021_2_label = one_hot(14, 64)
  B021_3 = concatenate_data(scipy.io.loadmat('./data/12k/B021_3.mat'), scale=scaler)
  B021_3_label = one_hot(15, 64)

  B028_0 = concatenate_data(scipy.io.loadmat('./data/12k/B028_0.mat'), scale=scaler)
  B028_0_label = one_hot(16, 64)
  B028_1 = concatenate_data(scipy.io.loadmat('./data/12k/B028_1.mat'), scale=scaler)
  B028_1_label = one_hot(17, 64)
  B028_2 = concatenate_data(scipy.io.loadmat('./data/12k/B028_2.mat'), scale=scaler)
  B028_2_label = one_hot(18, 64)
  B028_3 = concatenate_data(scipy.io.loadmat('./data/12k/B028_3.mat'), scale=scaler)
  B028_3_label = one_hot(19, 64)

  IR007_0 = concatenate_data(scipy.io.loadmat('./data/12k/IR007_0.mat'), scale=scaler)
  IR007_0_label = one_hot(20, 64)
  IR007_1 = concatenate_data(scipy.io.loadmat('./data/12k/IR007_1.mat'), scale=scaler)
  IR007_1_label = one_hot(21, 64)
  IR007_2 = concatenate_data(scipy.io.loadmat('./data/12k/IR007_2.mat'), scale=scaler)
  IR007_2_label = one_hot(22, 64)
  IR007_3 = concatenate_data(scipy.io.loadmat('./data/12k/IR007_3.mat'), scale=scaler)
  IR007_3_label = one_hot(23, 64)

  IR014_0 = concatenate_data(scipy.io.loadmat('./data/12k/IR014_0.mat'), scale=scaler)
  IR014_0_label = one_hot(24, 64)
  IR014_1 = concatenate_data(scipy.io.loadmat('./data/12k/IR014_1.mat'), scale=scaler)
  IR014_1_label = one_hot(25, 64)
  IR014_2 = concatenate_data(scipy.io.loadmat('./data/12k/IR014_2.mat'), scale=scaler)
  IR014_2_label = one_hot(26, 64)
  IR014_3 = concatenate_data(scipy.io.loadmat('./data/12k/IR014_3.mat'), scale=scaler)
  IR014_3_label = one_hot(27, 64)

  IR021_0 = concatenate_data(scipy.io.loadmat('./data/12k/IR021_0.mat'), scale=scaler)
  IR021_0_label = one_hot(28, 64)
  IR021_1 = concatenate_data(scipy.io.loadmat('./data/12k/IR021_1.mat'), scale=scaler)
  IR021_1_label = one_hot(29, 64)
  IR021_2 = concatenate_data(scipy.io.loadmat('./data/12k/IR021_2.mat'), scale=scaler)
  IR021_2_label = one_hot(30, 64)
  IR021_3 = concatenate_data(scipy.io.loadmat('./data/12k/IR021_3.mat'), scale=scaler)
  IR021_3_label = one_hot(31, 64)

  IR028_0 = concatenate_data(scipy.io.loadmat('./data/12k/IR028_0.mat'), scale=scaler)
  IR028_0_label = one_hot(32, 64)
  IR028_1 = concatenate_data(scipy.io.loadmat('./data/12k/IR028_1.mat'), scale=scaler)
  IR028_1_label = one_hot(33, 64)
  IR028_2 = concatenate_data(scipy.io.loadmat('./data/12k/IR028_2.mat'), scale=scaler)
  IR028_2_label = one_hot(34, 64)
  IR028_3 = concatenate_data(scipy.io.loadmat('./data/12k/IR028_3.mat'), scale=scaler)
  IR028_3_label = one_hot(35, 64)

  OR007_12_0 = concatenate_data(scipy.io.loadmat('./data/12k/OR007@12_0.mat'), scale=scaler)
  OR007_12_0_label = one_hot(36, 64)
  OR007_12_1 = concatenate_data(scipy.io.loadmat('./data/12k/OR007@12_1.mat'), scale=scaler)
  OR007_12_1_label = one_hot(37, 64)
  OR007_12_2 = concatenate_data(scipy.io.loadmat('./data/12k/OR007@12_2.mat'), scale=scaler)
  OR007_12_2_label = one_hot(38, 64)
  OR007_12_3 = concatenate_data(scipy.io.loadmat('./data/12k/OR007@12_3.mat'), scale=scaler)
  OR007_12_3_label = one_hot(39, 64)

  OR007_3_0 = concatenate_data(scipy.io.loadmat('./data/12k/OR007@3_0.mat'), scale=scaler)
  OR007_3_0_label = one_hot(40, 64)
  OR007_3_1 = concatenate_data(scipy.io.loadmat('./data/12k/OR007@3_1.mat'), scale=scaler)
  OR007_3_1_label = one_hot(41, 64)
  OR007_3_2 = concatenate_data(scipy.io.loadmat('./data/12k/OR007@3_2.mat'), scale=scaler)
  OR007_3_2_label = one_hot(42, 64)
  OR007_3_3 = concatenate_data(scipy.io.loadmat('./data/12k/OR007@3_3.mat'), scale=scaler)
  OR007_3_3_label = one_hot(43, 64)

  OR007_6_0 = concatenate_data(scipy.io.loadmat('./data/12k/OR007@6_0.mat'), scale=scaler)
  OR007_6_0_label = one_hot(44, 64)
  OR007_6_1 = concatenate_data(scipy.io.loadmat('./data/12k/OR007@6_1.mat'), scale=scaler)
  OR007_6_1_label = one_hot(45, 64)
  OR007_6_2 = concatenate_data(scipy.io.loadmat('./data/12k/OR007@6_2.mat'), scale=scaler)
  OR007_6_2_label = one_hot(46, 64)
  OR007_6_3 = concatenate_data(scipy.io.loadmat('./data/12k/OR007@6_3.mat'), scale=scaler)
  OR007_6_3_label = one_hot(47, 64)

  OR014_6_0 = concatenate_data(scipy.io.loadmat('./data/12k/OR014@6_0.mat'), scale=scaler)
  OR014_6_0_label = one_hot(48, 64)
  OR014_6_1 = concatenate_data(scipy.io.loadmat('./data/12k/OR014@6_1.mat'), scale=scaler)
  OR014_6_1_label = one_hot(49, 64)
  OR014_6_2 = concatenate_data(scipy.io.loadmat('./data/12k/OR014@6_2.mat'), scale=scaler)
  OR014_6_2_label = one_hot(50, 64)
  OR014_6_3 = concatenate_data(scipy.io.loadmat('./data/12k/OR014@6_3.mat'), scale=scaler)
  OR014_6_3_label = one_hot(51, 64)

  OR021_6_0 = concatenate_data(scipy.io.loadmat('./data/12k/OR021@6_0.mat'), scale=scaler)
  OR021_6_0_label = one_hot(52, 64)
  OR021_6_1 = concatenate_data(scipy.io.loadmat('./data/12k/OR021@6_1.mat'), scale=scaler)
  OR021_6_1_label = one_hot(53, 64)
  OR021_6_2 = concatenate_data(scipy.io.loadmat('./data/12k/OR021@6_2.mat'), scale=scaler)
  OR021_6_2_label = one_hot(54, 64)
  OR021_6_3 = concatenate_data(scipy.io.loadmat('./data/12k/OR021@6_3.mat'), scale=scaler)
  OR021_6_3_label = one_hot(55, 64)

  OR021_3_0 = concatenate_data(scipy.io.loadmat('./data/12k/OR021@3_0.mat'), scale=scaler)
  OR021_3_0_label = one_hot(56, 64)
  OR021_3_1 = concatenate_data(scipy.io.loadmat('./data/12k/OR021@3_1.mat'), scale=scaler)
  OR021_3_1_label = one_hot(57, 64)
  OR021_3_2 = concatenate_data(scipy.io.loadmat('./data/12k/OR021@3_2.mat'), scale=scaler)
  OR021_3_2_label = one_hot(58, 64)
  OR021_3_3 = concatenate_data(scipy.io.loadmat('./data/12k/OR021@3_3.mat'), scale=scaler)
  OR021_3_3_label = one_hot(59, 64)

  OR021_12_0 = concatenate_data(scipy.io.loadmat('./data/12k/OR021@12_0.mat'), scale=scaler)
  OR021_12_0_label = one_hot(60, 64)
  OR021_12_1 = concatenate_data(scipy.io.loadmat('./data/12k/OR021@12_1.mat'), scale=scaler)
  OR021_12_1_label = one_hot(61, 64)
  OR021_12_2 = concatenate_data(scipy.io.loadmat('./data/12k/OR021@12_2.mat'), scale=scaler)
  OR021_12_2_label = one_hot(62, 64)
  OR021_12_3 = concatenate_data(scipy.io.loadmat('./data/12k/OR021@12_3.mat'), scale=scaler)
  OR021_12_3_label = one_hot(63, 64)

if data_48k:
  B007_0 = concatenate_data(scipy.io.loadmat('./data/48k/B007_0.mat'), scale=scaler)
  B007_0_label = one_hot(4, 64)
  B007_1 = concatenate_data(scipy.io.loadmat('./data/48k/B007_1.mat'), scale=scaler)
  B007_1_label = one_hot(5, 64)
  B007_2 = concatenate_data(scipy.io.loadmat('./data/48k/B007_2.mat'), scale=scaler)
  B007_2_label = one_hot(6, 64)
  B007_3 = concatenate_data(scipy.io.loadmat('./data/48k/B007_3.mat'), scale=scaler)
  B007_3_label = one_hot(7, 64)

  B014_0 = concatenate_data(scipy.io.loadmat('./data/48k/B014_0.mat'), scale=scaler)
  B014_0_label = one_hot(8, 64)
  B014_1 = concatenate_data(scipy.io.loadmat('./data/48k/B014_1.mat'), scale=scaler)
  B014_1_label = one_hot(9, 64)
  B014_2 = concatenate_data(scipy.io.loadmat('./data/48k/B014_2.mat'), scale=scaler)
  B014_2_label = one_hot(10, 64)
  B014_3= concatenate_data(scipy.io.loadmat('./data/48k/B014_3.mat'), scale=scaler)
  B014_3_label = one_hot(11, 64)

  B021_0 = concatenate_data(scipy.io.loadmat('./data/48k/B021_0.mat'), scale=scaler)
  B021_0_label = one_hot(12, 64)
  B021_1 = concatenate_data(scipy.io.loadmat('./data/48k/B021_1.mat'), scale=scaler)
  B021_1_label = one_hot(13, 64)
  B021_2 = concatenate_data(scipy.io.loadmat('./data/48k/B021_2.mat'), scale=scaler)
  B021_2_label = one_hot(14, 64)
  B021_3 = concatenate_data(scipy.io.loadmat('./data/48k/B021_3.mat'), scale=scaler)
  B021_3_label = one_hot(15, 64)

  IR007_0 = concatenate_data(scipy.io.loadmat('./data/48k/IR007_0.mat'), scale=scaler)
  IR007_0_label = one_hot(20, 64)
  IR007_1 = concatenate_data(scipy.io.loadmat('./data/48k/IR007_1.mat'), scale=scaler)
  IR007_1_label = one_hot(21, 64)
  IR007_2 = concatenate_data(scipy.io.loadmat('./data/48k/IR007_2.mat'), scale=scaler)
  IR007_2_label = one_hot(22, 64)
  IR007_3 = concatenate_data(scipy.io.loadmat('./data/48k/IR007_3.mat'), scale=scaler)
  IR007_3_label = one_hot(23, 64)

  IR014_0 = concatenate_data(scipy.io.loadmat('./data/48k/IR014_0.mat'), scale=scaler)
  IR014_0_label = one_hot(24, 64)
  IR014_1 = concatenate_data(scipy.io.loadmat('./data/48k/IR014_1.mat'), scale=scaler)
  IR014_1_label = one_hot(25, 64)
  IR014_2 = concatenate_data(scipy.io.loadmat('./data/48k/IR014_2.mat'), scale=scaler)
  IR014_2_label = one_hot(26, 64)
  IR014_3 = concatenate_data(scipy.io.loadmat('./data/48k/IR014_3.mat'), scale=scaler)
  IR014_3_label = one_hot(27, 64)

  IR021_0 = concatenate_data(scipy.io.loadmat('./data/48k/IR021_0.mat'), scale=scaler)
  IR021_0_label = one_hot(28, 64)
  IR021_1 = concatenate_data(scipy.io.loadmat('./data/48k/IR021_1.mat'), scale=scaler)
  IR021_1_label = one_hot(29, 64)
  IR021_2 = concatenate_data(scipy.io.loadmat('./data/48k/IR021_2.mat'), scale=scaler)
  IR021_2_label = one_hot(30, 64)
  IR021_3 = concatenate_data(scipy.io.loadmat('./data/48k/IR021_3.mat'), scale=scaler)
  IR021_3_label = one_hot(31, 64)

  OR007_12_0 = concatenate_data(scipy.io.loadmat('./data/48k/OR007@12_0.mat'), scale=scaler)
  OR007_12_0_label = one_hot(36, 64)
  OR007_12_1 = concatenate_data(scipy.io.loadmat('./data/48k/OR007@12_1.mat'), scale=scaler)
  OR007_12_1_label = one_hot(37, 64)
  OR007_12_2 = concatenate_data(scipy.io.loadmat('./data/48k/OR007@12_2.mat'), scale=scaler)
  OR007_12_2_label = one_hot(38, 64)
  OR007_12_3 = concatenate_data(scipy.io.loadmat('./data/48k/OR007@12_3.mat'), scale=scaler)
  OR007_12_3_label = one_hot(39, 64)

  OR007_3_0 = concatenate_data(scipy.io.loadmat('./data/48k/OR007@3_0.mat'), scale=scaler)
  OR007_3_0_label = one_hot(40, 64)
  OR007_3_1 = concatenate_data(scipy.io.loadmat('./data/48k/OR007@3_1.mat'), scale=scaler)
  OR007_3_1_label = one_hot(41, 64)
  OR007_3_2 = concatenate_data(scipy.io.loadmat('./data/48k/OR007@3_2.mat'), scale=scaler)
  OR007_3_2_label = one_hot(42, 64)
  OR007_3_3 = concatenate_data(scipy.io.loadmat('./data/48k/OR007@3_3.mat'), scale=scaler)
  OR007_3_3_label = one_hot(43, 64)

  OR007_6_0 = concatenate_data(scipy.io.loadmat('./data/48k/OR007@6_0.mat'), scale=scaler)
  OR007_6_0_label = one_hot(44, 64)
  OR007_6_1 = concatenate_data(scipy.io.loadmat('./data/48k/OR007@6_1.mat'), scale=scaler)
  OR007_6_1_label = one_hot(45, 64)
  OR007_6_2 = concatenate_data(scipy.io.loadmat('./data/48k/OR007@6_2.mat'), scale=scaler)
  OR007_6_2_label = one_hot(46, 64)
  OR007_6_3 = concatenate_data(scipy.io.loadmat('./data/48k/OR007@6_3.mat'), scale=scaler)
  OR007_6_3_label = one_hot(47, 64)

  OR0014_6_0 = concatenate_data(scipy.io.loadmat('./data/48k/OR014@6_0.mat'), scale=scaler)
  OR0014_6_0_label = one_hot(48, 64)
  OR0014_6_1 = concatenate_data(scipy.io.loadmat('./data/48k/OR014@6_1.mat'), scale=scaler)
  OR0014_6_1_label = one_hot(49, 64)
  OR0014_6_2 = concatenate_data(scipy.io.loadmat('./data/48k/OR014@6_2.mat'), scale=scaler)
  OR0014_6_2_label = one_hot(50, 64)
  OR0014_6_3 = concatenate_data(scipy.io.loadmat('./data/48k/OR014@6_3.mat'), scale=scaler)
  OR0014_6_3_label = one_hot(51, 64)

  OR0021_6_0 = concatenate_data(scipy.io.loadmat('./data/48k/OR021@6_0.mat'), scale=scaler)
  OR0021_6_0_label = one_hot(52, 64)
  OR0021_6_1 = concatenate_data(scipy.io.loadmat('./data/48k/OR021@6_1.mat'), scale=scaler)
  OR0021_6_1_label = one_hot(53, 64)
  OR0021_6_2 = concatenate_data(scipy.io.loadmat('./data/48k/OR021@6_2.mat'), scale=scaler)
  OR0021_6_2_label = one_hot(54, 64)
  OR0021_6_3 = concatenate_data(scipy.io.loadmat('./data/48k/OR021@6_3.mat'), scale=scaler)
  OR0021_6_3_label = one_hot(55, 64)

  OR0021_3_0 = concatenate_data(scipy.io.loadmat('./data/48k/OR021@3_0.mat'), scale=scaler)
  OR0021_3_0_label = one_hot(56, 64)
  OR0021_3_1 = concatenate_data(scipy.io.loadmat('./data/48k/OR021@3_1.mat'), scale=scaler)
  OR0021_3_1_label = one_hot(57, 64)
  OR0021_3_2 = concatenate_data(scipy.io.loadmat('./data/48k/OR021@3_2.mat'), scale=scaler)
  OR0021_3_2_label = one_hot(58, 64)
  OR0021_3_3 = concatenate_data(scipy.io.loadmat('./data/48k/OR021@3_3.mat'), scale=scaler)
  OR0021_3_3_label = one_hot(59, 64)

  OR0021_12_0 = concatenate_data(scipy.io.loadmat('./data/48k/OR021@12_0.mat'), scale=scaler)
  OR0021_12_0_label = one_hot(60, 64)
  OR0021_12_1 = concatenate_data(scipy.io.loadmat('./data/48k/OR021@12_1.mat'), scale=scaler)
  OR0021_12_1_label = one_hot(61, 64)
  OR0021_12_2 = concatenate_data(scipy.io.loadmat('./data/48k/OR021@12_2.mat'), scale=scaler)
  OR0021_12_2_label = one_hot(62, 64)
  OR0021_12_3 = concatenate_data(scipy.io.loadmat('./data/48k/OR021@12_3.mat'), scale=scaler)
  OR0021_12_3_label = one_hot(63, 64)

    # if opt.denoise == 'DFK':
    #     Normal_0_X097_DE_time   = Fourier(f=Normal_0_X097_DE_time, num=num, get_result=True, thres=25).reshape(num, 1)
    #     Normal_0_X097_FE_time   = Fourier(f=Normal_0_X097_FE_time, num=num, get_result=True, thres=85).reshape(num, 1)

    #     B007_0_X122_DE_time     = Fourier(f=B007_0_X122_DE_time, num=num, get_result=True, thres=25).reshape(num, 1)
    #     B007_0_X122_FE_time     = Fourier(f=B007_0_X122_FE_time, num=num, get_result=True, thres=85).reshape(num, 1)

    #     IR007_0_X122_DE_time    = Fourier(f=IR007_0_X122_DE_time, num=num, get_result=True, thres=25).reshape(num, 1)
    #     IR007_0_X122_FE_time    = Fourier(f=IR007_0_X122_FE_time, num=num, get_result=True, thres=85).reshape(num, 1)

    #     OR007_3_0_X122_DE_time  = Fourier(f=OR007_3_0_X122_DE_time, num=num, get_result=True, thres=25).reshape(num, 1)
    #     OR007_3_0_X122_FE_time  = Fourier(f=OR007_3_0_X122_FE_time, num=num, get_result=True, thres=85).reshape(num, 1)

    #     OR007_6_0_X122_DE_time  = Fourier(f=OR007_6_0_X122_DE_time, num=num, get_result=True, thres=25).reshape(num, 1)
    #     OR007_6_0_X122_FE_time  = Fourier(f=OR007_6_0_X122_FE_time, num=num, get_result=True, thres=85).reshape(num, 1)

    #     OR007_12_0_X122_DE_time = Fourier(f=OR007_12_0_X122_DE_time, num=num, get_result=True, thres=25).reshape(num, 1)
    #     OR007_12_0_X122_FE_time = Fourier(f=OR007_12_0_X122_FE_time, num=num, get_result=True, thres=85).reshape(num, 1)

    #     DE_time = np.concatenate((Normal_0_X097_DE_time.reshape(int(num/n), n), B007_0_X122_DE_time.reshape(int(num/n), n), IR007_0_X122_DE_time.reshape(int(num/n), n), OR007_3_0_X122_DE_time.reshape(int(num/n), n), OR007_6_0_X122_DE_time.reshape(int(num/n), n), OR007_12_0_X122_DE_time.reshape(int(num/n), n)))
    #     FE_time = np.concatenate((Normal_0_X097_FE_time.reshape(int(num/n), n), B007_0_X122_FE_time.reshape(int(num/n), n), IR007_0_X122_FE_time.reshape(int(num/n), n), OR007_3_0_X122_FE_time.reshape(int(num/n), n), OR007_6_0_X122_FE_time.reshape(int(num/n), n), OR007_12_0_X122_FE_time.reshape(int(num/n), n)))

    #     merge_data = np.concatenate((DE_time, FE_time), axis=1)

    # if opt.denoise == 'savitzky_golay':
    #     window_size=15
    #     Normal_0_X097_DE_time   = savitzky_golay(y=Normal_0_X097_DE_time, window_size=window_size, order=4, range_y=num).reshape(num, 1)
    #     Normal_0_X097_FE_time   = savitzky_golay(y=Normal_0_X097_FE_time, window_size=window_size, order=4, range_y=num).reshape(num, 1)

    #     B007_0_X122_DE_time     = savitzky_golay(y=B007_0_X122_DE_time, window_size=window_size, order=4, range_y=num).reshape(num, 1)
    #     B007_0_X122_FE_time     = savitzky_golay(y=B007_0_X122_FE_time, window_size=window_size, order=4, range_y=num).reshape(num, 1)

    #     IR007_0_X122_DE_time    = savitzky_golay(y=IR007_0_X122_DE_time, window_size=window_size, order=4, range_y=num).reshape(num, 1)
    #     IR007_0_X122_FE_time    = savitzky_golay(y=IR007_0_X122_FE_time, window_size=window_size, order=4, range_y=num).reshape(num, 1)

    #     OR007_3_0_X122_DE_time  = savitzky_golay(y=OR007_3_0_X122_DE_time, window_size=window_size, order=4, range_y=num).reshape(num, 1)
    #     OR007_3_0_X122_FE_time  = savitzky_golay(y=OR007_3_0_X122_FE_time, window_size=window_size, order=4, range_y=num).reshape(num, 1)

    #     OR007_6_0_X122_DE_time  = savitzky_golay(y=OR007_6_0_X122_DE_time, window_size=window_size, order=4, range_y=num).reshape(num, 1)
    #     OR007_6_0_X122_FE_time  = savitzky_golay(y=OR007_6_0_X122_FE_time, window_size=window_size, order=4, range_y=num).reshape(num, 1)

    #     OR007_12_0_X122_DE_time = savitzky_golay(y=OR007_12_0_X122_DE_time, window_size=window_size, order=4, range_y=num).reshape(num, 1)
    #     OR007_12_0_X122_FE_time = savitzky_golay(y=OR007_12_0_X122_FE_time, window_size=window_size, order=4, range_y=num).reshape(num, 1)

    #     DE_time = np.concatenate((Normal_0_X097_DE_time.reshape(int(num/n), n), B007_0_X122_DE_time.reshape(int(num/n), n), IR007_0_X122_DE_time.reshape(int(num/n), n), OR007_3_0_X122_DE_time.reshape(int(num/n), n), OR007_6_0_X122_DE_time.reshape(int(num/n), n), OR007_12_0_X122_DE_time.reshape(int(num/n), n)))
    #     FE_time = np.concatenate((Normal_0_X097_FE_time.reshape(int(num/n), n), B007_0_X122_FE_time.reshape(int(num/n), n), IR007_0_X122_FE_time.reshape(int(num/n), n), OR007_3_0_X122_FE_time.reshape(int(num/n), n), OR007_6_0_X122_FE_time.reshape(int(num/n), n), OR007_12_0_X122_FE_time.reshape(int(num/n), n)))

    #     merge_data = np.concatenate((DE_time, FE_time), axis=1)


    # if opt.denoise == 'Wavelet_denoise':
    #     Normal_0_X097_DE_time = Wavelet_denoise(Normal_0_X097_DE_time)
    #     Normal_0_X097_FE_time = Wavelet_denoise(Normal_0_X097_FE_time)

    #     B007_0_X122_DE_time = Wavelet_denoise(B007_0_X122_DE_time)
    #     B007_0_X122_FE_time = Wavelet_denoise(B007_0_X122_FE_time)

    #     IR007_0_X122_DE_time = Wavelet_denoise(IR007_0_X122_DE_time)
    #     IR007_0_X122_FE_time = Wavelet_denoise(IR007_0_X122_FE_time)

    #     OR007_3_0_X122_DE_time = Wavelet_denoise(OR007_3_0_X122_DE_time)
    #     OR007_3_0_X122_FE_time = Wavelet_denoise(OR007_3_0_X122_FE_time)

    #     OR007_6_0_X122_DE_time = Wavelet_denoise(OR007_6_0_X122_DE_time)
    #     OR007_6_0_X122_FE_time = Wavelet_denoise(OR007_6_0_X122_FE_time)

    #     OR007_12_0_X122_DE_time = Wavelet_denoise(OR007_12_0_X122_DE_time)
    #     OR007_12_0_X122_FE_time = Wavelet_denoise(OR007_12_0_X122_FE_time)

    #     DE_time = np.concatenate((Normal_0_X097_DE_time.reshape(int(num/n), n), B007_0_X122_DE_time.reshape(int(num/n), n), IR007_0_X122_DE_time.reshape(int(num/n), n), OR007_3_0_X122_DE_time.reshape(int(num/n), n), OR007_6_0_X122_DE_time.reshape(int(num/n), n), OR007_12_0_X122_DE_time.reshape(int(num/n), n)))
    #     FE_time = np.concatenate((Normal_0_X097_FE_time.reshape(int(num/n), n), B007_0_X122_FE_time.reshape(int(num/n), n), IR007_0_X122_FE_time.reshape(int(num/n), n), OR007_3_0_X122_FE_time.reshape(int(num/n), n), OR007_6_0_X122_FE_time.reshape(int(num/n), n), OR007_12_0_X122_FE_time.reshape(int(num/n), n)))

    #     merge_data = np.concatenate((DE_time, FE_time), axis=1)

    # if opt.denoise == 'SVD':
    #     Normal_0_X097_DE_time = SVD_denoise(Normal_0_X097_DE_time)
    #     Normal_0_X097_FE_time = SVD_denoise(Normal_0_X097_FE_time)

    #     B007_0_X122_DE_time = SVD_denoise(B007_0_X122_DE_time)
    #     B007_0_X122_FE_time = SVD_denoise(B007_0_X122_FE_time)

    #     IR007_0_X122_DE_time = SVD_denoise(IR007_0_X122_DE_time)
    #     IR007_0_X122_FE_time = SVD_denoise(IR007_0_X122_FE_time)

    #     OR007_3_0_X122_DE_time = SVD_denoise(OR007_3_0_X122_DE_time)
    #     OR007_3_0_X122_FE_time = SVD_denoise(OR007_3_0_X122_FE_time)

    #     OR007_6_0_X122_DE_time = SVD_denoise(OR007_6_0_X122_DE_time)
    #     OR007_6_0_X122_FE_time = SVD_denoise(OR007_6_0_X122_FE_time)

    #     OR007_12_0_X122_DE_time = SVD_denoise(OR007_12_0_X122_DE_time)
    #     OR007_12_0_X122_FE_time = SVD_denoise(OR007_12_0_X122_FE_time)

    #     DE_time = np.concatenate((Normal_0_X097_DE_time.reshape(int(num/n), n), B007_0_X122_DE_time.reshape(int(num/n), n), IR007_0_X122_DE_time.reshape(int(num/n), n), OR007_3_0_X122_DE_time.reshape(int(num/n), n), OR007_6_0_X122_DE_time.reshape(int(num/n), n), OR007_12_0_X122_DE_time.reshape(int(num/n), n)))
    #     FE_time = np.concatenate((Normal_0_X097_FE_time.reshape(int(num/n), n), B007_0_X122_FE_time.reshape(int(num/n), n), IR007_0_X122_FE_time.reshape(int(num/n), n), OR007_3_0_X122_FE_time.reshape(int(num/n), n), OR007_6_0_X122_FE_time.reshape(int(num/n), n), OR007_12_0_X122_FE_time.reshape(int(num/n), n)))

    #     merge_data = np.concatenate((DE_time, FE_time), axis=1)

    # if opt.use_DNN_B:
    #     DE_time = np.concatenate((Normal_0_X097_DE_time.reshape(int(num/n), n), B007_0_X122_DE_time.reshape(int(num/n), n), IR007_0_X122_DE_time.reshape(int(num/n), n), OR007_3_0_X122_DE_time.reshape(int(num/n), n), OR007_6_0_X122_DE_time.reshape(int(num/n), n), OR007_12_0_X122_DE_time.reshape(int(num/n), n)))
    #     FE_time = np.concatenate((Normal_0_X097_FE_time.reshape(int(num/n), n), B007_0_X122_FE_time.reshape(int(num/n), n), IR007_0_X122_FE_time.reshape(int(num/n), n), OR007_3_0_X122_FE_time.reshape(int(num/n), n), OR007_6_0_X122_FE_time.reshape(int(num/n), n), OR007_12_0_X122_FE_time.reshape(int(num/n), n)))
    #     merge_data = np.concatenate((DE_time, FE_time), axis=1)
    # elif opt.use_DNN_A or opt.use_CNN_A or opt.use_CNN_C:
    #     Normal_0_group     = np.concatenate((Normal_0_X097_DE_time.reshape(int(num/n), n), Normal_0_X097_FE_time.reshape(int(num/n), n)), axis=1)
    #     B007_0_group       = np.concatenate((B007_0_X122_DE_time.reshape(int(num/n), n), B007_0_X122_FE_time.reshape(int(num/n), n)), axis=1)
    #     IR007_0_group      = np.concatenate((IR007_0_X122_DE_time.reshape(int(num/n), n), IR007_0_X122_FE_time.reshape(int(num/n), n)), axis=1)
    #     OR007_3_0_group    = np.concatenate((OR007_3_0_X122_DE_time.reshape(int(num/n), n), OR007_3_0_X122_FE_time.reshape(int(num/n), n)), axis=1)
    #     OR007_6_0_group    = np.concatenate((OR007_6_0_X122_DE_time.reshape(int(num/n), n), OR007_6_0_X122_FE_time.reshape(int(num/n), n)), axis=1)
    #     OR007_12_0_group   = np.concatenate((OR007_12_0_X122_DE_time.reshape(int(num/n), n), OR007_12_0_X122_FE_time.reshape(int(num/n), n)), axis=1)
    #     merge_data         = np.concatenate((Normal_0_group, B007_0_group, IR007_0_group, OR007_3_0_group, OR007_6_0_group, OR007_12_0_group))
    # elif opt.use_CNN_B:
    #     Normal_0_X097_DE_time   = Normal_0_X097_DE_time.reshape((num//200, 200))
    #     Normal_0_X097_FE_time   = Normal_0_X097_FE_time.reshape((num//200, 200))
    #     Normal_0_X097 = np.concatenate((Normal_0_X097_DE_time, Normal_0_X097_FE_time), axis=1)

    #     B007_0_X122_DE_time     = B007_0_X122_DE_time.reshape((num//200, 200))
    #     B007_0_X122_FE_time     = B007_0_X122_FE_time.reshape((num//200, 200))
    #     B007_0_X122 = np.concatenate((B007_0_X122_DE_time, B007_0_X122_FE_time), axis=1)

    #     IR007_0_X122_DE_time    = IR007_0_X122_DE_time.reshape((num//200, 200))
    #     IR007_0_X122_FE_time    = IR007_0_X122_FE_time.reshape((num//200, 200))
    #     IR007_0_X122 = np.concatenate((IR007_0_X122_DE_time, IR007_0_X122_FE_time), axis=1)

    #     OR007_3_0_X122_DE_time  = OR007_3_0_X122_DE_time.reshape((num//200, 200))
    #     OR007_3_0_X122_FE_time  = OR007_3_0_X122_FE_time.reshape((num//200, 200))
    #     OR007_3_0_X122 = np.concatenate((OR007_3_0_X122_DE_time, OR007_3_0_X122_FE_time), axis=1)

    #     OR007_6_0_X122_DE_time  = OR007_6_0_X122_DE_time.reshape((num//200, 200))
    #     OR007_6_0_X122_FE_time  = OR007_6_0_X122_FE_time.reshape((num//200, 200))
    #     OR007_6_0_X122 = np.concatenate((OR007_6_0_X122_DE_time, OR007_6_0_X122_FE_time), axis=1)

    #     OR007_12_0_X122_DE_time = OR007_12_0_X122_DE_time.reshape((num//200, 200))
    #     OR007_12_0_X122_FE_time = OR007_12_0_X122_FE_time.reshape((num//200, 200))
    #     OR007_12_0_X122 = np.concatenate((OR007_12_0_X122_DE_time, OR007_12_0_X122_FE_time), axis=1)

    #     Normal_0_X097 = np.array([get_spectrogram(i) for i in Normal_0_X097])
    #     B007_0_X122 = np.array([get_spectrogram(i) for i in B007_0_X122])
    #     IR007_0_X122 = np.array([get_spectrogram(i) for i in IR007_0_X122])
    #     OR007_3_0_X122 = np.array([get_spectrogram(i) for i in OR007_3_0_X122])
    #     OR007_6_0_X122 = np.array([get_spectrogram(i) for i in OR007_6_0_X122])
    #     OR007_12_0_X122 = np.array([get_spectrogram(i) for i in OR007_12_0_X122])

    #     merge_data = np.concatenate((Normal_0_X097, B007_0_X122, IR007_0_X122, OR007_3_0_X122, OR007_6_0_X122, OR007_12_0_X122))
    # return merge_data, label
