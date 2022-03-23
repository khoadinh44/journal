import numpy as np
import pandas as pd
import scipy.io
import tensorflow as tf
from preprocessing.denoise_signal import Fourier, SVD_denoise, Wavelet, Wavelet_denoise, savitzky_golay
from preprocessing.utils import get_spectrogram, one_hot, concatenate_data, divide_sample, load_PU_data, scale_data
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
# from train import parse_opt
from faceNet import parse_opt

opt = parse_opt()

scaler = None
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


if opt.data_normal:
  Normal_0_train, Normal_0_test = concatenate_data(scipy.io.loadmat('./data/normal/Normal_0.mat'), scale=scaler, opt=opt)
  Normal_0_label = one_hot(0, 64)
  Normal_1_train, Normal_1_test = concatenate_data(scipy.io.loadmat('./data/normal/Normal_1.mat'), scale=scaler, opt=opt)
  Normal_1_label = one_hot(1, 64)
  Normal_2_train, Normal_2_test = concatenate_data(scipy.io.loadmat('./data/normal/Normal_2.mat'), scale=scaler, opt=opt)
  Normal_2_label = one_hot(2, 64)
  Normal_3_train, Normal_3_test = concatenate_data(scipy.io.loadmat('./data/normal/Normal_3.mat'), scale=scaler, opt=opt)
  Normal_3_label = one_hot(3, 64)

if opt.data_12k:
  B007_0_train, B007_0_test = concatenate_data(scipy.io.loadmat('./data/12k/B007_0.mat'), scale=scaler, opt=opt)
  B007_0_label = one_hot(4, 64)
  B007_1_train, B007_1_test = concatenate_data(scipy.io.loadmat('./data/12k/B007_1.mat'), scale=scaler, opt=opt)
  B007_1_label = one_hot(5, 64)
  B007_2_train, B007_2_test = concatenate_data(scipy.io.loadmat('./data/12k/B007_2.mat'), scale=scaler, opt=opt)
  B007_2_label = one_hot(6, 64)
  B007_3_train, B007_3_test = concatenate_data(scipy.io.loadmat('./data/12k/B007_3.mat'), scale=scaler, opt=opt)
  B007_3_label = one_hot(7, 64)

  B014_0_train, B014_0_test = concatenate_data(scipy.io.loadmat('./data/12k/B014_0.mat'), scale=scaler, opt=opt)
  B014_0_label = one_hot(8, 64)
  B014_1_train, B014_1_test = concatenate_data(scipy.io.loadmat('./data/12k/B014_1.mat'), scale=scaler, opt=opt)
  B014_1_label = one_hot(9, 64)
  B014_2_train, B014_2_test = concatenate_data(scipy.io.loadmat('./data/12k/B014_2.mat'), scale=scaler, opt=opt)
  B014_2_label = one_hot(10, 64)
  B014_3_train, B014_3_test= concatenate_data(scipy.io.loadmat('./data/12k/B014_3.mat'), scale=scaler, opt=opt)
  B014_3_label = one_hot(11, 64)

  B021_0_train, B021_0_test = concatenate_data(scipy.io.loadmat('./data/12k/B021_0.mat'), scale=scaler, opt=opt)
  B021_0_label = one_hot(12, 64)
  B021_1_train, B021_1_test = concatenate_data(scipy.io.loadmat('./data/12k/B021_1.mat'), scale=scaler, opt=opt)
  B021_1_label = one_hot(13, 64)
  B021_2_train, B021_2_test = concatenate_data(scipy.io.loadmat('./data/12k/B021_2.mat'), scale=scaler, opt=opt)
  B021_2_label = one_hot(14, 64)
  B021_3_train, B021_3_test = concatenate_data(scipy.io.loadmat('./data/12k/B021_3.mat'), scale=scaler, opt=opt)
  B021_3_label = one_hot(15, 64)

  B028_0_train, B028_0_test = concatenate_data(scipy.io.loadmat('./data/12k/B028_0.mat'), scale=scaler, opt=opt)
  B028_0_label = one_hot(16, 64)
  B028_1_train, B028_1_test = concatenate_data(scipy.io.loadmat('./data/12k/B028_1.mat'), scale=scaler, opt=opt)
  B028_1_label = one_hot(17, 64)
  B028_2_train, B028_2_test = concatenate_data(scipy.io.loadmat('./data/12k/B028_2.mat'), scale=scaler, opt=opt)
  B028_2_label = one_hot(18, 64)
  B028_3_train, B028_3_test = concatenate_data(scipy.io.loadmat('./data/12k/B028_3.mat'), scale=scaler, opt=opt)
  B028_3_label = one_hot(19, 64)

  IR007_0_train, IR007_0_test = concatenate_data(scipy.io.loadmat('./data/12k/IR007_0.mat'), scale=scaler, opt=opt)
  IR007_0_label = one_hot(20, 64)
  IR007_1_train, IR007_1_test = concatenate_data(scipy.io.loadmat('./data/12k/IR007_1.mat'), scale=scaler, opt=opt)
  IR007_1_label = one_hot(21, 64)
  IR007_2_train, IR007_2_test = concatenate_data(scipy.io.loadmat('./data/12k/IR007_2.mat'), scale=scaler, opt=opt)
  IR007_2_label = one_hot(22, 64)
  IR007_3_train, IR007_3_test = concatenate_data(scipy.io.loadmat('./data/12k/IR007_3.mat'), scale=scaler, opt=opt)
  IR007_3_label = one_hot(23, 64)

  IR014_0_train, IR014_0_test = concatenate_data(scipy.io.loadmat('./data/12k/IR014_0.mat'), scale=scaler, opt=opt)
  IR014_0_label = one_hot(24, 64)
  IR014_1_train, IR014_1_test = concatenate_data(scipy.io.loadmat('./data/12k/IR014_1.mat'), scale=scaler, opt=opt)
  IR014_1_label = one_hot(25, 64)
  IR014_2_train, IR014_2_test = concatenate_data(scipy.io.loadmat('./data/12k/IR014_2.mat'), scale=scaler, opt=opt)
  IR014_2_label = one_hot(26, 64)
  IR014_3_train, IR014_3_test = concatenate_data(scipy.io.loadmat('./data/12k/IR014_3.mat'), scale=scaler, opt=opt)
  IR014_3_label = one_hot(27, 64)

  IR021_0_train, IR021_0_test = concatenate_data(scipy.io.loadmat('./data/12k/IR021_0.mat'), scale=scaler, opt=opt)
  IR021_0_label = one_hot(28, 64)
  IR021_1_train, IR021_1_test = concatenate_data(scipy.io.loadmat('./data/12k/IR021_1.mat'), scale=scaler, opt=opt)
  IR021_1_label = one_hot(29, 64)
  IR021_2_train, IR021_2_test = concatenate_data(scipy.io.loadmat('./data/12k/IR021_2.mat'), scale=scaler, opt=opt)
  IR021_2_label = one_hot(30, 64)
  IR021_3_train, IR021_3_test = concatenate_data(scipy.io.loadmat('./data/12k/IR021_3.mat'), scale=scaler, opt=opt)
  IR021_3_label = one_hot(31, 64)

  IR028_0_train, IR028_0_test = concatenate_data(scipy.io.loadmat('./data/12k/IR028_0.mat'), scale=scaler, opt=opt)
  IR028_0_label = one_hot(32, 64)
  IR028_1_train, IR028_1_test = concatenate_data(scipy.io.loadmat('./data/12k/IR028_1.mat'), scale=scaler, opt=opt)
  IR028_1_label = one_hot(33, 64)
  IR028_2_train, IR028_2_test = concatenate_data(scipy.io.loadmat('./data/12k/IR028_2.mat'), scale=scaler, opt=opt)
  IR028_2_label = one_hot(34, 64)
  IR028_3_train, IR028_3_test = concatenate_data(scipy.io.loadmat('./data/12k/IR028_3.mat'), scale=scaler, opt=opt)
  IR028_3_label = one_hot(35, 64)

  OR007_12_0_train, OR007_12_0_test = concatenate_data(scipy.io.loadmat('./data/12k/OR007@12_0.mat'), scale=scaler, opt=opt)
  OR007_12_0_label = one_hot(36, 64)
  OR007_12_1_train, OR007_12_1_test = concatenate_data(scipy.io.loadmat('./data/12k/OR007@12_1.mat'), scale=scaler, opt=opt)
  OR007_12_1_label = one_hot(37, 64)
  OR007_12_2_train, OR007_12_2_test = concatenate_data(scipy.io.loadmat('./data/12k/OR007@12_2.mat'), scale=scaler, opt=opt)
  OR007_12_2_label = one_hot(38, 64)
  OR007_12_3_train, OR007_12_3_test = concatenate_data(scipy.io.loadmat('./data/12k/OR007@12_3.mat'), scale=scaler, opt=opt)
  OR007_12_3_label = one_hot(39, 64)

  OR007_3_0_train, OR007_3_0_test = concatenate_data(scipy.io.loadmat('./data/12k/OR007@3_0.mat'), scale=scaler, opt=opt)
  OR007_3_0_label = one_hot(40, 64)
  OR007_3_1_train, OR007_3_1_test = concatenate_data(scipy.io.loadmat('./data/12k/OR007@3_1.mat'), scale=scaler, opt=opt)
  OR007_3_1_label = one_hot(41, 64)
  OR007_3_2_train, OR007_3_2_test = concatenate_data(scipy.io.loadmat('./data/12k/OR007@3_2.mat'), scale=scaler, opt=opt)
  OR007_3_2_label = one_hot(42, 64)
  OR007_3_3_train, OR007_3_3_test = concatenate_data(scipy.io.loadmat('./data/12k/OR007@3_3.mat'), scale=scaler, opt=opt)
  OR007_3_3_label = one_hot(43, 64)

  OR007_6_0_train, OR007_6_0_test = concatenate_data(scipy.io.loadmat('./data/12k/OR007@6_0.mat'), scale=scaler, opt=opt)
  OR007_6_0_label = one_hot(44, 64)
  OR007_6_1_train, OR007_6_1_test = concatenate_data(scipy.io.loadmat('./data/12k/OR007@6_1.mat'), scale=scaler, opt=opt)
  OR007_6_1_label = one_hot(45, 64)
  OR007_6_2_train, OR007_6_2_test = concatenate_data(scipy.io.loadmat('./data/12k/OR007@6_2.mat'), scale=scaler, opt=opt)
  OR007_6_2_label = one_hot(46, 64)
  OR007_6_3_train, OR007_6_3_test = concatenate_data(scipy.io.loadmat('./data/12k/OR007@6_3.mat'), scale=scaler, opt=opt)
  OR007_6_3_label = one_hot(47, 64)

  OR014_6_0_train, OR014_6_0_test = concatenate_data(scipy.io.loadmat('./data/12k/OR014@6_0.mat'), scale=scaler, opt=opt)
  OR014_6_0_label = one_hot(48, 64)
  OR014_6_1_train, OR014_6_1_test = concatenate_data(scipy.io.loadmat('./data/12k/OR014@6_1.mat'), scale=scaler, opt=opt)
  OR014_6_1_label = one_hot(49, 64)
  OR014_6_2_train, OR014_6_2_test = concatenate_data(scipy.io.loadmat('./data/12k/OR014@6_2.mat'), scale=scaler, opt=opt)
  OR014_6_2_label = one_hot(50, 64)
  OR014_6_3_train, OR014_6_3_test = concatenate_data(scipy.io.loadmat('./data/12k/OR014@6_3.mat'), scale=scaler, opt=opt)
  OR014_6_3_label = one_hot(51, 64)

  OR021_6_0_train, OR021_6_0_test = concatenate_data(scipy.io.loadmat('./data/12k/OR021@6_0.mat'), scale=scaler, opt=opt)
  OR021_6_0_label = one_hot(52, 64)
  OR021_6_1_train, OR021_6_1_test = concatenate_data(scipy.io.loadmat('./data/12k/OR021@6_1.mat'), scale=scaler, opt=opt)
  OR021_6_1_label = one_hot(53, 64)
  OR021_6_2_train, OR021_6_2_test = concatenate_data(scipy.io.loadmat('./data/12k/OR021@6_2.mat'), scale=scaler, opt=opt)
  OR021_6_2_label = one_hot(54, 64)
  OR021_6_3_train, OR021_6_3_test = concatenate_data(scipy.io.loadmat('./data/12k/OR021@6_3.mat'), scale=scaler, opt=opt)
  OR021_6_3_label = one_hot(55, 64)

  OR021_3_0_train, OR021_3_0_test = concatenate_data(scipy.io.loadmat('./data/12k/OR021@3_0.mat'), scale=scaler, opt=opt)
  OR021_3_0_label = one_hot(56, 64)
  OR021_3_1_train, OR021_3_1_test = concatenate_data(scipy.io.loadmat('./data/12k/OR021@3_1.mat'), scale=scaler, opt=opt)
  OR021_3_1_label = one_hot(57, 64)
  OR021_3_2_train, OR021_3_2_test = concatenate_data(scipy.io.loadmat('./data/12k/OR021@3_2.mat'), scale=scaler, opt=opt)
  OR021_3_2_label = one_hot(58, 64)
  OR021_3_3_train, OR021_3_3_test = concatenate_data(scipy.io.loadmat('./data/12k/OR021@3_3.mat'), scale=scaler, opt=opt)
  OR021_3_3_label = one_hot(59, 64)

  OR021_12_0_train, OR021_12_0_test = concatenate_data(scipy.io.loadmat('./data/12k/OR021@12_0.mat'), scale=scaler, opt=opt)
  OR021_12_0_label = one_hot(60, 64)
  OR021_12_1_train, OR021_12_1_test = concatenate_data(scipy.io.loadmat('./data/12k/OR021@12_1.mat'), scale=scaler, opt=opt)
  OR021_12_1_label = one_hot(61, 64)
  OR021_12_2_train, OR021_12_2_test = concatenate_data(scipy.io.loadmat('./data/12k/OR021@12_2.mat'), scale=scaler, opt=opt)
  OR021_12_2_label = one_hot(62, 64)
  OR021_12_3_train, OR021_12_3_test = concatenate_data(scipy.io.loadmat('./data/12k/OR021@12_3.mat'), scale=scaler, opt=opt)
  OR021_12_3_label = one_hot(63, 64)

if opt.data_48k:
  B007_0_train, B007_0_test = concatenate_data(scipy.io.loadmat('./data/48k/B007_0.mat'), scale=scaler, opt=opt)
  B007_0_label = one_hot(4, 64)
  B007_1_train, B007_1_test = concatenate_data(scipy.io.loadmat('./data/48k/B007_1.mat'), scale=scaler, opt=opt)
  B007_1_label = one_hot(5, 64)
  B007_2_train, B007_2_test = concatenate_data(scipy.io.loadmat('./data/48k/B007_2.mat'), scale=scaler, opt=opt)
  B007_2_label = one_hot(6, 64)
  B007_3_train, B007_3_test = concatenate_data(scipy.io.loadmat('./data/48k/B007_3.mat'), scale=scaler, opt=opt)
  B007_3_label = one_hot(7, 64)

  B014_0_train, B014_0_test = concatenate_data(scipy.io.loadmat('./data/48k/B014_0.mat'), scale=scaler, opt=opt)
  B014_0_label = one_hot(8, 64)
  B014_1_train, B014_1_test = concatenate_data(scipy.io.loadmat('./data/48k/B014_1.mat'), scale=scaler, opt=opt)
  B014_1_label = one_hot(9, 64)
  B014_2_train, B014_2_test = concatenate_data(scipy.io.loadmat('./data/48k/B014_2.mat'), scale=scaler, opt=opt)
  B014_2_label = one_hot(10, 64)
  B014_3_train, B014_3_test= concatenate_data(scipy.io.loadmat('./data/48k/B014_3.mat'), scale=scaler, opt=opt)
  B014_3_label = one_hot(11, 64)

  B021_0_train, B021_0_test = concatenate_data(scipy.io.loadmat('./data/48k/B021_0.mat'), scale=scaler, opt=opt)
  B021_0_label = one_hot(12, 64)
  B021_1_train, B021_1_test = concatenate_data(scipy.io.loadmat('./data/48k/B021_1.mat'), scale=scaler, opt=opt)
  B021_1_label = one_hot(13, 64)
  B021_2_train, B021_2_test = concatenate_data(scipy.io.loadmat('./data/48k/B021_2.mat'), scale=scaler, opt=opt)
  B021_2_label = one_hot(14, 64)
  B021_3_train, B021_3_test = concatenate_data(scipy.io.loadmat('./data/48k/B021_3.mat'), scale=scaler, opt=opt)
  B021_3_label = one_hot(15, 64)

  IR007_0_train, IR007_0_test = concatenate_data(scipy.io.loadmat('./data/48k/IR007_0.mat'), scale=scaler, opt=opt)
  IR007_0_label = one_hot(20, 64)
  IR007_1_train, IR007_1_test = concatenate_data(scipy.io.loadmat('./data/48k/IR007_1.mat'), scale=scaler, opt=opt)
  IR007_1_label = one_hot(21, 64)
  IR007_2_train, IR007_2_test = concatenate_data(scipy.io.loadmat('./data/48k/IR007_2.mat'), scale=scaler, opt=opt)
  IR007_2_label = one_hot(22, 64)
  IR007_3_train, IR007_3_test = concatenate_data(scipy.io.loadmat('./data/48k/IR007_3.mat'), scale=scaler, opt=opt)
  IR007_3_label = one_hot(23, 64)

  IR014_0_train, IR014_0_test = concatenate_data(scipy.io.loadmat('./data/48k/IR014_0.mat'), scale=scaler, opt=opt)
  IR014_0_label = one_hot(24, 64)
  IR014_1_train, IR014_1_test = concatenate_data(scipy.io.loadmat('./data/48k/IR014_1.mat'), scale=scaler, opt=opt)
  IR014_1_label = one_hot(25, 64)
  IR014_2_train, IR014_2_test = concatenate_data(scipy.io.loadmat('./data/48k/IR014_2.mat'), scale=scaler, opt=opt)
  IR014_2_label = one_hot(26, 64)
  IR014_3_train, IR014_3_test = concatenate_data(scipy.io.loadmat('./data/48k/IR014_3.mat'), scale=scaler, opt=opt)
  IR014_3_label = one_hot(27, 64)

  IR021_0_train, IR021_0_test = concatenate_data(scipy.io.loadmat('./data/48k/IR021_0.mat'), scale=scaler, opt=opt)
  IR021_0_label = one_hot(28, 64)
  IR021_1_train, IR021_1_test = concatenate_data(scipy.io.loadmat('./data/48k/IR021_1.mat'), scale=scaler, opt=opt)
  IR021_1_label = one_hot(29, 64)
  IR021_2_train, IR021_2_test = concatenate_data(scipy.io.loadmat('./data/48k/IR021_2.mat'), scale=scaler, opt=opt)
  IR021_2_label = one_hot(30, 64)
  IR021_3_train, IR021_3_test = concatenate_data(scipy.io.loadmat('./data/48k/IR021_3.mat'), scale=scaler, opt=opt)
  IR021_3_label = one_hot(31, 64)

  OR007_12_0_train, OR007_12_0_test = concatenate_data(scipy.io.loadmat('./data/48k/OR007@12_0.mat'), scale=scaler, opt=opt)
  OR007_12_0_label = one_hot(36, 64)
  OR007_12_1_train, OR007_12_1_test = concatenate_data(scipy.io.loadmat('./data/48k/OR007@12_1.mat'), scale=scaler, opt=opt)
  OR007_12_1_label = one_hot(37, 64)
  OR007_12_2_train, OR007_12_2_test = concatenate_data(scipy.io.loadmat('./data/48k/OR007@12_2.mat'), scale=scaler, opt=opt)
  OR007_12_2_label = one_hot(38, 64)
  OR007_12_3_train, OR007_12_3_test = concatenate_data(scipy.io.loadmat('./data/48k/OR007@12_3.mat'), scale=scaler, opt=opt)
  OR007_12_3_label = one_hot(39, 64)

  OR007_3_0_train, OR007_3_0_test = concatenate_data(scipy.io.loadmat('./data/48k/OR007@3_0.mat'), scale=scaler, opt=opt)
  OR007_3_0_label = one_hot(40, 64)
  OR007_3_1_train, OR007_3_1_test = concatenate_data(scipy.io.loadmat('./data/48k/OR007@3_1.mat'), scale=scaler, opt=opt)
  OR007_3_1_label = one_hot(41, 64)
  OR007_3_2_train, OR007_3_2_test = concatenate_data(scipy.io.loadmat('./data/48k/OR007@3_2.mat'), scale=scaler, opt=opt)
  OR007_3_2_label = one_hot(42, 64)
  OR007_3_3_train, OR007_3_3_test = concatenate_data(scipy.io.loadmat('./data/48k/OR007@3_3.mat'), scale=scaler, opt=opt)
  OR007_3_3_label = one_hot(43, 64)

  OR007_6_0_train, OR007_6_0_test = concatenate_data(scipy.io.loadmat('./data/48k/OR007@6_0.mat'), scale=scaler, opt=opt)
  OR007_6_0_label = one_hot(44, 64)
  OR007_6_1_train, OR007_6_1_test = concatenate_data(scipy.io.loadmat('./data/48k/OR007@6_1.mat'), scale=scaler, opt=opt)
  OR007_6_1_label = one_hot(45, 64)
  OR007_6_2_train, OR007_6_2_test = concatenate_data(scipy.io.loadmat('./data/48k/OR007@6_2.mat'), scale=scaler, opt=opt)
  OR007_6_2_label = one_hot(46, 64)
  OR007_6_3_train, OR007_6_3_test = concatenate_data(scipy.io.loadmat('./data/48k/OR007@6_3.mat'), scale=scaler, opt=opt)
  OR007_6_3_label = one_hot(47, 64)

  OR0014_6_0_train, OR0014_6_0_test = concatenate_data(scipy.io.loadmat('./data/48k/OR014@6_0.mat'), scale=scaler, opt=opt)
  OR0014_6_0_label = one_hot(48, 64)
  OR0014_6_1_train, OR0014_6_1_test = concatenate_data(scipy.io.loadmat('./data/48k/OR014@6_1.mat'), scale=scaler, opt=opt)
  OR0014_6_1_label = one_hot(49, 64)
  OR0014_6_2_train, OR0014_6_2_test = concatenate_data(scipy.io.loadmat('./data/48k/OR014@6_2.mat'), scale=scaler, opt=opt)
  OR0014_6_2_label = one_hot(50, 64)
  OR0014_6_3_train, OR0014_6_3_test = concatenate_data(scipy.io.loadmat('./data/48k/OR014@6_3.mat'), scale=scaler, opt=opt)
  OR0014_6_3_label = one_hot(51, 64)

  OR0021_6_0_train, OR0021_6_0_test = concatenate_data(scipy.io.loadmat('./data/48k/OR021@6_0.mat'), scale=scaler, opt=opt)
  OR0021_6_0_label = one_hot(52, 64)
  OR0021_6_1_train, OR0021_6_1_test = concatenate_data(scipy.io.loadmat('./data/48k/OR021@6_1.mat'), scale=scaler, opt=opt)
  OR0021_6_1_label = one_hot(53, 64)
  OR0021_6_2_train, OR0021_6_2_test = concatenate_data(scipy.io.loadmat('./data/48k/OR021@6_2.mat'), scale=scaler, opt=opt)
  OR0021_6_2_label = one_hot(54, 64)
  OR0021_6_3_train, OR0021_6_3_test = concatenate_data(scipy.io.loadmat('./data/48k/OR021@6_3.mat'), scale=scaler, opt=opt)
  OR0021_6_3_label = one_hot(55, 64)

  OR0021_3_0_train, OR0021_3_0_test = concatenate_data(scipy.io.loadmat('./data/48k/OR021@3_0.mat'), scale=scaler, opt=opt)
  OR0021_3_0_label = one_hot(56, 64)
  OR0021_3_1_train, OR0021_3_1_test = concatenate_data(scipy.io.loadmat('./data/48k/OR021@3_1.mat'), scale=scaler, opt=opt)
  OR0021_3_1_label = one_hot(57, 64)
  OR0021_3_2_train, OR0021_3_2_test = concatenate_data(scipy.io.loadmat('./data/48k/OR021@3_2.mat'), scale=scaler, opt=opt)
  OR0021_3_2_label = one_hot(58, 64)
  OR0021_3_3_train, OR0021_3_3_test = concatenate_data(scipy.io.loadmat('./data/48k/OR021@3_3.mat'), scale=scaler, opt=opt)
  OR0021_3_3_label = one_hot(59, 64)

  OR0021_12_0_train, OR0021_12_0_test = concatenate_data(scipy.io.loadmat('./data/48k/OR021@12_0.mat'), scale=scaler, opt=opt)
  OR0021_12_0_label = one_hot(60, 64)
  OR0021_12_1_train, OR0021_12_1_test = concatenate_data(scipy.io.loadmat('./data/48k/OR021@12_1.mat'), scale=scaler, opt=opt)
  OR0021_12_1_label = one_hot(61, 64)
  OR0021_12_2_train, OR0021_12_2_test = concatenate_data(scipy.io.loadmat('./data/48k/OR021@12_2.mat'), scale=scaler, opt=opt)
  OR0021_12_2_label = one_hot(62, 64)
  OR0021_12_3_train, OR0021_12_3_test = concatenate_data(scipy.io.loadmat('./data/48k/OR021@12_3.mat'), scale=scaler, opt=opt)
  OR0021_12_3_label = one_hot(63, 64)

if opt.MFPT_data:
    # Normal data ###############################################
    baseline_1 = divide_sample(scipy.io.loadmat('/content/drive/Shareddrives/newpro112233/signal_machine/MFPT Fault Data Sets/1 - Three Baseline Conditions/baseline_1.mat')['bearing'][0][0][1])
    baseline_2 = divide_sample(scipy.io.loadmat('/content/drive/Shareddrives/newpro112233/signal_machine/MFPT Fault Data Sets/1 - Three Baseline Conditions/baseline_2.mat')['bearing'][0][0][1])
    baseline_3 = divide_sample(scipy.io.loadmat('/content/drive/Shareddrives/newpro112233/signal_machine/MFPT Fault Data Sets/1 - Three Baseline Conditions/baseline_3.mat')['bearing'][0][0][1])
    baseline_1_label = one_hot(0, 3)
    baseline_2_label = one_hot(0, 3)
    baseline_3_label = one_hot(0, 3)
    
    # Outer rate ################################################
    OuterRaceFault_1 = divide_sample(scipy.io.loadmat('/content/drive/Shareddrives/newpro112233/signal_machine/MFPT Fault Data Sets/2 - Three Outer Race Fault Conditions/OuterRaceFault_1.mat')['bearing'][0][0][2])
    OuterRaceFault_2 = divide_sample(scipy.io.loadmat('/content/drive/Shareddrives/newpro112233/signal_machine/MFPT Fault Data Sets/2 - Three Outer Race Fault Conditions/OuterRaceFault_2.mat')['bearing'][0][0][2])
    OuterRaceFault_3 = divide_sample(scipy.io.loadmat('/content/drive/Shareddrives/newpro112233/signal_machine/MFPT Fault Data Sets/2 - Three Outer Race Fault Conditions/OuterRaceFault_3.mat')['bearing'][0][0][2])
    OuterRaceFault_vload_1 = divide_sample(scipy.io.loadmat('/content/drive/Shareddrives/newpro112233/signal_machine/MFPT Fault Data Sets/3 - Seven More Outer Race Fault Conditions/OuterRaceFault_vload_1.mat')['bearing'][0][0][2])
    OuterRaceFault_vload_2 = divide_sample(scipy.io.loadmat('/content/drive/Shareddrives/newpro112233/signal_machine/MFPT Fault Data Sets/3 - Seven More Outer Race Fault Conditions/OuterRaceFault_vload_2.mat')['bearing'][0][0][2])
    OuterRaceFault_vload_3 = divide_sample(scipy.io.loadmat('/content/drive/Shareddrives/newpro112233/signal_machine/MFPT Fault Data Sets/3 - Seven More Outer Race Fault Conditions/OuterRaceFault_vload_3.mat')['bearing'][0][0][2])
    OuterRaceFault_vload_4 = divide_sample(scipy.io.loadmat('/content/drive/Shareddrives/newpro112233/signal_machine/MFPT Fault Data Sets/3 - Seven More Outer Race Fault Conditions/OuterRaceFault_vload_4.mat')['bearing'][0][0][2])
    OuterRaceFault_vload_5 = divide_sample(scipy.io.loadmat('/content/drive/Shareddrives/newpro112233/signal_machine/MFPT Fault Data Sets/3 - Seven More Outer Race Fault Conditions/OuterRaceFault_vload_5.mat')['bearing'][0][0][2])
    OuterRaceFault_vload_6 = divide_sample(scipy.io.loadmat('/content/drive/Shareddrives/newpro112233/signal_machine/MFPT Fault Data Sets/3 - Seven More Outer Race Fault Conditions/OuterRaceFault_vload_6.mat')['bearing'][0][0][2])
    OuterRaceFault_vload_7 = divide_sample(scipy.io.loadmat('/content/drive/Shareddrives/newpro112233/signal_machine/MFPT Fault Data Sets/3 - Seven More Outer Race Fault Conditions/OuterRaceFault_vload_7.mat')['bearing'][0][0][2])
    OuterRaceFault_1_label = one_hot(1, 3)
    OuterRaceFault_2_label = one_hot(1, 3)
    OuterRaceFault_3_label = one_hot(1, 3)
    OuterRaceFault_vload_1_label = one_hot(1, 3)
    OuterRaceFault_vload_2_label = one_hot(1, 3)
    OuterRaceFault_vload_3_label = one_hot(1, 3)
    OuterRaceFault_vload_4_label = one_hot(1, 3)
    OuterRaceFault_vload_5_label = one_hot(1, 3)
    OuterRaceFault_vload_6_label = one_hot(1, 3)
    OuterRaceFault_vload_7_label = one_hot(1, 3)
    
    
    # Inter rate ###################################################################################################################################
    InnerRaceFault_vload_1 = divide_sample(scipy.io.loadmat('/content/drive/Shareddrives/newpro112233/signal_machine/MFPT Fault Data Sets/4 - Seven Inner Race Fault Conditions/InnerRaceFault_vload_1.mat')['bearing'][0][0][2])
    InnerRaceFault_vload_2 = divide_sample(scipy.io.loadmat('/content/drive/Shareddrives/newpro112233/signal_machine/MFPT Fault Data Sets/4 - Seven Inner Race Fault Conditions/InnerRaceFault_vload_2.mat')['bearing'][0][0][2])
    InnerRaceFault_vload_3 = divide_sample(scipy.io.loadmat('/content/drive/Shareddrives/newpro112233/signal_machine/MFPT Fault Data Sets/4 - Seven Inner Race Fault Conditions/InnerRaceFault_vload_3.mat')['bearing'][0][0][2])
    InnerRaceFault_vload_4 = divide_sample(scipy.io.loadmat('/content/drive/Shareddrives/newpro112233/signal_machine/MFPT Fault Data Sets/4 - Seven Inner Race Fault Conditions/InnerRaceFault_vload_4.mat')['bearing'][0][0][2])
    InnerRaceFault_vload_5 = divide_sample(scipy.io.loadmat('/content/drive/Shareddrives/newpro112233/signal_machine/MFPT Fault Data Sets/4 - Seven Inner Race Fault Conditions/InnerRaceFault_vload_5.mat')['bearing'][0][0][2])
    InnerRaceFault_vload_6 = divide_sample(scipy.io.loadmat('/content/drive/Shareddrives/newpro112233/signal_machine/MFPT Fault Data Sets/4 - Seven Inner Race Fault Conditions/InnerRaceFault_vload_6.mat')['bearing'][0][0][2])
    InnerRaceFault_vload_7 = divide_sample(scipy.io.loadmat('/content/drive/Shareddrives/newpro112233/signal_machine/MFPT Fault Data Sets/4 - Seven Inner Race Fault Conditions/InnerRaceFault_vload_7.mat')['bearing'][0][0][2])
    InnerRaceFault_vload_1_label = one_hot(2, 3)
    InnerRaceFault_vload_2_label = one_hot(2, 3)
    InnerRaceFault_vload_3_label = one_hot(2, 3)
    InnerRaceFault_vload_4_label = one_hot(2, 3)
    InnerRaceFault_vload_5_label = one_hot(2, 3)
    InnerRaceFault_vload_6_label = one_hot(2, 3)
    InnerRaceFault_vload_7_label = one_hot(2, 3)
    
if opt.PU_data:
    load = True

    # Training ################################################
    print('Load 1---------------------------------------------\n')
    K002 = load_PU_data('/content/drive/Shareddrives/newpro112233/signal_machine/new_data/training/K002')
    
    KA01 = load_PU_data('/content/drive/Shareddrives/newpro112233/signal_machine/new_data/training/KA01')
    KA05 = load_PU_data('/content/drive/Shareddrives/newpro112233/signal_machine/new_data/training/KA05')
    KA07 = load_PU_data('/content/drive/Shareddrives/newpro112233/signal_machine/new_data/training/KA07')

    KI01 = load_PU_data('/content/drive/Shareddrives/newpro112233/signal_machine/new_data/training/KI01')
    KI05 = load_PU_data('/content/drive/Shareddrives/newpro112233/signal_machine/new_data/training/KI05')
    KI07 = load_PU_data('/content/drive/Shareddrives/newpro112233/signal_machine/new_data/training/KI07')

    # Testing ###################################################
    print('Load 2---------------------------------------------\n')
    K001 = load_PU_data('/content/drive/Shareddrives/newpro112233/signal_machine/new_data/testing/K001')
    
    KA22 = load_PU_data('/content/drive/Shareddrives/newpro112233/signal_machine/new_data/testing/KA22')
    KA04 = load_PU_data('/content/drive/Shareddrives/newpro112233/signal_machine/new_data/testing/KA04')
    KA15 = load_PU_data('/content/drive/Shareddrives/newpro112233/signal_machine/new_data/testing/KA15')
    KA30 = load_PU_data('/content/drive/Shareddrives/newpro112233/signal_machine/new_data/testing/KA30')
    KA16 = load_PU_data('/content/drive/Shareddrives/newpro112233/signal_machine/new_data/testing/KA16')

    KI14 = load_PU_data('/content/drive/Shareddrives/newpro112233/signal_machine/new_data/testing/KI14')
    KI21 = load_PU_data('/content/drive/Shareddrives/newpro112233/signal_machine/new_data/testing/KI21')
    KI17 = load_PU_data('/content/drive/Shareddrives/newpro112233/signal_machine/new_data/testing/KI17')
    KI18 = load_PU_data('/content/drive/Shareddrives/newpro112233/signal_machine/new_data/testing/KI18')
    KI16 = load_PU_data('/content/drive/Shareddrives/newpro112233/signal_machine/new_data/testing/KI16')
    
    # load ###############################################
    print('Load 3---------------------------------------------\n')
    min_ = np.min((K002.shape[1], KA01.shape[1], KA07.shape[1], KI01.shape[1], KI05.shape[1], KI07.shape[1],\
                  K001.shape[1], KA22.shape[1], KA04.shape[1], KA15.shape[1], KA30.shape[1], KA16.shape[1],\
                  KA05.shape[1], KI14.shape[1], KI21.shape[1], KI17.shape[1], KI18.shape[1], KI16.shape[1]))
    
    if load:
        Healthy_train = np.load('/content/drive/Shareddrives/newpro112233/signal_machine/PU_data/case_1/Healthy_train.npy')
        Healthy_train_label = np.load('/content/drive/Shareddrives/newpro112233/signal_machine/PU_data/case_1/Healthy_train_label.npy')
    else:
        Healthy_train = K002[:, :min_]
        Healthy_train_label = one_hot(0, 3)
        with open('/content/drive/Shareddrives/newpro112233/signal_machine/PU_data/case_1/Healthy_train.npy', 'wb') as f:
            np.save(f, Healthy_train)
        with open('/content/drive/Shareddrives/newpro112233/signal_machine/PU_data/case_1/Healthy_train_label.npy', 'wb') as f:
            np.save(f, Healthy_train_label)

    if load:
        OR_Damage_train = np.load('/content/drive/Shareddrives/newpro112233/signal_machine/PU_data/case_1/OR_Damage_train.npy')
        OR_Damage_train_label = np.load('/content/drive/Shareddrives/newpro112233/signal_machine/PU_data/case_1/OR_Damage_train_label.npy')
    else:
        OR_Damage_train = np.concatenate((KA01[:, :min_], KA05[:, :min_], KA07[:, :min_]))
        OR_Damage_train_label = one_hot(1, 3)
        with open('/content/drive/Shareddrives/newpro112233/signal_machine/PU_data/case_1/OR_Damage_train.npy', 'wb') as f:
            np.save(f, OR_Damage_train)
        with open('/content/drive/Shareddrives/newpro112233/signal_machine/PU_data/case_1/OR_Damage_train_label.npy', 'wb') as f:
            np.save(f, OR_Damage_train_label)

    if load:
        IR_Damage_train = np.load('/content/drive/Shareddrives/newpro112233/signal_machine/PU_data/case_1/IR_Damage_train.npy')
        IR_Damage_train_label = np.load('/content/drive/Shareddrives/newpro112233/signal_machine/PU_data/case_1/IR_Damage_train_label.npy')
    else:
        IR_Damage_train = np.concatenate((KI01[:, :min_], KI05[:, :min_], KI07[:, :min_]))
        IR_Damage_train_label = one_hot(2, 3)
        with open('/content/drive/Shareddrives/newpro112233/signal_machine/PU_data/case_1/IR_Damage_train.npy', 'wb') as f:
            np.save(f, IR_Damage_train)
        with open('/content/drive/Shareddrives/newpro112233/signal_machine/PU_data/case_1/IR_Damage_train_label.npy', 'wb') as f:
            np.save(f, IR_Damage_train_label)
    
    if load:
        Healthy_test = np.load('/content/drive/Shareddrives/newpro112233/signal_machine/PU_data/case_1/Healthy_test.npy')
        Healthy_test_label = np.load('/content/drive/Shareddrives/newpro112233/signal_machine/PU_data/case_1/Healthy_test_label.npy')
    else:
        Healthy_test = K001[:, :min_]
        Healthy_test_label = one_hot(0, 3)
        with open('/content/drive/Shareddrives/newpro112233/signal_machine/PU_data/case_1/Healthy_test.npy', 'wb') as f:
            np.save(f, Healthy_test)
        with open('/content/drive/Shareddrives/newpro112233/signal_machine/PU_data/case_1/Healthy_test_label.npy', 'wb') as f:
            np.save(f, Healthy_test_label)
    
    if load:
        OR_Damage_test = np.load('/content/drive/Shareddrives/newpro112233/signal_machine/PU_data/case_1/OR_Damage_test.npy')
        OR_Damage_test_label = np.load('/content/drive/Shareddrives/newpro112233/signal_machine/PU_data/case_1/OR_Damage_test_label.npy')
    else:
        OR_Damage_test = np.concatenate((KA22[:, :min_], KA04[:, :min_], KA15[:, :min_], KA30[:, :min_], KA16[:, :min_]))
        OR_Damage_test_label = one_hot(1, 3)
        with open('/content/drive/Shareddrives/newpro112233/signal_machine/PU_data/case_1/OR_Damage_test.npy', 'wb') as f:
            np.save(f, OR_Damage_test)
        with open('/content/drive/Shareddrives/newpro112233/signal_machine/PU_data/case_1/OR_Damage_test_label.npy', 'wb') as f:
            np.save(f, OR_Damage_test_label)

    if load:
        IR_Damage_test = np.load('/content/drive/Shareddrives/newpro112233/signal_machine/PU_data/case_1/IR_Damage_test.npy')
        IR_Damage_test_label = np.load('/content/drive/Shareddrives/newpro112233/signal_machine/PU_data/case_1/IR_Damage_test_label.npy')
    else:  
        IR_Damage_test = np.concatenate((KI14[:, :min_], KI21[:, :min_], KI17[:, :min_], KI18[:, :min_], KI16[:, :min_]))
        IR_Damage_test_label = one_hot(2, 3)
        with open('/content/drive/Shareddrives/newpro112233/signal_machine/PU_data/case_1/IR_Damage_test.npy', 'wb') as f:
            np.save(f, IR_Damage_test)
        with open('/content/drive/Shareddrives/newpro112233/signal_machine/PU_data/case_1/IR_Damage_test_label.npy', 'wb') as f:
            np.save(f, IR_Damage_test_label)

    print('Finish loading data process!\n')
