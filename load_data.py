import numpy as np
import pandas as pd
import scipy.io
from preprocessing.denoise_signal import Fourier, SVD_denoise, Wavelet
import matplotlib.pyplot as plt

use_network=False

# actived merge_network()
use_Fourier=True
use_Wavelet=False
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
all_labels = {0: 'Normal_0', 1: 'B007_0', 2: 'IR007_0', 3: 'OR007_3_0', 4: 'OR007_6_0', 5: 'OR007_12_0'}

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

Normal_0_X097RPM       = Normal_0['X097RPM']
B007_0_X122RPM         = B007_0['X122RPM']
IR007_0_X122RPM        = IR007_0['X109RPM']
OR007_3_0_X122RPM      = OR007_3_0['X148RPM']
OR007_6_0_X122RPM      = OR007_6_0['X135RPM']
OR007_12_0_X122RPM     = OR007_12_0['X161RPM']

if use_network:
  Normal_0_group     = np.concatenate((Normal_0_X097_DE_time.reshape(1, num), Normal_0_X097_FE_time.reshape(1, num)), axis=0)
  Normal_0_reshape   = Normal_0_group.reshape(n, int(num/n)*2)
  B007_0_group       = np.concatenate((B007_0_X122_DE_time.reshape(1, num), B007_0_X122_FE_time.reshape(1, num)), axis=0)
  B007_0_reshape     = B007_0_group.reshape(n, int(num/n)*2)
  IR007_0_group      = np.concatenate((IR007_0_X122_DE_time.reshape(1, num), IR007_0_X122_FE_time.reshape(1, num)), axis=0)
  IR007_0_reshape    = IR007_0_group.reshape(n, int(num/n)*2) 
  OR007_3_0_group    = np.concatenate((OR007_3_0_X122_DE_time.reshape(1, num), OR007_3_0_X122_FE_time.reshape(1, num)), axis=0)
  OR007_3_0_reshape  = OR007_3_0_group.reshape(n, int(num/n)*2) 
  OR007_6_0_group    = np.concatenate((OR007_6_0_X122_DE_time.reshape(1, num), OR007_6_0_X122_FE_time.reshape(1, num)), axis=0)
  OR007_6_0_reshape  = OR007_6_0_group.reshape(n, int(num/n)*2)
  OR007_12_0_group   = np.concatenate((OR007_12_0_X122_DE_time.reshape(1, num), OR007_12_0_X122_FE_time.reshape(1, num)), axis=0)
  OR007_12_0_reshape = OR007_12_0_group.reshape(n, int(num/n)*2)
  merge_data         = np.concatenate((Normal_0_reshape, B007_0_reshape, IR007_0_reshape, OR007_3_0_reshape, OR007_6_0_reshape, OR007_12_0_reshape))

if use_Wavelet:
  Normal_0_X097_DE_time_0, Normal_0_X097_DE_time_1, Normal_0_X097_DE_time_2 = Wavelet(Normal_0_X097_DE_time)
  Normal_0_X097_FE_time_0, Normal_0_X097_FE_time_1, Normal_0_X097_FE_time_2 = Wavelet(Normal_0_X097_FE_time)

  B007_0_X122_DE_time_0, B007_0_X122_DE_time_1, B007_0_X122_DE_time_2 = Wavelet(B007_0_X122_DE_time)
  B007_0_X122_FE_time_0, B007_0_X122_FE_time_1, B007_0_X122_FE_time_2 = Wavelet(B007_0_X122_FE_time)

  IR007_0_X122_DE_time_0, IR007_0_X122_DE_time_1, IR007_0_X122_DE_time_2 = Wavelet(IR007_0_X122_DE_time)
  IR007_0_X122_FE_time_0, IR007_0_X122_FE_time_1, IR007_0_X122_FE_time_2 = Wavelet(IR007_0_X122_FE_time)

  OR007_3_0_X122_DE_time_0, OR007_3_0_X122_DE_time_1, OR007_3_0_X122_DE_time_2 = Wavelet(OR007_3_0_X122_DE_time)
  OR007_3_0_X122_FE_time_0, OR007_3_0_X122_FE_time_1, OR007_3_0_X122_FE_time_2 = Wavelet(OR007_3_0_X122_FE_time)

  OR007_6_0_X122_DE_time_0, OR007_6_0_X122_DE_time_1, OR007_6_0_X122_DE_time_2 = Wavelet(OR007_6_0_X122_DE_time)
  OR007_6_0_X122_FE_time_0, OR007_6_0_X122_FE_time_1, OR007_6_0_X122_FE_time_2 = Wavelet(OR007_6_0_X122_FE_time)

  OR007_12_0_X122_DE_time_0, OR007_12_0_X122_DE_time_1, OR007_12_0_X122_DE_time_2 = SVD_denoise(OR007_12_0_X122_DE_time)
  OR007_12_0_X122_DE_time_0, OR007_12_0_X122_DE_time_1, OR007_12_0_X122_DE_time_2 = SVD_denoise(OR007_12_0_X122_DE_time)

  DE_time_1 = np.concatenate((Normal_0_X097_DE_time_1.reshape(int(num/n), n), B007_0_X122_DE_time_1.reshape(int(num/n), n), IR007_0_X122_DE_time_1.reshape(int(num/n), n), OR007_3_0_X122_DE_time_1.reshape(int(num/n), n), OR007_6_0_X122_DE_time_1.reshape(int(num/n), n), OR007_12_0_X122_DE_time_1.reshape(int(num/n), n)))
  FE_time_1 = np.concatenate((Normal_0_X097_FE_time_1.reshape(int(num/n), n), B007_0_X122_FE_time_1.reshape(int(num/n), n), IR007_0_X122_FE_time_1.reshape(int(num/n), n), OR007_3_0_X122_FE_time_1.reshape(int(num/n), n), OR007_6_0_X122_FE_time_1.reshape(int(num/n), n), OR007_12_0_X122_FE_time_1.reshape(int(num/n), n)))
  merge_data_1 = np.concatenate((DE_time_1, FE_time_1), axis=1)

  DE_time_0 = np.concatenate((Normal_0_X097_DE_time_0.reshape(int(num/n), n), B007_0_X122_DE_time_0.reshape(int(num/n), n), IR007_0_X122_DE_time_0.reshape(int(num/n), n), OR007_3_0_X122_DE_time_0.reshape(int(num/n), n), OR007_6_0_X122_DE_time_0.reshape(int(num/n), n), OR007_12_0_X122_DE_time_0.reshape(int(num/n), n)))
  FE_time_0 = np.concatenate((Normal_0_X097_FE_time_0.reshape(int(num/n), n), B007_0_X122_FE_time_0.reshape(int(num/n), n), IR007_0_X122_FE_time_0.reshape(int(num/n), n), OR007_3_0_X122_FE_time_0.reshape(int(num/n), n), OR007_6_0_X122_FE_time_0.reshape(int(num/n), n), OR007_12_0_X122_FE_time_0.reshape(int(num/n), n)))
  merge_data_0 = np.concatenate((DE_time_0, FE_time_0), axis=1)

  DE_time_1 = np.concatenate((Normal_0_X097_DE_time_1.reshape(int(num/n), n), B007_0_X122_DE_time_1.reshape(int(num/n), n), IR007_0_X122_DE_time_1.reshape(int(num/n), n), OR007_3_0_X122_DE_time_1.reshape(int(num/n), n), OR007_6_0_X122_DE_time_1.reshape(int(num/n), n), OR007_12_0_X122_DE_time_1.reshape(int(num/n), n)))
  FE_time_1 = np.concatenate((Normal_0_X097_FE_time_1.reshape(int(num/n), n), B007_0_X122_FE_time_1.reshape(int(num/n), n), IR007_0_X122_FE_time_1.reshape(int(num/n), n), OR007_3_0_X122_FE_time_1.reshape(int(num/n), n), OR007_6_0_X122_FE_time_1.reshape(int(num/n), n), OR007_12_0_X122_FE_time_1.reshape(int(num/n), n)))
  merge_data_1 = np.concatenate((DE_time_1, FE_time_1), axis=1)

  DE_time_2 = np.concatenate((Normal_0_X097_DE_time_2.reshape(int(num/n), n), B007_0_X122_DE_time_2.reshape(int(num/n), n), IR007_0_X122_DE_time_2.reshape(int(num/n), n), OR007_3_0_X122_DE_time_2.reshape(int(num/n), n), OR007_6_0_X122_DE_time_2.reshape(int(num/n), n), OR007_12_0_X122_DE_time_2.reshape(int(num/n), n)))
  FE_time_2 = np.concatenate((Normal_0_X097_FE_time_2.reshape(int(num/n), n), B007_0_X122_FE_time_2.reshape(int(num/n), n), IR007_0_X122_FE_time_2.reshape(int(num/n), n), OR007_3_0_X122_FE_time_2.reshape(int(num/n), n), OR007_6_0_X122_FE_time_2.reshape(int(num/n), n), OR007_12_0_X122_FE_time_2.reshape(int(num/n), n)))
  merge_data_2 = np.concatenate((DE_time_2, FE_time_2), axis=1)



if use_Fourier:
  Normal_0_X097_DE_time   = Fourier(f=Normal_0_X097_DE_time, num=num, get_result=True, thres=25)
  Normal_0_X097_FE_time   = Fourier(f=Normal_0_X097_FE_time, num=num, get_result=True, thres=85)

  B007_0_X122_DE_time     = Fourier(f=B007_0_X122_DE_time, num=num, get_result=True, thres=25)
  B007_0_X122_FE_time     = Fourier(f=B007_0_X122_FE_time, num=num, get_result=True, thres=85)

  IR007_0_X122_DE_time    = Fourier(f=IR007_0_X122_DE_time, num=num, get_result=True, thres=25)
  IR007_0_X122_FE_time    = Fourier(f=IR007_0_X122_FE_time, num=num, get_result=True, thres=85)

  OR007_3_0_X122_DE_time  = Fourier(f=OR007_3_0_X122_DE_time, num=num, get_result=True, thres=25)
  OR007_3_0_X122_FE_time  = Fourier(f=OR007_3_0_X122_FE_time, num=num, get_result=True, thres=85)

  OR007_6_0_X122_DE_time  = Fourier(f=OR007_6_0_X122_DE_time, num=num, get_result=True, thres=25)
  OR007_6_0_X122_FE_time  = Fourier(f=OR007_6_0_X122_FE_time, num=num, get_result=True, thres=85)

  OR007_12_0_X122_DE_time = Fourier(f=OR007_12_0_X122_DE_time, num=num, get_result=True, thres=25)
  OR007_12_0_X122_FE_time = Fourier(f=OR007_12_0_X122_FE_time, num=num, get_result=True, thres=85)

  DE_time = np.concatenate((Normal_0_X097_DE_time.reshape(int(num/n), n), B007_0_X122_DE_time.reshape(int(num/n), n), IR007_0_X122_DE_time.reshape(int(num/n), n), OR007_3_0_X122_DE_time.reshape(int(num/n), n), OR007_6_0_X122_DE_time.reshape(int(num/n), n), OR007_12_0_X122_DE_time.reshape(int(num/n), n)))
  FE_time = np.concatenate((Normal_0_X097_FE_time.reshape(int(num/n), n), B007_0_X122_FE_time.reshape(int(num/n), n), IR007_0_X122_FE_time.reshape(int(num/n), n), OR007_3_0_X122_FE_time.reshape(int(num/n), n), OR007_6_0_X122_FE_time.reshape(int(num/n), n), OR007_12_0_X122_FE_time.reshape(int(num/n), n)))

  merge_data = np.concatenate((DE_time, FE_time), axis=1)
