import numpy as np
import pandas as pd
import scipy.io
import tensorflow as tf
from preprocessing.denoise_signal import Fourier, SVD_denoise, Wavelet, Wavelet_denoise
import matplotlib.pyplot as plt

use_network=False

# actived merge_network()
use_Fourier         = True
use_Wavelet         = False
use_Wavelet_denoise = False
use_SVD             = True

def get_spectrogram(waveform):
  waveform = waveform.reshape(int(waveform.shape[0]), )
  # Zero-padding for an audio waveform with less than 16,000 samples.
  input_len = 124600
  waveform = waveform[:input_len]
  zero_padding = tf.zeros([124600] - tf.shape(waveform), dtype=tf.float32)
  # Cast the waveform tensors' dtype to float32.
  waveform = tf.cast(waveform, dtype=tf.float32)
  # Concatenate the waveform with `zero_padding`, which ensures all audio
  # clips are of the same length.
  equal_length = tf.concat([waveform, zero_padding], 0)
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(equal_length, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

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

print('get_spectrogram')
spec = get_spectrogram(Normal_0_X097_DE_time)
print(spec)

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

if use_Fourier:
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
  
  if use_Wavelet==False:
    DE_time = np.concatenate((Normal_0_X097_DE_time.reshape(int(num/n), n), B007_0_X122_DE_time.reshape(int(num/n), n), IR007_0_X122_DE_time.reshape(int(num/n), n), OR007_3_0_X122_DE_time.reshape(int(num/n), n), OR007_6_0_X122_DE_time.reshape(int(num/n), n), OR007_12_0_X122_DE_time.reshape(int(num/n), n)))
    FE_time = np.concatenate((Normal_0_X097_FE_time.reshape(int(num/n), n), B007_0_X122_FE_time.reshape(int(num/n), n), IR007_0_X122_FE_time.reshape(int(num/n), n), OR007_3_0_X122_FE_time.reshape(int(num/n), n), OR007_6_0_X122_FE_time.reshape(int(num/n), n), OR007_12_0_X122_FE_time.reshape(int(num/n), n)))

    merge_data = np.concatenate((DE_time, FE_time), axis=1)

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

  OR007_12_0_X122_DE_time_0, OR007_12_0_X122_DE_time_1, OR007_12_0_X122_DE_time_2 = Wavelet(OR007_12_0_X122_DE_time)
  OR007_12_0_X122_FE_time_0, OR007_12_0_X122_FE_time_1, OR007_12_0_X122_FE_time_2 = Wavelet(OR007_12_0_X122_FE_time)

  m = int(n/4)
  num = 31150
  DE_time_0 = np.concatenate((Normal_0_X097_DE_time_0.reshape(int(num/m), m), B007_0_X122_DE_time_0.reshape(int(num/m), m), IR007_0_X122_DE_time_0.reshape(int(num/m), m), OR007_3_0_X122_DE_time_0.reshape(int(num/m), m), OR007_6_0_X122_DE_time_0.reshape(int(num/m), m), OR007_12_0_X122_DE_time_0.reshape(int(num/m), m)))
  FE_time_0 = np.concatenate((Normal_0_X097_FE_time_0.reshape(int(num/m), m), B007_0_X122_FE_time_0.reshape(int(num/m), m), IR007_0_X122_FE_time_0.reshape(int(num/m), m), OR007_3_0_X122_FE_time_0.reshape(int(num/m), m), OR007_6_0_X122_FE_time_0.reshape(int(num/m), m), OR007_12_0_X122_FE_time_0.reshape(int(num/m), m)))
  merge_data_0 = np.concatenate((DE_time_0, FE_time_0), axis=1)

  DE_time_1 = np.concatenate((Normal_0_X097_DE_time_1.reshape(int(num/m), m, 3), B007_0_X122_DE_time_1.reshape(int(num/m), m, 3), IR007_0_X122_DE_time_1.reshape(int(num/m), m, 3), OR007_3_0_X122_DE_time_1.reshape(int(num/m), m, 3), OR007_6_0_X122_DE_time_1.reshape(int(num/m), m, 3), OR007_12_0_X122_DE_time_1.reshape(int(num/m), m, 3)))
  FE_time_1 = np.concatenate((Normal_0_X097_FE_time_1.reshape(int(num/m), m, 3), B007_0_X122_FE_time_1.reshape(int(num/m), m, 3), IR007_0_X122_FE_time_1.reshape(int(num/m), m, 3), OR007_3_0_X122_FE_time_1.reshape(int(num/m), m, 3), OR007_6_0_X122_FE_time_1.reshape(int(num/m), m, 3), OR007_12_0_X122_FE_time_1.reshape(int(num/m), m, 3)))
  merge_data_1 = np.concatenate((DE_time_1, FE_time_1), axis=1)

  m = int(n/2)
  num = 62300
  DE_time_2 = np.concatenate((Normal_0_X097_DE_time_2.reshape(int(num/m), m, 3), B007_0_X122_DE_time_2.reshape(int(num/m), m, 3), IR007_0_X122_DE_time_2.reshape(int(num/m), m, 3), OR007_3_0_X122_DE_time_2.reshape(int(num/m), m, 3), OR007_6_0_X122_DE_time_2.reshape(int(num/m), m, 3), OR007_12_0_X122_DE_time_2.reshape(int(num/m), m, 3)))
  FE_time_2 = np.concatenate((Normal_0_X097_FE_time_2.reshape(int(num/m), m, 3), B007_0_X122_FE_time_2.reshape(int(num/m), m, 3), IR007_0_X122_FE_time_2.reshape(int(num/m), m, 3), OR007_3_0_X122_FE_time_2.reshape(int(num/m), m, 3), OR007_6_0_X122_FE_time_2.reshape(int(num/m), m, 3), OR007_12_0_X122_FE_time_2.reshape(int(num/m), m, 3)))
  merge_data_2 = np.concatenate((DE_time_2, FE_time_2), axis=1)

  Normal_0_name    = [[1, 0, 0, 0, 0, 0]]*int(num/(n/2))
  B007_0_name      = [[0, 1, 0, 0, 0, 0]]*int(num/(n/2))
  IR007_0_name     = [[0, 0, 1, 0, 0, 0]]*int(num/(n/2))
  OR007_3_0_name   = [[0, 0, 0, 1, 0, 0]]*int(num/(n/2))
  OR007_6_0_name   = [[0, 0, 0, 0, 1, 0]]*int(num/(n/2))
  OR007_12_0_name  = [[0, 0, 0, 0, 0, 1]]*int(num/(n/2))

  label = np.concatenate((Normal_0_name, B007_0_name, IR007_0_name, OR007_3_0_name, OR007_6_0_name, OR007_12_0_name))

if use_Wavelet_denoise:
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
  if use_Wavelet==False:
    DE_time = np.concatenate((Normal_0_X097_DE_time.reshape(int(num/n), n), B007_0_X122_DE_time.reshape(int(num/n), n), IR007_0_X122_DE_time.reshape(int(num/n), n), OR007_3_0_X122_DE_time.reshape(int(num/n), n), OR007_6_0_X122_DE_time.reshape(int(num/n), n), OR007_12_0_X122_DE_time.reshape(int(num/n), n)))
    FE_time = np.concatenate((Normal_0_X097_FE_time.reshape(int(num/n), n), B007_0_X122_FE_time.reshape(int(num/n), n), IR007_0_X122_FE_time.reshape(int(num/n), n), OR007_3_0_X122_FE_time.reshape(int(num/n), n), OR007_6_0_X122_FE_time.reshape(int(num/n), n), OR007_12_0_X122_FE_time.reshape(int(num/n), n)))

    merge_data = np.concatenate((DE_time, FE_time), axis=1)

if use_SVD:
  use = Normal_0_X097_DE_time
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
  if use_Wavelet==False:
    DE_time = np.concatenate((Normal_0_X097_DE_time.reshape(int(num/n), n), B007_0_X122_DE_time.reshape(int(num/n), n), IR007_0_X122_DE_time.reshape(int(num/n), n), OR007_3_0_X122_DE_time.reshape(int(num/n), n), OR007_6_0_X122_DE_time.reshape(int(num/n), n), OR007_12_0_X122_DE_time.reshape(int(num/n), n)))
    FE_time = np.concatenate((Normal_0_X097_FE_time.reshape(int(num/n), n), B007_0_X122_FE_time.reshape(int(num/n), n), IR007_0_X122_FE_time.reshape(int(num/n), n), OR007_3_0_X122_FE_time.reshape(int(num/n), n), OR007_6_0_X122_FE_time.reshape(int(num/n), n), OR007_12_0_X122_FE_time.reshape(int(num/n), n)))

    merge_data = np.concatenate((DE_time, FE_time), axis=1)
