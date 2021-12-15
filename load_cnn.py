import numpy as np
import pandas as pd
import scipy.io
import tensorflow as tf
from preprocessing.denoise_signal import Fourier, SVD_denoise, Wavelet, Wavelet_denoise, savitzky_golay
import matplotlib.pyplot as plt

use_network         = False
use_Fourier         = True
use_savitzky_golay  = True
use_Wavelet         = False
use_Wavelet_denoise = False
use_SVD             = False

def get_spectrogram(waveform):
  # Zero-padding for an audio waveform with less than 16,000 samples.
  input_len = 300
  respect_input_len = 128*128 
  waveform = waveform[:input_len]
  zero_padding = tf.zeros([(respect_input_len)] - tf.shape(waveform), dtype=tf.float32)
  # Cast the waveform tensors' dtype to float32.
  waveform = tf.cast(waveform, dtype=tf.float32)
  # Concatenate the waveform with `zero_padding`, which ensures all audio
  # clips are of the same length.
  equal_length = tf.concat([waveform, zero_padding], 0)
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(equal_length, frame_length=255, frame_step=127, fft_length=128*2-1)
  # spectrogram = tf.signal.stft(equal_length, frame_length=255, frame_step=128-1)
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

Normal_0_X097_DE_time   = Normal_0['X097_DE_time'][:num].reshape((num//200, 200))
Normal_0_X097_FE_time   = Normal_0['X097_FE_time'][:num].reshape((num//200, 200))
Normal_0_X097 = np.concatenate((Normal_0_X097_DE_time, Normal_0_X097_FE_time), axis=1)

B007_0_X122_DE_time     = B007_0['X122_DE_time'][:num].reshape((num//200, 200))
B007_0_X122_FE_time     = B007_0['X122_FE_time'][:num].reshape((num//200, 200))
B007_0_X122 = np.concatenate((B007_0_X122_DE_time, B007_0_X122_FE_time), axis=1)

IR007_0_X122_DE_time    = IR007_0['X109_DE_time'][:num].reshape((num//200, 200))
IR007_0_X122_FE_time    = IR007_0['X109_FE_time'][:num].reshape((num//200, 200))
IR007_0_X122 = np.concatenate((IR007_0_X122_DE_time, IR007_0_X122_FE_time), axis=1)

OR007_3_0_X122_DE_time  = OR007_3_0['X148_DE_time'][:num].reshape((num//200, 200))
OR007_3_0_X122_FE_time  = OR007_3_0['X148_FE_time'][:num].reshape((num//200, 200))
OR007_3_0_X122 = np.concatenate((OR007_3_0_X122_DE_time, OR007_3_0_X122_FE_time), axis=1)

OR007_6_0_X122_DE_time  = OR007_6_0['X135_DE_time'][:num].reshape((num//200, 200))
OR007_6_0_X122_FE_time  = OR007_6_0['X135_FE_time'][:num].reshape((num//200, 200))
OR007_6_0_X122 = np.concatenate((OR007_6_0_X122_DE_time, OR007_6_0_X122_FE_time), axis=1)

OR007_12_0_X122_DE_time = OR007_12_0['X161_DE_time'][:num].reshape((num//200, 200))
OR007_12_0_X122_FE_time = OR007_12_0['X161_FE_time'][:num].reshape((num//200, 200))
OR007_12_0_X122 = np.concatenate((OR007_12_0_X122_DE_time, OR007_12_0_X122_FE_time), axis=1)

Normal_0_X097RPM       = Normal_0['X097RPM']
B007_0_X122RPM         = B007_0['X122RPM']
IR007_0_X122RPM        = IR007_0['X109RPM']
OR007_3_0_X122RPM      = OR007_3_0['X148RPM']
OR007_6_0_X122RPM      = OR007_6_0['X135RPM']
OR007_12_0_X122RPM     = OR007_12_0['X161RPM']

Normal_0_X097 = np.array([get_spectrogram(i) for i in Normal_0_X097])
B007_0_X122 = np.array([get_spectrogram(i) for i in B007_0_X122])
IR007_0_X122 = np.array([get_spectrogram(i) for i in IR007_0_X122])
OR007_3_0_X122 = np.array([get_spectrogram(i) for i in OR007_3_0_X122])
OR007_6_0_X122 = np.array([get_spectrogram(i) for i in OR007_6_0_X122])
OR007_12_0_X122 = np.array([get_spectrogram(i) for i in OR007_12_0_X122])

merge_data = np.concatenate((Normal_0_X097, B007_0_X122, IR007_0_X122, OR007_3_0_X122, OR007_6_0_X122, OR007_12_0_X122))
print(merge_data.shape)
