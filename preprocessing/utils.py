from keras import backend as K
import os
import numpy as np
import tensorflow as tf
import scipy.io
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from preprocessing.extract_features import AudioFeatureExtractor
from preprocessing.denoise_signal import savitzky_golay, Fourier, SVD_denoise, Wavelet_denoise

def load_PU_data(path):
  data = []
  all_data = []
  min_l = 0
  for name in os.listdir(path):
    if name.split('.')[-1] == 'mat':
      path_signal = path + '/' + name
      file_name = path_signal.split('/')[-1]
      name = file_name.split('.')[0]
      signal = scipy.io.loadmat(path_signal)[name]
      signal = signal[0][0][2][0][6][2]  #Take out the data
      signal = signal.reshape(-1, )
      if min_l == 0:
        min_l = int(signal.shape[0])
      elif min_l > signal.shape[0]:
        min_l = int(signal.shape[0])   
      all_data.append(signal)
      
    for i in all_data:
      each_data = i[:min_l].tolist()
      data.append(each_data)
  return np.array(data)

def add_noise(signal, SNRdb, case_1=True, case_2=False):
  if len(signal.shape)>1:
    data = np.array([add_each_noise(i, SNRdb, case_1, case_2) for i in signal])
  else:
    data = add_each_noise(signal, SNRdb, case_1, case_2)
  return data

def add_each_noise(signal, SNRdb, case_1, case_2):
  np.random.seed()
  mean_S = np.mean(signal)
  mean_S_2 = np.mean(np.power(signal, 2))
  signal_diff = signal - mean_S
  # var_S = np.sum(np.mean(signal_diff**2)) 

  if case_1:
    mean_N = 0
    std = np.sqrt(mean_S_2/np.power(10, (SNRdb/10)))
  if case_2:
    mean_N = mean_S
    std = np.sqrt((mean_S_2/np.power(10, (SNRdb/10)))- np.power(mean_S, 2))
  noise = np.random.normal(mean_N, std, size=len(signal))
  noise_signal = signal + noise
  return noise_signal

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
  
def accuracy_m(y_true, y_pred):
  correct = 0
  total = 0
  for i in range(len(y_true)):
      act_label = np.argmax(y_true[i]) # act_label = 1 (index)
      pred_label = np.argmax(y_pred[i]) # pred_label = 1 (index)
      if(act_label == pred_label):
          correct += 1
      total += 1
  accuracy = (correct/total)
  return accuracy

def signal_to_IFMs(x):
    '''
    x: input signal
    The input signal is normalized to the bound of [0, 1]
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9211470
    '''
    min = np.min(x)
    max = np.max(x)
    return (x-min)/(max-min)
    
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

def one_hot(pos, num_class):
    num = np.zeros((1, num_class))
    num[0, pos] = 1
    return num

def divide_sample(x=None, window_length=400, hop_length=200):
  '''
  The shape of x must be (n_sample, )
  '''
  if len(x.shape) > 1:
    x = x.reshape(-1, )
  a = []
  window = 0
  all_hop_length = 0
  num_window = (x.shape[0]-(window_length-hop_length))//hop_length
  while window < num_window:
    if len(x[all_hop_length: all_hop_length+window_length])==window_length:
      a.append(x[all_hop_length: all_hop_length+window_length])
    all_hop_length += hop_length
    window += 1
  return np.array(a)

def handcrafted_features(x):
    data = []
    afe = AudioFeatureExtractor(400, 200, 1) # 22050, 1024, 4
    for i in x:
        extract_rms = afe.extract_rms(i)
        extract_spectral_centroid  = afe.extract_spectral_centroid(i)
        extract_spectral_bandwidth = afe.extract_spectral_bandwidth(i)
        extract_spectral_flatness  = afe.extract_spectral_flatness(i)
        extract_spectral_rolloff   = afe.extract_spectral_rolloff(i)
        all_i = np.concatenate((extract_rms, extract_spectral_centroid, extract_spectral_bandwidth, extract_spectral_flatness, extract_spectral_rolloff), axis=1)
        all_i = np.ndarray.flatten(all_i)  # convert the vectors of 400 samples to the vectors of 45 samples
        data.append(all_i)
    return np.array(data)
  
def scaler(signal, scale_method):
  scale = scale_method().fit(signal)
  return scale.transform(signal), scale

def concatenate_data(x=None, scale=None, window_length=400, hop_length=200, hand_fea=True, SNdb=10, opt=None):
  X_train_all = []
  X_test = []
  for idx, i in enumerate(x):
    if len(x[i]) > 80:
      if X_train_all == []:
        data = x[i].reshape(-1, 1)
        X_train_all, X_test = train_test_split(data, test_size=opt.test_rate, random_state=42, shuffle=False)
      else:
        data = x[i].reshape(-1, 1)
        each_X_train_all, each_X_test = train_test_split(data, test_size=opt.test_rate, random_state=42, shuffle=False)

        X_train_all = np.concatenate((X_train_all, each_X_train_all), axis=0)
        X_test = np.concatenate((X_test, each_X_test), axis=0)
  
  X_train_all = divide_sample(X_train_all, window_length, hop_length)
  X_test = divide_sample(X_test, window_length, hop_length)

  # Denoising methods ###################################################################################
  if opt.denoise == 'DFK':
    X_train_all = use_denoise(X_train_all, Fourier)
    X_test = use_denoise(X_test, Fourier)
  elif opt.denoise == 'Wavelet_denoise':
    X_train_all = use_denoise(X_train_all, Wavelet_denoise)
    X_test = use_denoise(X_test, Wavelet_denoise)
  elif opt.denoise == 'SVD':
    X_train_all = use_denoise(X_train_all, SVD_denoise)
    X_test = use_denoise(X_test, SVD_denoise)
  elif opt.denoise == 'savitzky_golay':
    X_train_all = use_denoise(X_train_all, savitzky_golay)
    X_test = use_denoise(X_test, savitzky_golay)
  
  # Normalizing methods ################################################################################
  if opt.scaler != None:
    X_train_all = np.squeeze(X_train_all)
    X_test      = np.squeeze(X_test)

    if opt.scaler == 'MinMaxScaler':
      X_train_all, scale = scaler(X_train_all, MinMaxScaler)
      X_test = scale.transform(X_test)
    elif opt.scaler == 'MaxAbsScaler':
      X_train_all, scale = scaler(X_train_all, MaxAbsScaler)
      X_test = scale.transform(X_test)
    elif opt.scaler == 'StandardScaler':
      X_train_all, scale = scaler(X_train_all, StandardScaler)
      X_test = scale.transform(X_test)
    elif opt.scaler == 'RobustScaler':
      X_train_all, scale = scaler(X_train_all, RobustScaler)
      X_test = scale.transform(X_test)
    elif opt.scaler == 'Normalizer':
      X_train_all, scale = scaler(X_train_all, Normalizer)
      X_test = scale.transform(X_test)
    elif opt.scaler == 'QuantileTransformer':
      X_train_all, scale = scaler(X_train_all, QuantileTransformer)
      X_test = scale.transform(X_test)
    elif opt.scaler == 'PowerTransformer':
      X_train_all, scale = scaler(X_train_all, PowerTransformer)
      X_test = scale.transform(X_test)
    elif opt.scaler == 'handcrafted_features':
      X_train_all = handcrafted_features(X_train_all)
      X_test = handcrafted_features(X_test)
      
  return X_train_all, X_test

def convert_one_hot(x, state=True):
    if state == False:
      index = None
      x = np.squeeze(x)

      for idx, i in enumerate(x):
        if i == 1:
          index = idx
      return [index]
    else:
      return x.tolist()

def invert_one_hot(x):
  labels = []
  for i in x:
    each = i.tolist()
    labels.append(each.index(1))
  return np.array(labels)

def use_denoise(x, denoise_method):
  data = []
  for i in x:
    data.append(denoise_method(i))  
  return np.array(data)
