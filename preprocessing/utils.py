from keras import backend as K
import numpy as np
import tensorflow as tf
from preprocessing.extract_features import AudioFeatureExtractor
from preprocessing.denoise_signal import savitzky_golay, Fourier, SVD_denoise, Wavelet_denoise

def add_noise(signal, SNRdb, case_1=True, case_2=False):
  if len(signal.shape)>1:
    data = [add_each_noise(i, SNRdb, case_1, case_2) for i in signal]
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

def signaltonoise_dB(x, y):
    '''
    x: pure signal
    y: noised signal
    '''
#     a = np.asanyarray(a[0, :])
#     m = a.mean(axis)
#     sd = a.std(axis=axis, ddof=ddof)
#     return 20*np.log10(abs(np.where(sd == 0, 0, m/sd)))
    return np.sqrt(np.abs(np.sum(y))/ np.sum(np.square(x)))
    

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

def divide_sample(x, window_length, hop_length):
  '''
  The shape of x must be (n_sample, )
  '''
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
        # print(all_i.shape)
        data.append(all_i)
    return np.array(data)

def concatenate_data(x=None, scale=None, window_length=400, hop_length=200, hand_fea=True, SNRdb=10):
  data = []
  for idx, i in enumerate(x):
    if len(x[i]) > 80:
      if data == []:
        data = x[i]
      else:
        if hand_fea == False:
          if int(data.shape[0]) < int(x[i].shape[0]):
            row = int(data.shape[0])
            data = np.concatenate((data, x[i][:row, :]), axis=1)
          else:
            row = int(x[i].shape[0])
            data = np.concatenate((data[:row, :], x[i]), axis=1)
        else:
          if int(data.shape[0]) < int(x[i].shape[0]):
            data = np.concatenate((data, x[i]), axis=0)

  data = data.reshape(-1, 1)
  if scale != None:
    data = scale.fit_transform(data)
  data = data.reshape((-1, ))
  # data = add_noise(data, SNRdb)
  # data = Fourier(data)
  data = divide_sample(data, window_length, hop_length)
  return data

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

def invert_one_hot(x, num_class):
    all_labels = []
    label_zeros = np.zeros((num_class, ))
    for i in x:
        label = np.copy(label_zeros)
        label[i] = 1
        all_labels.append(label)
    return np.array(all_labels)

def use_denoise(x, denoise_method):
  data = []
  for i in x:
    data.append(denoise_method(i))  
  return np.array(data)
