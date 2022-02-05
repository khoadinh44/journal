from keras import backend as K
import numpy as np
import tensorflow as tf

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
    a.append(x[all_hop_length: all_hop_length+window_length])
    all_hop_length += hop_length
    window += 1
  return np.array(a)

def concatenate_data(x=None, scale=None, window_length=400, hop_length=200):
  data=None
  for idx, i in enumerate(x):
    if idx == 3:
      data = x[i]
    if idx == 4:
      data = np.concatenate((data, x[i]), axis=1)
      data = data.reshape((-1, 1))
    if idx == 5:
      data = np.concatenate((data, x[i]), axis=0)
  data = scale.fit_transform(data)
  data = data.reshape((-1, ))
  
  data = divide_sample(data, window_length, hop_length)

  return data

def convert_one_hot(x):
  index = None
  for idx, i in enumerate(x):
    if i == 1:
      index = idx
  return index
