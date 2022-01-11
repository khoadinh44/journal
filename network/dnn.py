import keras
import tensorflow as tf

'''
After 200 epochs merge_network() was more overstanding than network(), 
with reach up more than 93% validation accuracy while the accuracy of network() was just ~40%.

network+None | merge_network+None | merge_network+Fourier  | merge_network+SVD_denoise
40%           93%                   ~100% (After 6 epochs)  ~90%
'''

def network(use_network=False, use_Wavelet=False, use_Fourier=False, use_Wavelet_denoise=False, use_SVD=False, use_savitzky_golay=False, none=False):
  if use_network:
    input_ = keras.layers.Input(shape=[400, ])
    hidden3 = keras.layers.Dense(300, activation=tf.keras.layers.ReLU())(input_)
    hidden4 = keras.layers.Dense(100, activation=tf.keras.layers.ReLU())(hidden3)
    concat = keras.layers.concatenate([input_, hidden4])
    output = keras.layers.Dense(6, activation=tf.keras.layers.Softmax())(concat)
    model = keras.models.Model(inputs=[input_], outputs=[output])
  else:
    if use_Wavelet:
      input_A = keras.layers.Input(shape=[300], name="wide_input")
      input_B = keras.layers.Input(shape=[300], name="deep_input") 
    if use_Fourier or use_Wavelet_denoise or use_SVD or use_savitzky_golay or none:
      input_A = keras.layers.Input(shape=[200], name="wide_input")
      input_B = keras.layers.Input(shape=[200], name="deep_input") 
    hidden1 = keras.layers.Dense(300, activation=tf.keras.layers.ReLU())(input_B)
    hidden2 = keras.layers.Dense(100, activation=tf.keras.layers.ReLU())(hidden1)
    concat = keras.layers.concatenate([input_A, hidden2])
    output = keras.layers.Dense(6, activation=tf.keras.layers.Softmax(), name="output")(concat)
    model = keras.models.Model(inputs=[input_A, input_B], outputs=[output])
  return model
