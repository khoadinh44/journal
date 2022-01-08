import tensorflow as tf
import numpy as np
from network.dnn import network, merge_network
from sklearn.model_selection import train_test_split
from preprocessing.denoise_signal import signaltonoise_dB
from load_data import label, use_network, use_Wavelet, use_Fourier, use_Wavelet_denoise, use_SVD, use_savitzky_golay

if use_Wavelet:
  from load_data import merge_data_0, merge_data_1, merge_data_2
if use_Fourier or use_Wavelet_denoise or use_SVD or use_savitzky_golay:
  from load_data import merge_data

if use_Wavelet:
  X_train, X_test, y_train, y_test = train_test_split(merge_data_2, label, test_size=0.1, random_state=42, shuffle=True)
  X_train_A, X_train_B             = X_train[:, :100, :], X_train[:, 100:, :]
  X_test_A, X_test_B               = X_test[:, :100, :],  X_test[:, 100:, :]

  X_train_A, X_train_B             = X_train_A.reshape(int(X_train_A.shape[0]), 300), X_train_B.reshape(int(X_train_B.shape[0]), 300)   
  X_test_A, X_test_B               = X_test_A.reshape(int(X_test_A.shape[0]), 300),  X_test_B.reshape(int(X_test_B.shape[0]), 300)

if use_Fourier or use_Wavelet_denoise or use_SVD or use_savitzky_golay:
  X_train, X_test, y_train, y_test = train_test_split(merge_data, label, test_size=0.1, random_state=42, shuffle=True)
  X_train_A, X_train_B             = X_train[:, :200], X_train[:, 200:]
  X_test_A, X_test_B               = X_test[:, :200],  X_test[:, 200:]

# Load model------------------------------------------------------------------------------------
def train(data=None, labels=None,\
          val_data=None, val_labels=None,\
          network=None, num_epochs=20,\
          batch_size=32, show_metric=True, name_saver=None):
  model = network(use_Fourier=use_Fourier)
  model.compile(optimizer="Adam", loss="mse", metrics=["mae", "acc"])
  history = model.fit(data, labels, epochs=num_epochs,
                    validation_data=(val_data, val_labels))
  model.save(name_saver)

train((X_train_A, X_train_B), y_train, (X_test_A, X_test_B), y_test, merge_network, 10, 32, True, 'model.h5')

#Test noise:
print(f'Level of noise in One signal: {signaltonoise_dB(X_train)}')
