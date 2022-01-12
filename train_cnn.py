import tensorflow as tf
import numpy as np
from network.cnn import network
from sklearn.model_selection import train_test_split
from preprocessing.denoise_signal import signaltonoise_dB
from preprocessing.utils import recall_m, precision_m, f1_m
from load_cnn import merge_data, label
import pickle
from load_data import use_Fourier, use_Wavelet_denoise, use_SVD, use_savitzky_golay

X_train, X_test, y_train, y_test = train_test_split(merge_data, label, test_size=0.25, random_state=42, shuffle=True)

if use_Fourier:
  folder = 'Fourier'
elif use_Wavelet_denoise:
  folder = 'Wavelet_denoise'
elif use_SVD:
  folder = 'SVD'
elif use_savitzky_golay:
  folder = 'savitzky_golay' 
else:
  folder = 'evaluate' 

use_callback = False
if use_callback:  
  callback = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)]
else:
  callback = None

def train(data=None,     labels=None,\
          val_data=None, val_labels=None,\
          network=None,  num_epochs=20,\
          batch_size=32, show_metric=True, name_saver=None):

  model = network()
  model.compile(optimizer="Adam", loss="mse", metrics=['acc', f1_m, precision_m, recall_m])
  history = model.fit(data, labels, 
                      epochs=num_epochs,
                      callbacks=callback,
                      validation_data=(val_data, val_labels))
  model.save(name_saver)

  _, cnn_train_acc, cnn_train_f1_m, cnn_train_precision_m, cnn_train_recall_m = model.evaluate(data, labels, verbose=0)
  _, cnn_test_acc, cnn_test_f1_m, cnn_test_precision_m, cnn_test_recall_m = model.evaluate(val_data, val_labels, verbose=0)
  with open('/content/drive/Shareddrives/newpro112233/signal_machine/{folder}/cnn_history', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

  np.save(f'/content/drive/Shareddrives/newpro112233/signal_machine/{folder}/cnn_train_acc.npy', cnn_train_acc)
  np.save(f'/content/drive/Shareddrives/newpro112233/signal_machine/{folder}/cnn_train_f1_m.npy', cnn_train_f1_m)
  np.save(f'/content/drive/Shareddrives/newpro112233/signal_machine/{folder}/cnn_train_precision_m.npy', cnn_train_precision_m)
  np.save(f'/content/drive/Shareddrives/newpro112233/signal_machine/{folder}/cnn_train_recall_m.npy', cnn_train_recall_m)

  np.save(f'/content/drive/Shareddrives/newpro112233/signal_machine/{folder}/cnn_test_acc.npy', cnn_test_acc)
  np.save(f'/content/drive/Shareddrives/newpro112233/signal_machine/{folder}/cnn_test_f1_m.npy', cnn_test_f1_m)
  np.save(f'/content/drive/Shareddrives/newpro112233/signal_machine/{folder}/cnn_test_precision_m.npy', cnn_test_precision_m)
  np.save(f'/content/drive/Shareddrives/newpro112233/signal_machine/{folder}/cnn_test_recall_m.npy', cnn_test_recall_m)
  print('Train: %.3f, Test: %.3f' % (cnn_train_acc, cnn_test_acc))
train(X_train, y_train, X_test, y_test, network, 100, 32, True, 'model.h5')
