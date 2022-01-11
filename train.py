import tensorflow as tf
import numpy as np
from network.dnn import network
from sklearn.model_selection import train_test_split
from preprocessing.denoise_signal import signaltonoise_dB
from preprocessing.utils import recall_m, precision_m, f1_m
from load_data import label, use_network, use_Wavelet, use_Fourier, use_Wavelet_denoise, use_SVD, use_savitzky_golay, none

use_network = True

if use_Wavelet:
  from load_data import merge_data_0, merge_data_1, merge_data_2
if use_Fourier or use_Wavelet_denoise or use_SVD or use_savitzky_golay or none:
  from load_data import merge_data

if use_Wavelet:
  X_train, X_test, y_train, y_test = train_test_split(merge_data_2, label, test_size=0.25, random_state=42, shuffle=True)
  X_train_A, X_train_B             = X_train[:, :100, :], X_train[:, 100:, :]
  X_test_A, X_test_B               = X_test[:, :100, :],  X_test[:, 100:, :]

  X_train_A, X_train_B             = X_train_A.reshape(int(X_train_A.shape[0]), 300), X_train_B.reshape(int(X_train_B.shape[0]), 300)   
  X_test_A, X_test_B               = X_test_A.reshape(int(X_test_A.shape[0]), 300),  X_test_B.reshape(int(X_test_B.shape[0]), 300)

if use_Fourier or use_Wavelet_denoise or use_SVD or use_savitzky_golay or none:
  X_train, X_test, y_train, y_test = train_test_split(merge_data, label, test_size=0.25, random_state=42, shuffle=True)
  X_train_A, X_train_B             = X_train[:, :200], X_train[:, 200:]
  X_test_A, X_test_B               = X_test[:, :200],  X_test[:, 200:]

if use_network:
  def train(data=None, labels=None,\
          val_data=None, val_labels=None,\
          network=None, num_epochs=20,\
          batch_size=32, show_metric=True, name_saver=None):

    model = network(use_network = use_network)
    model.compile(optimizer="Adam", loss="mse", metrics=['acc', f1_m, precision_m, recall_m])
    history = model.fit(data, labels, epochs=num_epochs,
                      validation_data=(val_data, val_labels))
    model.save(name_saver)

    _, dnn_train_acc, dnn_train_f1_m, dnn_train_precision_m, dnn_train_recall_m = model.evaluate(data, labels, verbose=0)
    _, dnn_test_acc, dnn_test_f1_m, dnn_test_precision_m, dnn_test_recall_m = model.evaluate(val_data, val_labels, verbose=0)
    np.save('/content/drive/Shareddrives/newpro112233/signal_machine/dnn_history.npy', history)

    np.save('/content/drive/Shareddrives/newpro112233/signal_machine/dnn_train_acc.npy', dnn_train_acc)
    np.save('/content/drive/Shareddrives/newpro112233/signal_machine/dnn_train_f1_m.npy', dnn_train_f1_m)
    np.save('/content/drive/Shareddrives/newpro112233/signal_machine/dnn_train_precision_m.npy', dnn_train_precision_m)
    np.save('/content/drive/Shareddrives/newpro112233/signal_machine/dnn_train_recall_m.npy', dnn_train_recall_m)

    np.save('/content/drive/Shareddrives/newpro112233/signal_machine/dnn_test_acc.npy', dnn_test_acc)
    np.save('/content/drive/Shareddrives/newpro112233/signal_machine/dnn_test_f1_m.npy', dnn_test_f1_m)
    np.save('/content/drive/Shareddrives/newpro112233/signal_machine/dnn_test_precision_m.npy', dnn_test_precision_m)
    np.save('/content/drive/Shareddrives/newpro112233/signal_machine/dnn_test_recall_m.npy', dnn_test_recall_m)
    print('Train: %.3f, Test: %.3f' % (dnn_train_acc, dnn_test_acc))

    # pyplot.plot(history.history['loss'], label='train')
    # pyplot.plot(history.history['val_loss'], label='test')
    # pyplot.legend()
    # pyplot.show()

  train(X_train, y_train, X_test, y_test, network, 100, 32, True, 'model.h5')

else:
  def train(data=None, labels=None,\
            val_data=None, val_labels=None,\
            network=None, num_epochs=20,\
            batch_size=32, show_metric=True, name_saver=None):

    model = network(none = none)
    model.compile(optimizer="Adam", loss="mse", metrics=['acc', f1_m, precision_m, recall_m])
    history = model.fit(data, labels, epochs=num_epochs,
                      validation_data=(val_data, val_labels))
    model.save(name_saver)

    _, best_train_acc, best_train_f1_m, best_train_precision_m, best_train_recall_m = model.evaluate(data, labels, verbose=0)
    _, best_test_acc, best_test_f1_m, best_test_precision_m, best_test_recall_m = model.evaluate(val_data, val_labels, verbose=0)

    np.save('/content/drive/Shareddrives/newpro112233/signal_machine/evaluate/best_history.npy', history)
    
    np.save('/content/drive/Shareddrives/newpro112233/signal_machine/evaluate/best_train_acc.npy', best_train_acc)
    np.save('/content/drive/Shareddrives/newpro112233/signal_machine/evaluate/best_train_f1_m.npy', best_train_f1_m)
    np.save('/content/drive/Shareddrives/newpro112233/signal_machine/evaluate/best_train_precision_m.npy', best_train_precision_m)
    np.save('/content/drive/Shareddrives/newpro112233/signal_machine/evaluate/best_train_recall_m.npy', best_train_recall_m)

    np.save('/content/drive/Shareddrives/newpro112233/signal_machine/evaluate/best_test_acc.npy', best_test_acc)
    np.save('/content/drive/Shareddrives/newpro112233/signal_machine/evaluate/best_test_f1_m.npy', best_test_f1_m)
    np.save('/content/drive/Shareddrives/newpro112233/signal_machine/evaluate/best_test_precision_m.npy', best_test_precision_m)
    np.save('/content/drive/Shareddrives/newpro112233/signal_machine/evaluate/best_test_recall_m.npy', best_test_recall_m)
    print('Train: %.3f, Test: %.3f' % (best_train_acc, best_test_acc))

    # pyplot.plot(history.history['loss'], label='train')
    # pyplot.plot(history.history['val_loss'], label='test')
    # pyplot.legend()
    # pyplot.show()
  train((X_train_A, X_train_B), y_train, (X_test_A, X_test_B), y_test, network, 100, 32, True, 'model.h5')
