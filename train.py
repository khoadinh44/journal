import tensorflow as tf
import pickle
import numpy as np
from network.dnn import network
from sklearn.model_selection import train_test_split
from preprocessing.denoise_signal import signaltonoise_dB
from preprocessing.utils import recall_m, precision_m, f1_m
from load_data import use_model_A, use_model_B, \
                      label, merge_data, \
                      use_Wavelet, use_Fourier, use_Wavelet_denoise, use_SVD, use_savitzky_golay, none
if use_Wavelet:
  from load_data import merge_data_0, merge_data_1, merge_data_2

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


X_train, X_test, y_train, y_test = train_test_split(merge_data, label, test_size=0.25, random_state=42, shuffle=True)

if use_Wavelet:
  X_train_A, X_train_B             = X_train[:, :100, :], X_train[:, 100:, :]
  X_test_A, X_test_B               = X_test[:, :100, :],  X_test[:, 100:, :]

  X_train_A, X_train_B             = X_train_A.reshape(int(X_train_A.shape[0]), 300), X_train_B.reshape(int(X_train_B.shape[0]), 300)   
  X_test_A, X_test_B               = X_test_A.reshape(int(X_test_A.shape[0]), 300),  X_test_B.reshape(int(X_test_B.shape[0]), 300)

if use_model_B:
  X_train_A, X_train_B             = X_train[:, :200], X_train[:, 200:]
  X_test_A, X_test_B               = X_test[:, :200],  X_test[:, 200:]

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

if use_model_A:
  def train(data=None, labels=None,\
          val_data=None, val_labels=None,\
          network=None, num_epochs=20,\
          batch_size=32, show_metric=True, name_saver=None):

    model = network(use_network = use_model_A)
    model.compile(optimizer="Adam", loss="mse", metrics=['acc', f1_m, precision_m, recall_m])
    history = model.fit(data, labels,
                        epochs=num_epochs,
                        callbacks=[callback],
                        validation_data=(val_data, val_labels))
    model.save(name_saver)

    _, model_A_train_acc, model_A_train_f1_m, model_A_train_precision_m, model_A_train_recall_m = model.evaluate(data,    labels, verbose=0)
    _, model_A_test_acc,  model_A_test_f1_m,  model_A_test_precision_m,  model_A_test_recall_m  = model.evaluate(val_data, val_labels, verbose=0)
    with open(f'/content/drive/Shareddrives/newpro112233/signal_machine/{folder}/dnn_history', 'wb') as file_pi:
      pickle.dump(history.history, file_pi)

    np.save(f'/content/drive/Shareddrives/newpro112233/signal_machine/{folder}/model_A_train_acc.npy', model_A_train_acc)
    np.save(f'/content/drive/Shareddrives/newpro112233/signal_machine/{folder}/model_A_train_f1_m.npy', model_A_train_f1_m)
    np.save(f'/content/drive/Shareddrives/newpro112233/signal_machine/{folder}/model_A_train_precision_m.npy', model_A_train_precision_m)
    np.save(f'/content/drive/Shareddrives/newpro112233/signal_machine/{folder}/model_A_train_recall_m.npy', model_A_train_recall_m)

    np.save(f'/content/drive/Shareddrives/newpro112233/signal_machine/{folder}/model_A_test_acc.npy', model_A_test_acc)
    np.save(f'/content/drive/Shareddrives/newpro112233/signal_machine/{folder}/model_A_test_f1_m.npy', model_A_test_f1_m)
    np.save(f'/content/drive/Shareddrives/newpro112233/signal_machine/{folder}/model_A_test_precision_m.npy', model_A_test_precision_m)
    np.save(f'/content/drive/Shareddrives/newpro112233/signal_machine/{folder}/model_A_test_recall_m.npy', model_A_test_recall_m)
    print('Train: %.3f, Test: %.3f' % (dnn_train_acc, dnn_test_acc))

  train(X_train, y_train, X_test, y_test, network, 100, 32, True, 'model.h5')

else:
  def train(data=None, labels=None,\
            val_data=None, val_labels=None,\
            network=None, num_epochs=20,\
            batch_size=32, show_metric=True, name_saver=None):

    model = network(none = none)
    model.compile(optimizer="Adam", loss="mse", metrics=['acc', f1_m, precision_m, recall_m])
    history = model.fit(data, labels, 
                        epochs=num_epochs,
                        callbacks=[callback],
                        validation_data=(val_data, val_labels))
    model.save(name_saver)

    _, model_B_train_acc, model_B_train_f1_m, model_B_train_precision_m, model_B_train_recall_m = model.evaluate(data,    labels, verbose=0)
    _, model_B_test_acc,  model_B_test_f1_m,  model_B_test_precision_m,  model_B_test_recall_m  = model.evaluate(val_data, val_labels, verbose=0)
    
    with open('/content/drive/Shareddrives/newpro112233/signal_machine/{folder}/model_B_history', 'wb') as file_pi:
      pickle.dump(history.history, file_pi)
    
    np.save('/content/drive/Shareddrives/newpro112233/signal_machine/{folder}/model_B_train_acc.npy', model_B_train_acc)
    np.save('/content/drive/Shareddrives/newpro112233/signal_machine/{folder}/model_B_train_f1_m.npy', model_B_train_f1_m)
    np.save('/content/drive/Shareddrives/newpro112233/signal_machine/{folder}/model_B_train_precision_m.npy', model_B_train_precision_m)
    np.save('/content/drive/Shareddrives/newpro112233/signal_machine/{folder}/model_B_train_recall_m.npy', model_B_train_recall_m)

    np.save('/content/drive/Shareddrives/newpro112233/signal_machine/{folder}/model_B_test_acc.npy', model_B_test_acc)
    np.save('/content/drive/Shareddrives/newpro112233/signal_machine/{folder}/model_B_test_f1_m.npy', model_B_test_f1_m)
    np.save('/content/drive/Shareddrives/newpro112233/signal_machine/{folder}/model_B_test_precision_m.npy', model_B_test_precision_m)
    np.save('/content/drive/Shareddrives/newpro112233/signal_machine/{folder}/model_B_test_recall_m.npy', model_B_test_recall_m)
    print('Train: %.3f, Test: %.3f' % (best_train_acc, best_test_acc))

  train((X_train_A, X_train_B), y_train, (X_test_A, X_test_B), y_test, network, 100, 32, True, 'model.h5')
