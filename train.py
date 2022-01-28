import tensorflow as tf
import pickle
import numpy as np
import argparse
from network.nn import DNN_A, DNN_B, CNN_A, CNN_B
from sklearn.model_selection import train_test_split
from preprocessing.utils import recall_m, precision_m, f1_m, signaltonoise_dB
from load_data import load_all

def train(data, labels,
          val_data, val_labels,
          network, num_epochs, batch_size, name_saver, folder, opt):
  
  if opt.use_CNN_A:
    data = np.expand_dims(data, axis=-1)
    val_data = np.expand_dims(val_data, axis=-1)
    
  model = network()
  model.compile(optimizer="Adam", loss="mse", metrics=['acc', f1_m, precision_m, recall_m])
  model.summary()

  history = model.fit(data, labels,
                      epochs     = num_epochs,
                      batch_size = batch_size,
                      validation_data=(val_data, val_labels))
  model.save(name_saver)

#   _, model_A_train_acc, model_A_train_f1_m, model_A_train_precision_m, model_A_train_recall_m = model.evaluate(data,     labels,     verbose=0)
#   _, model_A_test_acc,  model_A_test_f1_m,  model_A_test_precision_m,  model_A_test_recall_m  = model.evaluate(val_data, val_labels, verbose=0)

  if opt.type == 'use_DNN_A':
    with open(f'/content/drive/Shareddrives/newpro112233/signal_machine/{folder}/DNN_A_history', 'wb') as file_pi:
      pickle.dump(history.history, file_pi)
  elif opt.type == 'use_DNN_B':
    with open(f'/content/drive/Shareddrives/newpro112233/signal_machine/{folder}/DNN_B_history', 'wb') as file_pi:
      pickle.dump(history.history, file_pi)
  if opt.type == 'use_CNN_A':
    with open(f'/content/drive/Shareddrives/newpro112233/signal_machine/{folder}/CNN_A_history', 'wb') as file_pi:
      pickle.dump(history.history, file_pi)
  elif opt.type == 'use_CNN_A':
    with open(f'/content/drive/Shareddrives/newpro112233/signal_machine/{folder}/CNN_B_history', 'wb') as file_pi:
      pickle.dump(history.history, file_pi)
  

    
def main(opt):
  callback = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)]
  # The direction for saving---------------------------------------------------------------
  if opt.denoise == 'Fourier':
    folder = 'Fourier'
  elif opt.denoise == 'Wavelet_denoise':
    folder = 'Wavelet_denoise'
  elif opt.denoise == 'SVD':
    folder = 'SVD'
  elif opt.denoise == 'savitzky_golay':
    folder = 'savitzky_golay' 
  else:
    folder = 'none_denoise' 
  #---------------------------------------------------------------------------------------  
   

  # Loading data-----------------------------------------------------------------------
  merge_data, label = load_all(opt)
  X_train, X_test, y_train, y_test = train_test_split(merge_data, label, test_size=opt.test_rate, random_state=42, shuffle=True)

  if opt.use_DNN_B:
    X_train_A, X_train_B             = X_train[:, :200], X_train[:, 200:]
    X_test_A, X_test_B               = X_test[:, :200],  X_test[:, 200:]
  #---------------------------------------------------------------------------------------
  

  if opt.use_DNN_A:
    train(X_train, y_train, X_test, y_test, DNN_A, opt.epochs, opt.batch_size, opt.save, folder, opt)
  elif opt.use_DNN_B:
    train((X_train_A, X_train_B), y_train, (X_test_A, X_test_B), y_test, DNN_B, opt.epochs, opt.batch_size, opt.save, folder, opt)
  elif opt.use_CNN_A:
    train(X_train, y_train, X_test, y_test, CNN_A, opt.epochs, opt.batch_size, opt.save, folder, opt)
  elif opt.use_CNN_B:
    train(X_train, y_train, X_test, y_test, CNN_B, opt.epochs, opt.batch_size, opt.save, folder, opt)
    
  
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    
    # Models and denoising methods--------------------------
    parser.add_argument('--use_DNN_A', default=False, type=bool)
    parser.add_argument('--use_DNN_B', default=False, type=bool)
    parser.add_argument('--use_CNN_A', default=False, type=bool)
    parser.add_argument('--use_CNN_B', default=False, type=bool)
    parser.add_argument('--denoise', type=str, default=None, help='types of NN: DFK, Wavelet_denoise, SVD, savitzky_golay, None. DFK is our proposal.')
    
    # Parameters---------------------------------------------
    parser.add_argument('--save', type=str, default='model.h5', help='Position to save weights')
    parser.add_argument('--epochs', type=int, default=100, help='Number of iterations for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of batch size for training')
    parser.add_argument('--test_rate', type=float, default=0.25, help='rate of split data for testing')
    parser.add_argument('--use_type', type=str, default=None, help='types of NN: use_CNN_A')
    
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

    
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
