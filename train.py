import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import argparse

from network.nn import DNN_A, DNN_B, CNN_A, CNN_B, CNN_C
from network.wavenet import  WaveNet
from preprocessing.utils import recall_m, precision_m, f1_m, signaltonoise_dB, use_denoise
from preprocessing.denoise_signal import Fourier
from ML_methods import get_data

gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)

def train(data, labels,
          val_data, val_labels,
          test_data, test_labels,
          network, folder, opt):
  
  if opt.use_CNN_A:
    data = np.expand_dims(data, axis=-1)
    val_data = np.expand_dims(val_data, axis=-1)

  with strategy.scope():
    model = network(opt.num_classes, opt)
    model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['acc', f1_m, precision_m, recall_m]) # loss='mse'
    model.summary()

    history = model.fit(data, labels,
                        epochs     = opt.epochs,
                        batch_size = opt.batch_size,
                        validation_data=(val_data, val_labels))

  _, test_acc,  test_f1_m,  test_precision_m,  test_recall_m  = model.evaluate(test_data, test_labels, verbose=0)
  print(f'Score in test set: \n Accuracy: {test_acc}, F1: {test_f1_m}, Precision: {test_precision_m}, recall: {test_recall_m}' )

  if opt.use_DNN_A:
    model.save(opt.save+'DNN_A.hdf5')
    with open(f'/content/drive/Shareddrives/newpro112233/signal_machine/{folder}/DNN_A_history', 'wb') as file_pi:
#     with open('DNN_A_history', 'wb') as file_pi: 
      pickle.dump(history.history, file_pi)
  elif opt.use_DNN_B:
    with open(f'/content/drive/Shareddrives/newpro112233/signal_machine/{folder}/DNN_B_history', 'wb') as file_pi:
#     with open('DNN_B_history', 'wb') as file_pi: 
      pickle.dump(history.history, file_pi)
  elif opt.use_CNN_A:
    model.save(opt.save+'CNN_A.hdf5')
    with open(f'/content/drive/Shareddrives/newpro112233/signal_machine/{folder}/CNN_A_history', 'wb') as file_pi:
#     with open('CNN_A_history', 'wb') as file_pi: 
      pickle.dump(history.history, file_pi)
  elif opt.use_CNN_B:
    with open(f'/content/drive/Shareddrives/newpro112233/signal_machine/{folder}/CNN_B_history', 'wb') as file_pi:
#     with open('CNN_B_history', 'wb') as file_pi: 
      pickle.dump(history.history, file_pi)
  elif opt.use_CNN_C:
    model.save(opt.save+'CNN_C.hdf5')
    with open(f'/content/drive/Shareddrives/newpro112233/signal_machine/{folder}/CNN_C_history', 'wb') as file_pi:
#     with open('CNN_C_history', 'wb') as file_pi: 
      pickle.dump(history.history, file_pi)
  

    
def main(opt):
  callback = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)]
  tf.get_logger().setLevel('ERROR')
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
  with tf.device('/CPU:0'):
    X_train_all, X_test, y_train_all, y_test = get_data(opt)
    with strategy.scope():
      if opt.denoise == 'DFK':
        X_train_all = use_denoise(X_train_all, Fourier)
        X_test = use_denoise(X_test, Fourier)
    X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.1, random_state=42, shuffle=True)

  
  if opt.use_DNN_A:
    train(X_train, y_train, X_val, y_val, X_test, y_test, DNN_A, folder, opt)
  elif opt.use_DNN_B:
    train((X_train_A, X_train_B), y_train, (X_test_A, X_test_B), y_test, DNN_B, folder, opt)
  elif opt.use_CNN_A:
    train(X_train, y_train, X_val, y_val, X_test, y_test, CNN_A, folder, opt)
  elif opt.use_CNN_B:
    train(X_train, y_train, X_val, y_val, X_test, y_test, CNN_B, folder, opt)
  elif opt.use_CNN_C:
    train(X_train, y_train, X_val, y_val, X_test, y_test, CNN_C, folder, opt)
  elif opt.use_wavenet:
    train(X_train, y_train, X_val, y_val, X_test, y_test, WaveNet, folder, opt)
  elif opt.use_wavenet_head:
    train(X_train, y_train, X_val, y_val, X_test, y_test, WaveNet_Head, folder, opt)
    
  
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    
    # Models and denoising methods--------------------------
    parser.add_argument('--use_ML',      default=False, type=bool)
    parser.add_argument('--use_DNN_A',   default=False, type=bool)
    parser.add_argument('--use_DNN_B',   default=False, type=bool)
    parser.add_argument('--use_CNN_A',   default=False, type=bool)
    parser.add_argument('--use_CNN_B',   default=False, type=bool)
    parser.add_argument('--use_CNN_C',   default=False, type=bool)
    parser.add_argument('--use_wavenet',      default=False, type=bool)
    parser.add_argument('--denoise', type=str, default=None, help='types of NN: DFK, Wavelet_denoise, SVD, savitzky_golay, None. DFK is our proposal.')
    
    # Run case------------------------------------------------
    parser.add_argument('--case_0_6',  default=False,  type=bool)
    parser.add_argument('--case_1_7',  default=False,  type=bool)
    parser.add_argument('--case_2_8',  default=False,  type=bool)
    parser.add_argument('--case_3_9',  default=False,  type=bool)
    parser.add_argument('--case_4_10', default=False,  type=bool) # Turn on all cases before
    parser.add_argument('--case_5_11', default=False, type=bool)
    
    parser.add_argument('--case_12', default=False, type=bool) # turn on case_4_10
    parser.add_argument('--case_13', default=False,  type=bool)  # turn on case_5_11
    parser.add_argument('--case_14', default=False,  type=bool)  # turn on case 12 and case_4_11
    parser.add_argument('--case_15', default=False,  type=bool)  # turn on case 12 and case_4_11

    parser.add_argument('--data_normal', default=True, type=bool)
    parser.add_argument('--data_12k', default=False, type=bool)
    parser.add_argument('--data_48k', default=False, type=bool)
    parser.add_argument('--multi_head', default=False, type=bool)

    # Parameters---------------------------------------------
    parser.add_argument('--save',            type=str,   default='/content/drive/Shareddrives/newpro112233/signal_machine/', help='Position to save weights')
    parser.add_argument('--epochs',          type=int,   default=100,        help='Number of iterations for training')
    parser.add_argument('--num_classes',     type=int,   default=64,         help='Number of classes')
    parser.add_argument('--batch_size',      type=int,   default=32,         help='Number of batch size for training')
    parser.add_argument('--test_rate',       type=float, default=0.2,        help='rate of split data for testing')
    parser.add_argument('--learning_rate',   type=float, default=0.001,      help='learning rate')

    parser.add_argument('--num_mels',                 type=int,     default=80,          help='num_mels')
    parser.add_argument('--upsample_scales',          type=str,     default=[4, 8, 8],   help='num_mels')
    parser.add_argument('--model_names',              type=str,     default=['DNN', 'CNN_A', 'CNN_C', 'WaveNet', 'WaveNet_Multi_head'],   help='name of all NN models')
    parser.add_argument('--exponential_decay_steps',  type=int,     default=200000,      help='exponential_decay_steps')
    parser.add_argument('--exponential_decay_rate',   type=float,   default=0.5,         help='exponential_decay_rate')
    parser.add_argument('--beta_1',                   type=float,   default=0.9,         help='beta_1')
    parser.add_argument('--result_dir',               type=str,     default="./result/", help='exponential_decay_rate')
    parser.add_argument('--model_dir',               type=str,     default="/content/drive/Shareddrives/newpro112233/signal_machine/", help='direction to save model')
    parser.add_argument('--load_path',                type=str,      default=None,        help='path weight')

          
          
    
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

    
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
