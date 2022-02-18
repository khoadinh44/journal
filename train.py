import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import accuracy_score
from preprocessing.utils import handcrafted_features
import argparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC

from network.nn import DNN_A, DNN_B, CNN_A, CNN_B, CNN_C 
from network.enssemble import semble_transfer
from network.wavenet import  WaveNet
from preprocessing.utils import recall_m, precision_m, f1_m, signaltonoise_dB, use_denoise, add_noise, scaler, invert_one_hot
from preprocessing.denoise_signal import Fourier
from load_cases import get_data
from preprocessing.denoise_signal import savitzky_golay, Fourier, SVD_denoise, Wavelet_denoise

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
    model = network(opt)
    model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['acc', f1_m, precision_m, recall_m]) # loss='mse'

    model.summary()
    # history = model.fit(data, labels,
    #                     epochs     = opt.epochs,
    #                     batch_size = opt.batch_size,
    #                     validation_data=(val_data, val_labels))

    if opt.use_DNN_A:
      model.save(opt.save + opt.model_names[0] )
      with open(f'/content/drive/Shareddrives/newpro112233/signal_machine/{folder}/DNN_A_history', 'wb') as file_pi:
  #     with open('DNN_A_history', 'wb') as file_pi: 
        pickle.dump(history.history, file_pi)
    elif opt.use_DNN_B:
      with open(f'/content/drive/Shareddrives/newpro112233/signal_machine/{folder}/DNN_B_history', 'wb') as file_pi:
  #     with open('DNN_B_history', 'wb') as file_pi: 
        pickle.dump(history.history, file_pi)
    elif opt.use_CNN_A:
      model.save(opt.save + opt.model_names[1])
      with open(f'/content/drive/Shareddrives/newpro112233/signal_machine/{folder}/CNN_A_history', 'wb') as file_pi:
  #     with open('CNN_A_history', 'wb') as file_pi: 
        pickle.dump(history.history, file_pi)
    elif opt.use_CNN_B:
      with open(f'/content/drive/Shareddrives/newpro112233/signal_machine/{folder}/CNN_B_history', 'wb') as file_pi:
  #     with open('CNN_B_history', 'wb') as file_pi: 
        pickle.dump(history.history, file_pi)
    elif opt.use_CNN_C:
      # model.save(opt.save + opt.model_names[2])
      model.load_weights(opt.save + opt.model_names[2])
      # with open(f'/content/drive/Shareddrives/newpro112233/signal_machine/{folder}/CNN_C_history', 'wb') as file_pi:
  #     with open('CNN_C_history', 'wb') as file_pi: 
        # pickle.dump(history.history, file_pi)

    for i in range(len(opt.SNRdb)):
      test = add_noise(test_data, opt.SNRdb[i])
      test = use_denoise(test, Fourier)
      _, test_acc,  test_f1_m,  test_precision_m,  test_recall_m  = model.evaluate(test, test_labels, verbose=0)
      print(f'Score in test set in {opt.SNRdb[i]}dB: \n Accuracy: {test_acc}, F1: {test_f1_m}, Precision: {test_precision_m}, recall: {test_recall_m}' )

  
    
def main(opt):
  callback = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)]
  tf.get_logger().setLevel('ERROR')
          
  with tf.device('/CPU:0'):
    X_train_all, X_test, y_train_all, y_test = get_data(opt)

  # Denoising methods ###################################################################################
  if opt.denoise == 'DFK':
    folder = 'Fourier'
    print('Using DFK \n')
    X_train_all = use_denoise(X_train_all, Fourier)
    X_test = use_denoise(X_test, Fourier)
  elif opt.denoise == 'Wavelet_denoise':
    folder = 'Wavelet_denoise'
    print('Using Wavelet_denoise \n')
    X_train_all = use_denoise(X_train_all, Wavelet_denoise)
    X_test = use_denoise(X_test, Wavelet_denoise)
  elif opt.denoise == 'SVD':
    folder = 'SVD'
    print('Using SVD \n')
    X_train_all = use_denoise(X_train_all, SVD_denoise)
    X_test = use_denoise(X_test, SVD_denoise)
  elif opt.denoise == 'savitzky_golay':
    folder = 'savitzky_golay' 
    print('Using savitzky_golay \n')
    X_train_all = use_denoise(X_train_all, savitzky_golay)
    X_test = use_denoise(X_test, savitzky_golay)
  else:
    print('none_denoise \n')
    folder = 'none_denoise' 
  
  # Normalizing methods ################################################################################
  if opt.scaler == 'MinMaxScaler':
    print('Using MinMaxScaler')
    X_train_all = scaler(X_train_all, MinMaxScaler)
    X_test = scaler(X_test, MinMaxScaler)
  elif opt.scaler == 'MaxAbsScaler':
    print('Using MaxAbsScaler')
    X_train_all = scaler(X_train_all, MaxAbsScaler)
    X_test = scaler(X_test, MaxAbsScaler)
  elif opt.scaler == 'StandardScaler':
    print('Using StandardScaler')
    X_train_all = scaler(X_train_all, StandardScaler)
    X_test = scaler(X_test, StandardScaler)
  elif opt.scaler == 'RobustScaler':
    print('Using RobustScaler')
    X_train_all = scaler(X_train_all, RobustScaler)
    X_test = scaler(X_test, RobustScaler)
  elif opt.scaler == 'Normalizer':
    print('Using Normalizer')
    X_train_all = scaler(X_train_all, Normalizer)
    X_test = scaler(X_test, Normalizer)
  elif opt.scaler == 'QuantileTransformer':
    print('Using QuantileTransformer')
    X_train_all = scaler(X_train_all, QuantileTransformer)
    X_test = scaler(X_test, QuantileTransformer)
  elif opt.scaler == 'PowerTransformer':
    print('Using PowerTransformer')
    X_train_all = scaler(X_train_all, PowerTransformer)
    X_test = scaler(X_test, PowerTransformer)
  elif opt.scaler == 'handcrafted_features':
    print('Using handcrafted_features')
    X_train_all = handcrafted_features(np.squeeze(X_train_all))
    X_test = handcrafted_features(np.squeeze(X_test))
  
  X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.2, random_state=42, shuffle=True)

  # Machine learning models #####################################################################################################
  if opt.ML_method != None:
    if opt.ML_method == 'SVM':
      model = SVC(kernel='rbf', probability=True)
    elif opt.ML_method == 'RandomForestClassifier':
      model = RandomForestClassifier(n_estimators= 300, max_features = "sqrt", n_jobs = -1, random_state = 38)
    elif opt.ML_method == 'LogisticRegression':     
      model = LogisticRegression(random_state=1)
    elif opt.ML_method == 'GaussianNB':
      model = GaussianNB()
    # Train the model
    X_train_all = np.squeeze(X_train_all)
    y_train_all = invert_one_hot(y_train_all)

    X_test = np.squeeze(X_test)
    y_test = invert_one_hot(y_test)

    model.fit(X_train_all, y_train_all)
    
    test_predictions = model.predict(X_test)
    print("Test accuracy:", accuracy_score(y_test, test_predictions))
  
  # Deep learning models ########################################################################################################
  elif opt.use_DNN_A:
    train(X_train, y_train, X_val, y_val, X_test, y_test, DNN_A, folder, opt)
  elif opt.use_CNN_A:
    train(X_train, y_train, X_val, y_val, X_test, y_test, CNN_A, folder, opt)
  elif opt.use_CNN_C:
    train(X_train, y_train, X_val, y_val, X_test, y_test, CNN_C, folder, opt)
  elif opt.use_wavenet:
    train(X_train, y_train, X_val, y_val, X_test, y_test, WaveNet, folder, opt)
  elif opt.use_wavenet_head:
    train(X_train, y_train, X_val, y_val, X_test, y_test, WaveNet_Head, folder, opt)
  elif opt.ensemble:
    semble_transfer(opt, X_test, y_test)
  
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    
    # Models and denoising methods--------------------------
    parser.add_argument('--ML_method',      default=None, type=str)
    parser.add_argument('--use_DNN_A',   default=False, type=bool)
    parser.add_argument('--use_DNN_B',   default=False, type=bool)
    parser.add_argument('--use_CNN_A',   default=False, type=bool)
    parser.add_argument('--use_CNN_B',   default=False, type=bool)
    parser.add_argument('--use_CNN_C',   default=False, type=bool)
    parser.add_argument('--use_wavenet',      default=False, type=bool)
    parser.add_argument('--use_wavenet_head',      default=False, type=bool)
    parser.add_argument('--ensemble',      default=False, type=bool)
    parser.add_argument('--denoise', type=str, default=None, help='types of NN: DFK, Wavelet_denoise, SVD, savitzky_golay, None. DFK is our proposal.')
    parser.add_argument('--scaler',  type=str, default=None, help='handcrafted_features, MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler, Normalizer, QuantileTransformer, PowerTransformer')
    
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

    parser.add_argument('--SNRdb',                    type=str,     default=[0, 5, 10, 15, 20, 25, 30],         help='intensity of noise')
    parser.add_argument('--num_mels',                 type=int,     default=80,          help='num_mels')
    parser.add_argument('--upsample_scales',          type=str,     default=[4, 8, 8],   help='num_mels')
    parser.add_argument('--model_names',              type=str,     default=['DNN', 'CNN_A', 'CNN_C'],   help='name of all NN models')
    parser.add_argument('--exponential_decay_steps',  type=int,     default=200000,      help='exponential_decay_steps')
    parser.add_argument('--exponential_decay_rate',   type=float,   default=0.5,         help='exponential_decay_rate')
    parser.add_argument('--beta_1',                   type=float,   default=0.9,         help='beta_1')
    parser.add_argument('--result_dir',               type=str,     default="./result/", help='exponential_decay_rate')
    parser.add_argument('--model_dir',                type=str,     default="/content/drive/Shareddrives/newpro112233/signal_machine/", help='direction to save model')
    parser.add_argument('--load_path',                type=str,      default=None,        help='path weight')

          
          
    
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

    
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
