import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import keras
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import argparse

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from tensorflow.keras.models import load_model

from network.nn import DNN_A, DNN_B, CNN_A, CNN_B, CNN_C
from network.resnet import resnet18, resnet34
from network.enssemble import semble_transfer
from network.wavenet import  WaveNet
from preprocessing.utils import recall_m, precision_m, f1_m, use_denoise, add_noise, scaler, invert_one_hot, convert_spectrogram
from preprocessing.denoise_signal import Fourier
from preprocessing.utils import handcrafted_features
from load_cases import get_data
from preprocessing.denoise_signal import savitzky_golay, Fourier, SVD_denoise, Wavelet_denoise

# Can use K-fold: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)
# callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)
# callback = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=1)
          
def train(data, labels,
          val_data, val_labels,
          test_data, test_labels,
          network, folder, opt):
  
  if opt.use_CNN_A:
    data = np.expand_dims(data, axis=-1)
    val_data = np.expand_dims(val_data, axis=-1)
    model = network(opt)
  elif opt.use_CNN_B:
    inputs = keras.Input(shape=(354, 354, 1))
    outputs = network(inputs)
    model = keras.Model(inputs, outputs)
  else:
    model = network(opt)
  model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['acc', f1_m, precision_m, recall_m]) # loss='mse'

  model.summary()
  history = model.fit(data, labels,
                      epochs     = opt.epochs,
                      batch_size = opt.batch_size,
                      validation_data=(val_data, val_labels),)
                      # callbacks=[callback])

  if opt.use_DNN_A:
    model.save(opt.save + opt.model_names[0])
    with open(f'/content/drive/Shareddrives/newpro112233/signal_machine/{folder}/DNN_A_history', 'wb') as file_pi:
#     with open('DNN_A_history', 'wb') as file_pi: 
      pickle.dump(history.history, file_pi)
  elif opt.use_CNN_A:
    model.save(opt.save + opt.model_names[1])
    with open(f'/content/drive/Shareddrives/newpro112233/signal_machine/{folder}/CNN_A_history', 'wb') as file_pi:
#     with open('CNN_A_history', 'wb') as file_pi: 
      pickle.dump(history.history, file_pi)
  elif opt.use_CNN_B:
    model.save(opt.save + opt.model_names[2])
    with open(f'/content/drive/Shareddrives/newpro112233/signal_machine/{folder}/CNN_B_history', 'wb') as file_pi:
#     with open('CNN_A_history', 'wb') as file_pi: 
      pickle.dump(history.history, file_pi)
  elif opt.use_CNN_C:
    model.save(opt.save + opt.model_names[3])
    with open(f'/content/drive/Shareddrives/newpro112233/signal_machine/{folder}/CNN_C_history', 'wb') as file_pi:
    # with open('CNN_C_history', 'wb') as file_pi: 
      pickle.dump(history.history, file_pi)
  elif opt.use_wavenet:
    model.save(opt.save + opt.model_names[4])
    with open(f'/content/drive/Shareddrives/newpro112233/signal_machine/{folder}/wavenet_history', 'wb') as file_pi:
    # with open('CNN_C_history', 'wb') as file_pi: 
      pickle.dump(history.history, file_pi)
  elif opt.use_wavenet_head:
    model.save(opt.save + opt.model_names[5])
    with open(f'/content/drive/Shareddrives/newpro112233/signal_machine/{folder}/wavenet_head_history', 'wb') as file_pi:
    # with open('CNN_C_history', 'wb') as file_pi: 
      pickle.dump(history.history, file_pi)

  if opt.use_SNRdb: 
    for i in range(len(opt.SNRdb)):
      test_data = np.squeeze(test_data)
      test = add_noise(test_data, opt.SNRdb[i])
      test = np.expand_dims(test, axis=-1)
      print('\n----------------Adding noise Phase -----------------------')
      _, test_acc,  test_f1_m,  test_precision_m,  test_recall_m  = model.evaluate(test, test_labels, verbose=0)
      print(f'Score in test set in {opt.SNRdb[i]}dB: \n Accuracy: {test_acc}, F1: {test_f1_m}, Precision: {test_precision_m}, recall: {test_recall_m}' )
  else:
    _, test_acc,  test_f1_m,  test_precision_m,  test_recall_m  = model.evaluate(test_data, test_labels, verbose=0)
    print(f'Score in test set: \n Accuracy: {test_acc}, F1: {test_f1_m}, Precision: {test_precision_m}, recall: {test_recall_m}' )

  
    
def main(opt):
  tf.get_logger().setLevel('ERROR')
          
  X_train_all, X_test, y_train_all, y_test = get_data(opt)
                    
  if opt.use_CNN_B:
    X_train_all = convert_spectrogram(X_train_all)/255.
    X_test = convert_spectrogram(X_test)/255.

  print('Shape of training data:', X_train_all.shape)
  print('Shape of testing data:', X_test.shape)

  # Denoising methods ###################################################################################
  if opt.denoise == 'DFK':
    folder = 'Fourier'
    print('\n------------------Using DFK------------------')
    X_train_all = use_denoise(X_train_all, Fourier)
    X_test = use_denoise(X_test, Fourier)
  elif opt.denoise == 'Wavelet_denoise':
    folder = 'Wavelet_denoise'
    print('\n------------------Using Wavelet_denoise------------------')
    X_train_all = use_denoise(X_train_all, Wavelet_denoise)
    X_test = use_denoise(X_test, Wavelet_denoise)
  elif opt.denoise == 'SVD':
    folder = 'SVD'
    print('\n------------------Using SVD------------------')
    X_train_all = use_denoise(X_train_all, SVD_denoise)
    X_test = use_denoise(X_test, SVD_denoise)
  elif opt.denoise == 'savitzky_golay':
    print('\n------------------Using savitzky_golay------------------')
    X_train_all = use_denoise(X_train_all, savitzky_golay)
    X_test = use_denoise(X_test, savitzky_golay)
  else:
    print('\n------------------none_denoise------------------')
    folder = 'none_denoise' 
  
  # Normalizing methods ################################################################################
  if opt.scaler != None:
    X_train_all = np.squeeze(X_train_all)
    X_test      = np.squeeze(X_test)

    if opt.scaler == 'MinMaxScaler':
      print('\n------------------MinMaxScaler------------------')
      X_train_all, scale = scaler(X_train_all, MinMaxScaler)
      X_test = scale.transform(X_test)
    elif opt.scaler == 'MaxAbsScaler':
      print('\n------------------MaxAbsScaler------------------')
      X_train_all, scale = scaler(X_train_all, MaxAbsScaler)
      X_test = scale.transform(X_test)
    elif opt.scaler == 'StandardScaler':
      print('\n------------------StandardScaler------------------')
      X_train_all, scale = scaler(X_train_all, StandardScaler)
      X_test = scale.transform(X_test)
    elif opt.scaler == 'RobustScaler':
      print('\n------------------RobustScaler------------------')
      X_train_all, scale = scaler(X_train_all, RobustScaler)
      X_test = scale.transform(X_test)
    elif opt.scaler == 'Normalizer':
      print('\n------------------Normalizer------------------')
      X_train_all, scale = scaler(X_train_all, Normalizer)
      X_test = scale.transform(X_test)
    elif opt.scaler == 'QuantileTransformer':
      print('\n------------------QuantileTransformer------------------')
      X_train_all, scale = scaler(X_train_all, QuantileTransformer)
      X_test = scale.transform(X_test)
    elif opt.scaler == 'PowerTransformer':
      print('\n------------------PowerTransformer------------------')
      X_train_all, scale = scaler(X_train_all, PowerTransformer)
      X_test = scale.transform(X_test)
    elif opt.scaler == 'handcrafted_features':
      print('\n------------------handcrafted_features------------------')
      X_train_all = handcrafted_features(X_train_all)
      X_test = handcrafted_features(X_test)
    else:
      print('\n------------------none scaler------------------')
  
#   X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=opt.test_rate, random_state=42, shuffle=True)
  X_train, X_val, y_train, y_val = X_train_all, X_test, y_train_all, y_test

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
  elif opt.use_CNN_B:
    train(X_train, y_train, X_val, y_val, X_test, y_test, resnet18, folder, opt)
  elif opt.use_CNN_C:
    train(X_train, y_train, X_val, y_val, X_test, y_test, CNN_C, folder, opt)
  elif opt.use_wavenet:
    train(X_train, y_train, X_val, y_val, X_test, y_test, WaveNet, folder, opt)
  elif opt.use_wavenet_head:
    train(X_train, y_train, X_val, y_val, X_test, y_test, WaveNet_Head, folder, opt)
  elif opt.ensemble:
    semble_transfer(opt, X_test, y_test, X_train, y_train)
  
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
   
    parser.add_argument('--outdir',   default='/content/drive/Shareddrives/newpro112233/signal_machine/runs/', help="Directory containing the Checkpoints")
    parser.add_argument('--threshold',  default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9], type=str, help='num_mels')
    parser.add_argument('--faceNet',          default=True, type=bool)
    parser.add_argument('--Use_euclidean',    default=False, type=bool)

    # Models and denoising methods--------------------------
    parser.add_argument('--ML_method',   default='RandomForestClassifier', type=str)
    parser.add_argument('--use_DNN_A',   default=False, type=bool)
    parser.add_argument('--use_DNN_B',   default=False, type=bool)
    parser.add_argument('--use_CNN_A',   default=False, type=bool)
    parser.add_argument('--use_CNN_B',   default=False, type=bool)
    parser.add_argument('--use_CNN_C',   default=False, type=bool)
    parser.add_argument('--use_wavenet',       default=False, type=bool)
    parser.add_argument('--use_wavenet_head',  default=False, type=bool)
    parser.add_argument('--ensemble',          default=False, type=bool)
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
    
    parser.add_argument('--PU_data_table_10',            default=False, type=bool)
    parser.add_argument('--PU_data_table_10_case_0',     default=False, type=bool)
    parser.add_argument('--PU_data_table_10_case_1',     default=False, type=bool)
    parser.add_argument('--PU_data_table_8',      default=True, type=bool)
    parser.add_argument('--MFPT_data',            default=False, type=bool)
    parser.add_argument('--data_normal',          default=False, type=bool)
    parser.add_argument('--data_12k',             default=False, type=bool)
    parser.add_argument('--data_48k',             default=False, type=bool)
    parser.add_argument('--multi_head',           default=False, type=bool)

    # Parameters---------------------------------------------
    parser.add_argument('--epoch',              type=int,   default=16, help="Number epochs to train the model for")
    parser.add_argument('--save',               type=str,   default='/content/drive/Shareddrives/newpro112233/signal_machine/', help='Position to save weights')
    parser.add_argument('--num_classes',        type=int,   default=3,          help='3 Number of classes in faceNet')
    parser.add_argument('--embedding_size',     type=int,   default=128,        help='128 Number of embedding in faceNet')
    parser.add_argument('--input_shape',        type=int,   default=250604,       help='127950 or 255900 in 5-fold or table 8: 6270 for handcrafted, 250604 in the only training.')
    parser.add_argument('--batch_size',         type=int,   default=32,         help='80 for arc_face, 32 for others')
    parser.add_argument('--test_rate',          type=float, default=0.2,        help='rate of split data for testing')
    parser.add_argument('--learning_rate',      type=float, default=0.001,      help='learning rate')

    parser.add_argument('--use_FFT',                  type=bool,    default=False)
    parser.add_argument('--use_SNRdb',                type=bool,    default=False)
    parser.add_argument('--use_weight',               type=bool,    default=False)
    parser.add_argument('--lambda_',                  type=float,   default=0.5,         help='lambda_')
    parser.add_argument('--SNRdb',                    type=str,     default=[0, 5, 10, 15, 20, 25, 30],         help='intensity of noise')
    parser.add_argument('--num_mels',                 type=int,     default=32,          help='num_mels')
    parser.add_argument('--upsample_scales',          type=str,     default=[4, 8, 8],   help='num_mels')
    parser.add_argument('--model_names',              type=str,     default=['DNN', 'CNN_A', 'CNN_B', 'CNN_C', 'wavenet', 'wavelet_head'],   help='name of all NN models')
    parser.add_argument('--embedding_model',          type=str,     default='arcface',   help='new_triplet_loss, new_triplet_loss, triplet, center, new_center, triplet_center')
    parser.add_argument('--type_PU_data',             type=str,     default='MCS2', help='vibration, MCS1, MCS2')
    parser.add_argument('--activation',               type=str,     default='relu',      help='softmax, sigmoid, softplus, softsign, tanh, selu, elu, exponential')
    parser.add_argument('--exponential_decay_steps',  type=int,     default=200000,      help='exponential_decay_steps')
    parser.add_argument('--exponential_decay_rate',   type=float,   default=0.5,         help='exponential_decay_rate')
    parser.add_argument('--beta_1',                   type=float,   default=0.9,         help='beta_1')
    parser.add_argument('--result_dir',               type=str,     default="./result/", help='exponential_decay_rate')
    parser.add_argument('--model_dir',                type=str,     default="/content/drive/Shareddrives/newpro112233/signal_machine/", help='direction to save model')
    parser.add_argument('--load_path',                type=str,     default=None,        help='path weight')
    
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

    
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
