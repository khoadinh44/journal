from load_cases import get_data
from FaceNet_predict import FaceNetOneShotRecognitor
from preprocessing.utils import invert_one_hot, load_table_10_spe, recall_m, precision_m, f1_m, to_one_hot, use_denoise, scaler_transform
from network.nn import CNN_C
from src.model import CNN_C_trip
from load_cases import get_data
from train import parse_opt
from arc_face import train_ArcFaceModel
from train_routines.triplet_loss import train
from train_routines.center_loss import train_center_loss
from train_routines.new_center_loss import train_new_center_loss
from train_routines.triplet_center_loss import train_triplet_center_loss
# from train_routines.new_triplet_center import train_new_triplet_center
from train_routines.new_triplet_center_version_2 import train_new_triplet_center
from preprocessing.utils import handcrafted_features, FFT
from preprocessing.denoise_signal import savitzky_golay, Fourier, SVD_denoise, Wavelet_denoise

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.utils import shuffle
from scipy.spatial.distance import cosine, euclidean
import angular_grad
import tensorflow as tf
import glob 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os
from sklearn.metrics import accuracy_score

tf.compat.v1.reset_default_graph()

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

def plot_confusion(y_test, y_pred_inv, outdir, each_ML, opt):
  if opt.case_14:
    commands = [str(i) for i in set(y_test)]
  else:
    commands = ['Healthy', 'OR Damage', 'IR Damage']
  confusion_mtx = tf.math.confusion_matrix(y_test, y_pred_inv)

  plt.figure(figsize=(10, 8))
  sns.heatmap(confusion_mtx,
            xticklabels=commands,
            yticklabels=commands,
            annot=True, fmt='g')
  plt.xlabel('Prediction')
  plt.ylabel('Label')
  plt.savefig(os.path.join(outdir, each_ML))
  plt.show()

def main(opt):
  print(color.GREEN + '\n\n\t *************START*************\n\n' + color.END)
  
  print('\n loading data...')
   # y_train for the PU data
   # y_train_CWRU for the CWRU data in case of 14
  if os.path.exists('/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/y_train_CWRU.npy'):  
    X_train = np.load('/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/X_train_CWRU.npy')
    X_test = np.load('/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/X_test_CWRU.npy')
    y_train = np.load('/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/y_train_CWRU.npy')
    y_test = np.load('/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/y_test_CWRU.npy')
  else:
    X_train, X_test, y_train, y_test = get_data(opt)

    print('\n Converting data...')
    y_train = invert_one_hot(y_train)
    y_test = invert_one_hot(y_test)


    with open('/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/y_train_CWRU.npy', 'wb') as f:
      np.save(f, y_train)
    with open('/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/y_test_CWRU.npy', 'wb') as f:
      np.save(f, y_test)
    with open('/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/X_train_CWRU.npy', 'wb') as f:
      np.save(f, X_train)
    with open('/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/X_test_CWRU.npy', 'wb') as f:
      np.save(f, X_test)

  if opt.use_FFT:
    print('\n Using FFT...')
    X_train_FFT = FFT(X_train)
    X_test_FFT = FFT(X_test)
  else:
    X_train_FFT = X_train
    X_test_FFT = X_test

  if opt.scaler == 'handcrafted_features':
    print('\n Using hand Crafted feature..')
    X_train_FFT = handcrafted_features(X_train)
    X_test_FFT  = handcrafted_features(X_test)
  
  if opt.denoise == 'DFK':
    X_train_FFT = use_denoise(X_train, Fourier)
    X_test_FFT  = use_denoise(X_test, Fourier)
  elif opt.denoise == 'Wavelet_denoise':
    X_train_FFT = use_denoise(X_train, Wavelet_denoise)
    X_test_FFT  = use_denoise(X_test, Wavelet_denoise)
  elif opt.denoise == 'SVD':
    X_train_FFT = use_denoise(X_train, SVD_denoise)
    X_test_FFT  = use_denoise(X_test, SVD_denoise)
  elif opt.denoise == 'savitzky_golay':
    X_train_FFT = X_train = use_denoise(X_train, savitzky_golay)
    X_test_FFT  = X_test = use_denoise(X_test, savitzky_golay)

  if opt.scaler == 'MinMaxScaler':
    X_train_FFT = scaler_transform(X_train, MinMaxScaler)
    X_test_FFT = scaler_transform(X_test, MinMaxScaler)
  elif opt.scaler == 'MaxAbsScaler':
    X_train_FFT = scaler_transform(X_train, MaxAbsScaler)
    X_test_FFT = scaler_transform(X_test, MaxAbsScaler)
  elif opt.scaler == 'StandardScaler':
    X_train_FFT = scaler_transform(X_train, StandardScaler)
    X_test_FFT = scaler_transform(X_test, StandardScaler)
  elif opt.scaler == 'RobustScaler':
    X_train_FFT = scaler_transform(X_train, RobustScaler)
    X_test_FFT = scaler_transform(X_test, RobustScaler)
  elif opt.scaler == 'Normalizer':
    X_train_FFT = scaler_transform(X_train, Normalizer)
    X_test_FFT = scaler_transform(X_test, Normalizer)
  elif opt.scaler == 'QuantileTransformer':
    X_train_FFT = scaler_transform(X_train, QuantileTransformer)
    X_test_FFT = scaler_transform(X_test, QuantileTransformer)
  elif opt.scaler == 'PowerTransformer':
    X_train_FFT = scaler_transform(X_train, PowerTransformer)
    X_test_FFT = scaler_transform(X_test, PowerTransformer)
#     X_train_FFT= np.load('/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/X_train_FFT.npy')
#     X_test_FFT = np.load('/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/X_test_FFT.npy')

  print(f' Training data shape: {X_train_FFT.shape},  Training label shape: {y_train.shape}')
  print(f' Testing data shape: {X_test_FFT.shape},   Testing label shape: {y_test.shape}')
  
  # # Save data ----------------------------------------------
  # with open('/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/X_train_FFT.npy', 'wb') as f:
  #   np.save(f, X_train_FFT)
  # with open('/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/X_test_FFT.npy', 'wb') as f:
  #   np.save(f, X_test_FFT)

  # Convert label 2 to label 1. Train with 30 epochs
  # y_train = np.where(y_train!=2, y_train, 1)

  print('\n Loading model...')
  if opt.embedding_model == 'triplet':
    train_embs, test_embs, y_test_solf, y_train, outdir = train(opt, X_train_FFT, y_train, X_test_FFT, y_test, CNN_C_trip) 
  if opt.embedding_model == 'center': 
    train_embs, test_embs, y_test_solf, y_train, outdir = train_center_loss(opt, X_train_FFT, y_train, X_test_FFT, y_test, CNN_C_trip) 
  if opt.embedding_model == 'new_center': 
    train_embs, test_embs, y_test_solf, y_train, outdir = train_new_center_loss(opt, X_train_FFT, X_train, y_train, X_test_FFT, X_test, y_test, CNN_C_trip) 
  if opt.embedding_model == 'triplet_center':
    train_embs, test_embs, y_test_solf, y_train, outdir = train_triplet_center_loss(opt, X_train_FFT, y_train, X_test_FFT, y_test, CNN_C_trip) 
  if opt.embedding_model == 'new_triplet_center':
    train_embs, test_embs, y_test_solf, y_train, outdir = train_new_triplet_center(opt, X_train_FFT, X_train, y_train, X_test_FFT, X_test, y_test, CNN_C_trip)
  if opt.embedding_model == 'arcface':
    train_embs, test_embs, y_train, outdir = train_ArcFaceModel(opt, X_train_FFT, y_train, X_test_FFT, y_test)
  
  

  print('\n Saving embedding phase...')   
  this_acc = []
  y_pred_Lo_Co = []
  y_pred_SVM_Ran = []

  if opt.embedding_model != 'arcface':
    y_test_solf = np.argmax(y_test_solf, axis=1)
    solf_acc = accuracy_score(y_test, y_test_solf)
    confusion_mtx = tf.math.confusion_matrix(y_test, y_test_solf)
    plot_confusion(y_test, y_test_solf, outdir, 'softmax', opt)
    
    print(f'\n-------------- Test accuracy: {solf_acc} with the solfmax method--------------')

  y_pred_all = []
  y_pred_SVM_RandomForestClassifier = []
  y_pred_KNN_RandomForestClassifier = []

  count1 = 0
  count2 = 0
  count3 = 0
  for each_ML in ['SVM', 'RandomForestClassifier', 'LogisticRegression', 'GaussianNB', 'KNN', 'BT', 'euclidean', 'cosine']:
    model = FaceNetOneShotRecognitor(opt, X_train, y_train, X_test, y_test) 
    y_pred = model.predict(test_embs=test_embs, train_embs=train_embs, ML_method=each_ML)
    y_pred_inv = np.argmax(y_pred, axis=1)
    
    plot_confusion(y_test, y_pred_inv, outdir, each_ML, opt)
    acc = accuracy_score(y_test, y_pred_inv)
    
    if each_ML not in ['euclidean', 'cosine']:
       if y_pred_all == []:
         y_pred_all = y_pred
       else:
         y_pred_all += y_pred
       count1 += 1

    if each_ML in ['SVM', 'RandomForestClassifier', 'KNN']:
      if y_pred_SVM_RandomForestClassifier == []:
        y_pred_SVM_RandomForestClassifier = y_pred
      else:
        y_pred_SVM_RandomForestClassifier += y_pred
      count2 += 1
    
    if each_ML in ['KNN', 'RandomForestClassifier']:
      if y_pred_KNN_RandomForestClassifier == []:
        y_pred_KNN_RandomForestClassifier = y_pred
      else:
        y_pred_KNN_RandomForestClassifier += y_pred
      count3 += 1

    print(f'\n-------------- Test accuracy: {acc} with the {each_ML} method--------------')
    
    if each_ML in ['euclidean', 'cosine']:
       y_pred = model.predict(test_embs=test_embs, train_embs=train_embs, ML_method=each_ML, use_mean=False)
       y_pred_inv = np.argmax(y_pred, axis=1)

       plot_confusion(y_test, y_pred_inv, outdir, each_ML+'no_mean', opt)
       acc = accuracy_score(y_test, y_pred_inv)
       print(f'\n-------------- Test accuracy: {acc} with the {each_ML} no mean method--------------')
    

    # X_train_hand = handcrafted_features(X_train)
    # X_test_hand  = handcrafted_features(X_test)
    # model = FaceNetOneShotRecognitor(opt, X_train_hand, y_train, X_test_hand, y_test) 
    # y_pred_no_emb = model.predict(test_embs=test_embs, train_embs=train_embs, ML_method=each_ML, emb=False)
    
    # y_pred_all += y_pred_no_emb
    # count1 += 1
    # acc = accuracy_score(y_test, np.argmax(y_pred_no_emb, axis=1))

    # print(f'\n-------------- 2.Test accuracy: {acc} with the {each_ML} method--------------')
  
  y_pred_SVM_RandomForestClassifier = y_pred_SVM_RandomForestClassifier.astype(np.float32) / count2
  y_pred_SVM_RandomForestClassifier = np.argmax(y_pred_SVM_RandomForestClassifier, axis=1)
  acc_case_1 = accuracy_score(y_test, y_pred_SVM_RandomForestClassifier)
  print(f'\n--------------Ensemble for SVM vs RandomForestClassifier vs KNN: {acc_case_1}--------------')

  y_pred_KNN_RandomForestClassifier = y_pred_KNN_RandomForestClassifier.astype(np.float32) / count3
  y_pred_KNN_RandomForestClassifier = np.argmax(y_pred_KNN_RandomForestClassifier, axis=1)
  acc_case_2 = accuracy_score(y_test, y_pred_KNN_RandomForestClassifier)
  print(f'\n--------------Ensemble for BT vs RandomForestClassifier vs cosine: {acc_case_2}--------------')

  y_pred_all = y_pred_all.astype(np.float32) / count1
  y_pred_all = np.argmax(y_pred_all, axis=1)
  acc_all = accuracy_score(y_test, y_pred_all)
  print(f'\n--------------Ensemble for all: {acc_all}--------------')

  # with open('/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/ensemble.npy', 'wb') as f:
  #   np.save(f, y_pred_all)
  # f = open('/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/ensemble.txt', 'a') 
  # for i in list(y_pred_all):
  #   i = str(i)
  #   f.write(f"{i}, ")
  # f.close()

if __name__ == '__main__':
  opt = parse_opt()
  # if os.path.exists(opt.outdir + "triplet_loss/triplet_loss_model.h5"):
  #   os.remove(opt.outdir + "triplet_loss/triplet_loss_model.h5")
  main(opt)
