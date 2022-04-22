from load_cases import get_data
from FaceNet_predict import FaceNetOneShotRecognitor
from preprocessing.utils import invert_one_hot, load_table_10_spe, recall_m, precision_m, f1_m, to_one_hot
from network.nn import CNN_C
from src.model import CNN_C_trip
from load_cases import get_data
from train import parse_opt
from arc_face import train_ArcFaceModel
from train_routines.triplet_loss import train
from train_routines.center_loss import train_center_loss
from train_routines.triplet_center_loss import train_triplet_center_loss
from train_routines.new_triplet_center import train_new_triplet_center
from preprocessing.utils import handcrafted_features, FFT

from sklearn.utils import shuffle
from scipy.spatial.distance import cosine, euclidean
import angular_grad
import tensorflow as tf
import glob 
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
from sklearn.metrics import accuracy_score

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

def main(opt):
  print(color.GREEN + '\n\n\t *************START*************\n\n' + color.END)
  
  print('\n loading data...')

  if os.path.exists('/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/y_train.npy'):
    X_train = np.load('/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/X_train.npy')
    X_test = np.load('/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/X_test.npy')
    y_train = np.load('/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/y_train.npy')
    y_test = np.load('/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/y_test.npy')
  else:
    X_train, X_test, y_train, y_test = get_data(opt)

    print('\n Converting data...')
    y_train = invert_one_hot(y_train)
    y_test = invert_one_hot(y_test)


    with open('/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/y_train.npy', 'wb') as f:
      np.save(f, y_train)
    with open('/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/y_test.npy', 'wb') as f:
      np.save(f, y_test)
    with open('/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/X_train.npy', 'wb') as f:
      np.save(f, X_train)
    with open('/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/y_train.npy', 'wb') as f:
      np.save(f, y_train)
    with open('/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/X_test.npy', 'wb') as f:
      np.save(f, X_test)

  print(f' Training data shape: {X_train.shape},  Training label shape: {y_train.shape}')
  print(f' Testing data shape: {X_test.shape},    Testing label shape: {y_test.shape}')

  if opt.use_FFT:
    print('\n Using FFT...')
    X_train_FFT = FFT(X_train)
    X_test_FFT = FFT(X_test)
  else:
    X_train_FFT = X_train
    X_test_FFT = X_test
  
  print('\n Loading model...')
  if opt.embedding_model == 'triplet':
    train_embs, test_embs, y_test_solf, y_train = train(opt, X_train_FFT, y_train, X_test_FFT, y_test, CNN_C_trip) 
  if opt.embedding_model == 'center':
    train_embs, test_embs, y_test_solf, y_train = train_center_loss(opt, X_train_FFT, y_train, X_test_FFT, y_test, CNN_C_trip) 
  if opt.embedding_model == 'triplet_center':
    train_embs, test_embs, y_test_solf, y_train = train_triplet_center_loss(opt, X_train_FFT, y_train, X_test_FFT, y_test, CNN_C_trip) 
  if opt.embedding_model == 'new_triplet_center':
    train_embs, test_embs, y_test_solf, y_train = train_new_triplet_center(opt, X_train_FFT, y_train, X_test_FFT, y_test, CNN_C_trip)
  if opt.embedding_model == 'arcface':
    train_embs, test_embs, y_train = train_ArcFaceModel(opt, X_train_FFT, y_train, X_test_FFT, y_test)
    

  print('\n Saving embedding phase...')   
  this_acc = []
  y_pred_Lo_Co = []
  y_pred_SVM_Ran = []

  if opt.embedding_model != 'arcface':
    y_test_solf = np.argmax(y_test_solf, axis=1)
    solf_acc = accuracy_score(y_test, y_test_solf)
    print(f'\n-------------- Test accuracy: {solf_acc} with the solfmax method--------------')

  y_pred_all = []
  y_pred_SVM_RandomForestClassifier = []
  y_pred_BT_RandomForestClassifier_cosine = []

  count1 = 0
  count2 = 0
  count3 = 0
  for each_ML in ['SVM', 'RandomForestClassifier', 'LogisticRegression', 'GaussianNB', 'KNN', 'BT', 'euclidean', 'cosine']:
    model = FaceNetOneShotRecognitor(opt, X_train, y_train, X_test, y_test) 
    y_pred = model.predict(test_embs=test_embs, train_embs=train_embs, ML_method=each_ML)
    count1 += 1
    acc = accuracy_score(y_test, y_pred)

    y_pred_onehot = to_one_hot(y_pred)
    if y_pred_all == []:
      y_pred_all = y_pred_onehot
    else:
      y_pred_all += y_pred_onehot

    if each_ML == 'SVM' or each_ML == 'RandomForestClassifier':
      if y_pred_SVM_RandomForestClassifier == []:
        y_pred_SVM_RandomForestClassifier = y_pred_onehot
      else:
        y_pred_SVM_RandomForestClassifier += y_pred_onehot
      count2 += 1
    
    if each_ML == 'BT' or each_ML == 'RandomForestClassifier' or each_ML == 'cosine':
      if y_pred_BT_RandomForestClassifier_cosine == []:
        y_pred_BT_RandomForestClassifier_cosine = y_pred_onehot
      else:
        y_pred_BT_RandomForestClassifier_cosine += y_pred_onehot
      count3 += 1

    print(f'\n-------------- 1. Test accuracy: {acc} with the {each_ML} method--------------')

    X_train_hand = handcrafted_features(X_train)
    X_test_hand  = handcrafted_features(X_test)
    model = FaceNetOneShotRecognitor(opt, X_train_hand, y_train, X_test_hand, y_test) 
    y_pred_no_emb = model.predict(test_embs=test_embs, train_embs=train_embs, ML_method=each_ML, emb=False)
    y_pred_onehot_no_emb = to_one_hot(y_pred_no_emb)
    
    y_pred_all += y_pred_onehot_no_emb
    count1 += 1
    acc = accuracy_score(y_test, y_pred_no_emb)

    print(f'\n-------------- 2.Test accuracy: {acc} with the {each_ML} method--------------')
  
  y_pred_SVM_RandomForestClassifier = y_pred_SVM_RandomForestClassifier.astype(np.float32) / count2
  y_pred_SVM_RandomForestClassifier = np.argmax(y_pred_SVM_RandomForestClassifier, axis=1)
  acc_case_1 = accuracy_score(y_test, y_pred_SVM_RandomForestClassifier)
  print(f'\n--------------Ensemble for case 1: {acc_case_1}--------------')

  y_pred_BT_RandomForestClassifier_cosine = y_pred_BT_RandomForestClassifier_cosine.astype(np.float32) / count3
  y_pred_BT_RandomForestClassifier_cosine = np.argmax(y_pred_BT_RandomForestClassifier_cosine, axis=1)
  acc_case_2 = accuracy_score(y_test, y_pred_BT_RandomForestClassifier_cosine)
  print(f'\n--------------Ensemble for case 2: {acc_case_2}--------------')

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