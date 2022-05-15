from FaceNet_predict import FaceNetOneShotRecognitor
from load_data import Healthy, Outer_ring_damage, Inner_ring_damage 
from preprocessing.utils import invert_one_hot, load_table_10_spe, recall_m, precision_m, f1_m, to_one_hot, handcrafted_features, scaler_transform
from network.nn import CNN_C
from src.model import CNN_C_trip
from load_cases import get_data
from train import parse_opt

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

from train_routines.triplet_loss import train
from train_routines.center_loss import train_center_loss
from train_routines.new_center_loss import train_new_center_loss
from train_routines.triplet_center_loss import train_triplet_center_loss
from train_routines.new_triplet_center_version_2 import train_new_triplet_center
from train_routines.xentropy import train_xentropy

from itertools import combinations
from sklearn.utils import shuffle
from scipy.spatial.distance import cosine, euclidean
from angular_grad import AngularGrad
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
import glob 
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import warnings
import gc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
def plot_confusion(y_test, y_pred_inv, outdir, each_ML):
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


warnings.filterwarnings("ignore", category=FutureWarning)
np.seterr(all="ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
opt = parse_opt()
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

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)

print('\t\t\t Loading labels...')
Healthy_label           = np.array([0]*len(Healthy))
Outer_ring_damage_label = np.array([1]*len(Outer_ring_damage))
Inner_ring_damage_label = np.array([2]*len(Inner_ring_damage))

if opt.PU_data_table_10_case_0:
   Healthy, Healthy_label = load_table_10_spe(Healthy, Healthy_label)
   Outer_ring_damage, Outer_ring_damage_label = load_table_10_spe(Outer_ring_damage, Outer_ring_damage_label)
   Inner_ring_damage, Inner_ring_damage_label = load_table_10_spe(Inner_ring_damage, Inner_ring_damage_label)
   if os.path.exists('/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/Healthy_10.npy'):
      Healthy = np.load('/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/Healthy_10.npy')  
      Outer_ring_damage = np.load('/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/Outer_ring_damage_10.npy')
      Inner_ring_damage = np.load('/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/Inner_ring_damage_10.npy')
   else: 
      Healthy = scaler_transform(Healthy, PowerTransformer)
      Outer_ring_damage = scaler_transform(Outer_ring_damage, PowerTransformer)
      Inner_ring_damage = scaler_transform(Inner_ring_damage, PowerTransformer)

      with open('/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/Healthy_10.npy', 'wb') as f:
         np.save(f, Healthy)
      with open('/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/Outer_ring_damage_10.npy', 'wb') as f:
         np.save(f, Outer_ring_damage)
      with open('/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/Inner_ring_damage_10.npy', 'wb') as f:
         np.save(f, Inner_ring_damage)


   np.random.seed(0)
   Healthy, Healthy_label = shuffle(Healthy, Healthy_label, random_state=0)
   Outer_ring_damage, Outer_ring_damage_label = shuffle(Outer_ring_damage, Outer_ring_damage_label, random_state=0)
   Inner_ring_damage, Inner_ring_damage_label = shuffle(Inner_ring_damage, Inner_ring_damage_label, random_state=0)

print(color.GREEN + '\n\n\t *************START*************\n\n' + color.END)
emb_accuracy_SVM = []
emb_accuracy_RandomForestClassifier = []
emb_accuracy_LogisticRegression = []
emb_accuracy_GaussianNB = []
emb_accuracy_KNN = []
emb_accuracy_BT = []

emb_accuracy_SVM_no_emb = []
emb_accuracy_RandomForestClassifier_no_emb = []
emb_accuracy_LogisticRegression_no_emb = []
emb_accuracy_GaussianNB_no_emb = []
emb_accuracy_KNN_no_emb = []
emb_accuracy_BT_no_emb = []

emb_accuracy_euclidean = []
emb_accuracy_cosine = []
emb_accuracy_ensemble = []

#------------------------------------------Case 0: shuffle------------------------------------------------
if opt.PU_data_table_10_case_0:
  for i in range(5):
    distance_Healthy = int(0.6*len(Healthy))
    start_Healthy    = int(0.2*i*len(Healthy))
    X_train_Healthy = Healthy[start_Healthy: start_Healthy+distance_Healthy]
    if len(X_train_Healthy) < distance_Healthy:
      break
    y_train_Healthy = Healthy_label[start_Healthy: start_Healthy+distance_Healthy]
    print(f'\n Shape of the Health train data and label: {X_train_Healthy.shape}, {y_train_Healthy.shape}')
    
    distance_Outer_ring_damage = int(0.6*len(Outer_ring_damage))
    start_Outer_ring_damage    = int(0.2*i*len(Outer_ring_damage))
    X_train_Outer_ring_damage, y_train_Outer_ring_damage = Outer_ring_damage[start_Outer_ring_damage: start_Outer_ring_damage+distance_Outer_ring_damage], Outer_ring_damage_label[start_Outer_ring_damage: start_Outer_ring_damage + distance_Outer_ring_damage]
    print(f'\n Shape of the Outer ring damage train data and label: {X_train_Outer_ring_damage.shape}, {y_train_Outer_ring_damage.shape}')
    
    distance_Inner_ring_damage = int(0.6*len(Inner_ring_damage))
    start_Inner_ring_damage    = int(0.2*i*len(Inner_ring_damage))
    X_train_Inner_ring_damage, y_train_Inner_ring_damage = Inner_ring_damage[start_Inner_ring_damage: start_Inner_ring_damage + distance_Inner_ring_damage], Inner_ring_damage_label[start_Inner_ring_damage: start_Inner_ring_damage + distance_Inner_ring_damage]
    print(f'\n Shape of the Inner ring damage train data and label: {X_train_Inner_ring_damage.shape}, {y_train_Inner_ring_damage.shape}')
    
    X_train = np.concatenate((X_train_Healthy, X_train_Outer_ring_damage, X_train_Inner_ring_damage))
    y_train = np.concatenate((y_train_Healthy, y_train_Outer_ring_damage, y_train_Inner_ring_damage))
    print(f'\n Shape of train data: {X_train.shape}, {y_train.shape}')
    
    print('\n'+ '-'*100)

    h = [a for a in range(len(Healthy)) if a not in range(start_Healthy, start_Healthy+distance_Healthy)]
    
    X_test_Healthy = Healthy[h]
    y_test_Healthy = Healthy_label[h]
    print(f'\n Shape of the Health test data and label: {X_test_Healthy.shape}, {y_test_Healthy.shape}')
    
    k = [a for a in range(len(Outer_ring_damage)) if a not in range(start_Outer_ring_damage, start_Outer_ring_damage+distance_Outer_ring_damage)]
    X_test_Outer_ring_damage = Outer_ring_damage[k]
    y_test_Outer_ring_damage = Outer_ring_damage_label[k]
    print(f'\n Shape of the Outer ring damage test data and label: {X_test_Outer_ring_damage.shape}, {y_test_Outer_ring_damage.shape}')
    
    l = [a for a in range(len(Inner_ring_damage)) if a not in range(start_Inner_ring_damage, start_Inner_ring_damage+distance_Inner_ring_damage)]
    X_test_Inner_ring_damage = Inner_ring_damage[l]
    y_test_Inner_ring_damage = Inner_ring_damage_label[l]
    print(f'\n Shape of the Inner ring damage test data and label: {X_test_Inner_ring_damage.shape}, {y_test_Inner_ring_damage.shape}')
    
    X_test = np.concatenate((X_test_Healthy, X_test_Outer_ring_damage, X_test_Inner_ring_damage))
    y_test = np.concatenate((y_test_Healthy, y_test_Outer_ring_damage, y_test_Inner_ring_damage))
    print(f'\n Shape of test data: {X_test.shape}, {y_test.shape}')
    print('\n'+ '-'*100)
    
    if opt.faceNet:
      print('\n Train phase...')
      train_embs, test_embs = train_new_triplet_center(opt, X_train, y_train, X_test, y_test, CNN_C_trip, i) 
      
      print('\n Saving embedding phase...')   
      this_acc = []

      y_pred_all = []
      l = 0
      for each_ML in ['SVM', 'RandomForestClassifier', 'LogisticRegression', 'GaussianNB', 'euclidean', 'cosine']:
        model = FaceNetOneShotRecognitor(opt, X_train, y_train) 
        y_pred = model.predict(test_embs=test_embs, train_embs=train_embs, ML_method=each_ML)
        y_pred_inv = np.argmax(y_pred, axis=1)

        y_pred_onehot = to_one_hot(y_pred)
        if y_pred_all == []:
          y_pred_all = y_pred_onehot
        else:
          y_pred_all += y_pred_onehot
        l += 1

        acc = accuracy_score(y_test, y_pred)
        if each_ML == 'SVM':
          emb_accuracy_SVM.append(acc)
        elif each_ML == 'RandomForestClassifier':
          emb_accuracy_RandomForestClassifier.append(acc)
        elif each_ML == 'LogisticRegression':
          emb_accuracy_LogisticRegression.append(acc)
        elif each_ML == 'GaussianNB':
          emb_accuracy_GaussianNB.append(acc)
        elif each_ML == 'euclidean':
          emb_accuracy_euclidean.append(acc)
        elif each_ML == 'cosine':
          emb_accuracy_cosine.append(acc)

        print(f'\n--------------Test accuracy: {acc} with the {each_ML} method--------------')
    else:
      y_train = to_one_hot(y_train)
      y_test = to_one_hot(y_test)
      print('\n\t\t\t Load model...')
      model = CNN_C(opt)
      model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['acc', f1_m, precision_m, recall_m]) # loss='mse'

      model.summary()
      history = model.fit(X_train, y_train,
                          epochs     = opt.epoch,
                          batch_size = opt.batch_size,
                          validation_data=(X_test, y_test),)
      _, test_acc,  test_f1_m,  test_precision_m,  test_recall_m  = model.evaluate(X_test, y_test, verbose=0)
      accuracy.append(test_acc)

    y_pred_all = y_pred_all.astype(np.float32) / l
    y_pred_all = np.argmax(y_pred_all, axis=1)
    acc_all = accuracy_score(y_test, y_pred_all)
    emb_accuracy_ensemble.append(acc_all)

    print(f'\n --------------Ensemble: {acc_all}--------------')
    print(color.GREEN + f'\n\t\t********* FINISHING ROUND {i} *********\n\n\n' + color.END)

#------------------------------------------Case 1: no shuffle------------------------------------------------
if opt.PU_data_table_10_case_1:
  comb = combinations([0, 1, 2, 3, 4], 3)
  train_l = [[1, 2, 3, 4], [0, 2, 3,  4], [0, 1, 3, 4], [0, 1, 2, 4], [0, 1, 2, 3]]
  
  # Print the obtained combinations
#   for idx, i in enumerate(list(comb)):
  for idx, i in enumerate(train_l):
    tf.keras.backend.clear_session()
    gc.collect()
    if os.path.exists(f'/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/X_train_table10_{i}.npy'):
      X_train = np.load(f'/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/X_train_table10_{i}.npy')
      X_train_scaled = np.load(f'/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/X_train_scaled_table10_{i}.npy')
      y_train = np.load(f'/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/y_train_table10_{i}.npy')
  
      X_test = np.load(f'/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/X_test_table10_{i}.npy')
      X_test_scaled = np.load(f'/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/X_test_scaled_table10_{i}.npy')
      y_test = np.load(f'/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/y_test_scaled_table10_{i}.npy')
    else:
      X_train_Healthy = Healthy[list(i)]
      y_train_Healthy = Healthy_label[list(i)]
      X_train_Healthy, y_train_Healthy = load_table_10_spe(X_train_Healthy, y_train_Healthy)
      X_train_Healthy_scaled = scaler_transform(X_train_Healthy, PowerTransformer)
      print(f'\n Shape of the Health train data and label: {X_train_Healthy.shape}, {y_train_Healthy.shape}')
      
      X_train_Outer_ring_damage, y_train_Outer_ring_damage = Outer_ring_damage[list(i)], Outer_ring_damage_label[list(i)]
      X_train_Outer_ring_damage, y_train_Outer_ring_damage = load_table_10_spe(X_train_Outer_ring_damage, y_train_Outer_ring_damage)
      X_train_Outer_ring_damage_scaled = scaler_transform(X_train_Outer_ring_damage, PowerTransformer)
      print(f'\n Shape of the Outer ring damage train data and label: {X_train_Outer_ring_damage.shape}, {y_train_Outer_ring_damage.shape}')
      
      X_train_Inner_ring_damage, y_train_Inner_ring_damage = Inner_ring_damage[list(i)], Inner_ring_damage_label[list(i)]
      X_train_Inner_ring_damage, y_train_Inner_ring_damage = load_table_10_spe(X_train_Inner_ring_damage, y_train_Inner_ring_damage)
      X_train_Inner_ring_damage_scaled = scaler_transform(X_train_Inner_ring_damage, PowerTransformer)
      print(f'\n Shape of the Inner ring damage train data and label: {X_train_Inner_ring_damage.shape}, {y_train_Inner_ring_damage.shape}')
      
      X_train = np.concatenate((X_train_Healthy, X_train_Outer_ring_damage, X_train_Inner_ring_damage))
      X_train_scaled = np.concatenate((X_train_Healthy_scaled, X_train_Outer_ring_damage_scaled, X_train_Inner_ring_damage_scaled))
      y_train = np.concatenate((y_train_Healthy, y_train_Outer_ring_damage, y_train_Inner_ring_damage))
      with open(f'/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/X_train_table10_{i}.npy', 'wb') as f:
        np.save(f, X_train)
      with open(f'/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/X_train_scaled_table10_{i}.npy', 'wb') as f:
        np.save(f, X_train_scaled)
      with open(f'/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/y_train_table10_{i}.npy', 'wb') as f:
        np.save(f, y_train)
      
      print('\n'+ '-'*100)

      h = [a for a in range(len(Healthy)) if a not in list(i)]
      X_test_Healthy = Healthy[h]
      y_test_Healthy = Healthy_label[h]
      X_test_Healthy, y_test_Healthy = load_table_10_spe(X_test_Healthy, y_test_Healthy)
      X_test_Healthy_scaled = scaler_transform(X_test_Healthy, PowerTransformer)
      print(f'\n Shape of the Health test data and label: {X_test_Healthy.shape}, {y_test_Healthy.shape}')
      
      k = [a for a in range(len(Outer_ring_damage)) if a not in list(i)]
      X_test_Outer_ring_damage = Outer_ring_damage[k]
      y_test_Outer_ring_damage = Outer_ring_damage_label[k]
      X_test_Outer_ring_damage, y_test_Outer_ring_damage = load_table_10_spe(X_test_Outer_ring_damage, y_test_Outer_ring_damage)
      X_test_Outer_ring_damage_scaled = scaler_transform(X_test_Outer_ring_damage, PowerTransformer)  
      
      l = [a for a in range(len(Inner_ring_damage)) if a not in list(i)]
      X_test_Inner_ring_damage = Inner_ring_damage[l]
      y_test_Inner_ring_damage = Inner_ring_damage_label[l]
      X_test_Inner_ring_damage, y_test_Inner_ring_damage = load_table_10_spe(X_test_Inner_ring_damage, y_test_Inner_ring_damage)
      X_test_Inner_ring_damage_scaled = scaler_transform(X_test_Inner_ring_damage, PowerTransformer)
            
      X_test = np.concatenate((X_test_Healthy, X_test_Outer_ring_damage, X_test_Inner_ring_damage))
      X_test_scaled = np.concatenate((X_test_Healthy_scaled, X_test_Outer_ring_damage_scaled, X_test_Inner_ring_damage_scaled))
      y_test = np.concatenate((y_test_Healthy, y_test_Outer_ring_damage, y_test_Inner_ring_damage))
      
      with open(f'/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/X_test_table10_{i}.npy', 'wb') as f:
        np.save(f, X_test)
      with open(f'/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/X_test_scaled_table10_{i}.npy', 'wb') as f:
        np.save(f, X_test_scaled)
      with open(f'/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/y_test_scaled_table10_{i}.npy', 'wb') as f:
        np.save(f, y_test)

    print(f'\n Shape of train data: {X_train.shape}, {y_train.shape}')
    print(f'\n Shape of test data: {X_test.shape}, {y_test.shape}')
    print('\n'+ '-'*100)

    if opt.faceNet:
      print('\n Train phase...')
      # X_train = handcrafted_features(X_train)
      # X_test  = handcrafted_features(X_test)
      print(f'\n Length the handcrafted feature vector: {X_train.shape}')
      train_embs, test_embs, y_test_solf, y_train, outdir = train_new_triplet_center(opt, X_train_scaled, X_train, y_train, X_test_scaled, X_test, y_test, CNN_C_trip, idx) 
      
      print('\n Saving embedding phase...')   
      this_acc = []

      y_pred_all = []
      count = 0
      for each_ML in ['SVM', 'RandomForestClassifier', 'LogisticRegression', 'GaussianNB', 'euclidean', 'cosine', 'KNN', 'BT']:
        model = FaceNetOneShotRecognitor(opt, X_train, y_train, X_test, y_test) 
        y_pred = model.predict(test_embs=test_embs, train_embs=train_embs, ML_method=each_ML, use_mean=False)
        y_pred_inv = np.argmax(y_pred, axis=1)
        acc = accuracy_score(y_test, y_pred_inv)
        plot_confusion(y_test, y_pred_inv, outdir, each_ML)
      
        if each_ML not in ['euclidean', 'cosine']:
          if y_pred_all == []:
            y_pred_all = y_pred
          else:
            y_pred_all += y_pred
          count += 1

        if each_ML == 'SVM':
          emb_accuracy_SVM.append(acc)
        if each_ML == 'RandomForestClassifier':
          emb_accuracy_RandomForestClassifier.append(acc)
        if each_ML == 'LogisticRegression':
          emb_accuracy_LogisticRegression.append(acc)
        if each_ML == 'GaussianNB':
          emb_accuracy_GaussianNB.append(acc)
        if each_ML == 'KNN':
          emb_accuracy_KNN.append(acc)
        if each_ML == 'BT':
          emb_accuracy_BT.append(acc)
        if each_ML == 'euclidean':
          emb_accuracy_euclidean.append(acc)
        if each_ML == 'cosine':
          emb_accuracy_cosine.append(acc)

        print(f'\n-------------- 1.Test accuracy: {acc} with the {each_ML} method--------------')
        
        # model = FaceNetOneShotRecognitor(opt, X_train, y_train, X_test, y_test) 
        # y_pred_no_emb = model.predict(test_embs=X_test, train_embs=X_train, ML_method=each_ML, emb=False)
        # y_pred_no_emb_inv = np.argmax(y_pred_no_emb, axis=1)

        # y_pred_all += y_pred_no_emb
        # count += 1
        # acc = accuracy_score(y_test, y_pred_no_emb_inv)

        # if each_ML == 'SVM':
        #   emb_accuracy_SVM_no_emb.append(acc)
        # if each_ML == 'RandomForestClassifier':
        #   emb_accuracy_RandomForestClassifier_no_emb.append(acc)
        # if each_ML == 'LogisticRegression':
        #   emb_accuracy_LogisticRegression_no_emb.append(acc)
        # if each_ML == 'GaussianNB':
        #   emb_accuracy_GaussianNB_no_emb.append(acc)
        # if each_ML == 'KNN':
        #   emb_accuracy_KNN_no_emb.append(acc)
        # if each_ML == 'BT':
        #   emb_accuracy_BT_no_emb.append(acc)
        
        # print(f'\n-------------- 2.Test accuracy: {acc} with the {each_ML} method--------------')

      y_pred_all = y_pred_all.astype(np.float32) / count
      y_pred_all = np.argmax(y_pred_all, axis=1)
      acc_all = accuracy_score(y_test, y_pred_all)
      emb_accuracy_ensemble.append(acc_all)
      print(f'\n --------------Ensemble: {acc_all}--------------')
    
    else:
      y_train = to_one_hot(y_train)
      y_test = to_one_hot(y_test)
      print('\n\t\t\t Load model...')
      # input_   = Input((opt.input_shape, 1), name='supervise_input')
      # output = CNN_C_trip(opt, input_, sup=True)
      # model = Model(inputs=input_, outputs=output)
      model = CNN_C(opt)
      model.compile(optimizer=AngularGrad(), loss='categorical_crossentropy', metrics=['acc', f1_m, precision_m, recall_m]) # loss='mse'

      model.summary()
      if idx==0:
        epoch_ = opt.epoch
      else:
        epoch_ = 10
      outdir = '/content/journal/train_routines/supervise_model'
      if os.path.isdir(outdir):
        model.load_weights(outdir)
        print(f'\n Load weight : {outdir}')
      else:
        print('\n No weight file.')
      history = model.fit(X_train, y_train,
                          epochs     = epoch_,
                          batch_size = opt.batch_size, 
                          # callbacks  = [callback],
                          validation_data = (X_test, y_test),)
      tf.saved_model.save(model, outdir)
      _, test_acc,  test_f1_m,  test_precision_m,  test_recall_m  = model.evaluate(X_test, y_test, verbose=0)
      emb_accuracy_ensemble.append(test_acc)
      print(f'\n Test accuracy: {test_acc}')
      
    print(color.GREEN + f'\n\t\t********* FINISHING ROUND {idx} *********\n\n\n' + color.END)

if opt.faceNet:
  print(color.CYAN + 'FINISH!\n' + color.END)
  print(color.CYAN + f'Test accuracy: {np.mean(emb_accuracy_SVM)} with SVM' + color.END)
  print(color.CYAN + f'Test accuracy: {np.mean(emb_accuracy_RandomForestClassifier)} with RandomForestClassifier' + color.END)
  print(color.CYAN + f'Test accuracy: {np.mean(emb_accuracy_LogisticRegression)} with LogisticRegression' + color.END)
  print(color.CYAN + f'Test accuracy: {np.mean(emb_accuracy_GaussianNB)} with GaussianNB' + color.END)
  print(color.CYAN + f'Test accuracy: {np.mean(emb_accuracy_KNN)} with KNN' + color.END)
  print(color.CYAN + f'Test accuracy: {np.mean(emb_accuracy_BT)} with BT' + color.END)

  # print(color.CYAN + f'Test accuracy: {np.mean(emb_accuracy_SVM_no_emb)} with no embedding  SVM' + color.END)
  # print(color.CYAN + f'Test accuracy: {np.mean(emb_accuracy_RandomForestClassifier_no_emb)} with no embedding RandomForestClassifier' + color.END)
  # print(color.CYAN + f'Test accuracy: {np.mean(emb_accuracy_LogisticRegression_no_emb)} with no embedding LogisticRegression' + color.END)
  # print(color.CYAN + f'Test accuracy: {np.mean(emb_accuracy_GaussianNB_no_emb)} with no embedding GaussianNB' + color.END)
  # print(color.CYAN + f'Test accuracy: {np.mean(emb_accuracy_KNN_no_emb)} with no embedding KNN' + color.END)
  # print(color.CYAN + f'Test accuracy: {np.mean(emb_accuracy_BT_no_emb)} withno embedding  BT' + color.END)

  print(color.CYAN + f'Test accuracy: {np.mean(emb_accuracy_euclidean)} with euclidean' + color.END)
  print(color.CYAN + f'Test accuracy: {np.mean(emb_accuracy_cosine)} with cosine' + color.END)

print(color.CYAN + f'Test accuracy: {np.mean(emb_accuracy_ensemble)} with ensemble' + color.END)
