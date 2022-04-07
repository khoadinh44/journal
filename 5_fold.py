from FaceNet_predict import FaceNetOneShotRecognitor
from load_data import Healthy, Outer_ring_damage, Inner_ring_damage 
from preprocessing.utils import invert_one_hot, load_table_10_spe, recall_m, precision_m, f1_m, to_one_hot
from network.nn import CNN_C
from src.model import CNN_C_trip
from load_cases import get_data
from train_routines.triplet_loss import train, parse_opt
from train_routines.center_loss import train_center_loss
from train_routines.triplet_center_loss import train_triplet_center_loss
from train_routines.xentropy import train_xentropy


from sklearn.utils import shuffle
from scipy.spatial.distance import cosine, euclidean
import angular_grad
import tensorflow as tf
import glob 
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score

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


print('\t\t\t Loading labels...')
Healthy_label           = np.array([0]*len(Healthy))
Outer_ring_damage_label = np.array([1]*len(Outer_ring_damage))
Inner_ring_damage_label = np.array([2]*len(Inner_ring_damage))

Healthy           = np.array(Healthy)
Outer_ring_damage = np.array(Outer_ring_damage)
Inner_ring_damage = np.array(Inner_ring_damage)

if opt.PU_data_table_10_case_0:
   Healthy, Healthy_label = load_table_10_spe(Healthy, Healthy_label)
   Outer_ring_damage, Outer_ring_damage_label = load_table_10_spe(Outer_ring_damage, Outer_ring_damage_label)
   Inner_ring_damage, Inner_ring_damage_label = load_table_10_spe(Inner_ring_damage, Inner_ring_damage_label)

   np.random.seed(0)
   Healthy, Healthy_label = shuffle(Healthy, Healthy_label, random_state=0)
   Outer_ring_damage, Outer_ring_damage_label = shuffle(Outer_ring_damage, Outer_ring_damage_label, random_state=0)
   Inner_ring_damage, Inner_ring_damage_label = shuffle(Inner_ring_damage, Inner_ring_damage_label, random_state=0)

print(color.GREEN + '\n\n\t *************START*************\n\n' + color.END)
emb_accuracy_SVM = []
emb_accuracy_RandomForestClassifier = []
emb_accuracy_LogisticRegression = []
emb_accuracy_GaussianNB = []
emb_accuracy_euclidean = []
emb_accuracy_cosine = []

for i in range(5):
  distance_Healthy = int(0.6*len(Healthy))
  start_Healthy    = int(0.2*i*len(Healthy))
  X_train_Healthy = Healthy[start_Healthy: start_Healthy+distance_Healthy]
  if len(X_train_Healthy) < distance_Healthy:
    break
  y_train_Healthy = Healthy_label[start_Healthy: start_Healthy+distance_Healthy]
  if opt.PU_data_table_10_case_1:
    X_train_Healthy, y_train_Healthy = load_table_10_spe(X_train_Healthy, y_train_Healthy)
  print(f'\n Shape of the Health train data and label: {X_train_Healthy.shape}, {y_train_Healthy.shape}')
  
  distance_Outer_ring_damage = int(0.6*len(Outer_ring_damage))
  start_Outer_ring_damage    = int(0.2*i*len(Outer_ring_damage))
  X_train_Outer_ring_damage, y_train_Outer_ring_damage = Outer_ring_damage[start_Outer_ring_damage: start_Outer_ring_damage+distance_Outer_ring_damage], Outer_ring_damage_label[start_Outer_ring_damage: start_Outer_ring_damage + distance_Outer_ring_damage]
  if opt.PU_data_table_10_case_1:
    X_train_Outer_ring_damage, y_train_Outer_ring_damage = load_table_10_spe(X_train_Outer_ring_damage, y_train_Outer_ring_damage)
  print(f'\n Shape of the Outer ring damage train data and label: {X_train_Outer_ring_damage.shape}, {y_train_Outer_ring_damage.shape}')
  
  distance_Inner_ring_damage = int(0.6*len(Inner_ring_damage))
  start_Inner_ring_damage    = int(0.2*i*len(Inner_ring_damage))
  X_train_Inner_ring_damage, y_train_Inner_ring_damage = Inner_ring_damage[start_Inner_ring_damage: start_Inner_ring_damage + distance_Inner_ring_damage], Inner_ring_damage_label[start_Inner_ring_damage: start_Inner_ring_damage + distance_Inner_ring_damage]
  if opt.PU_data_table_10_case_1:
    X_train_Inner_ring_damage, y_train_Inner_ring_damage = load_table_10_spe(X_train_Inner_ring_damage, y_train_Inner_ring_damage)
  print(f'\n Shape of the Inner ring damage train data and label: {X_train_Inner_ring_damage.shape}, {y_train_Inner_ring_damage.shape}')
  
  X_train = np.concatenate((X_train_Healthy, X_train_Outer_ring_damage, X_train_Inner_ring_damage))
  y_train = np.concatenate((y_train_Healthy, y_train_Outer_ring_damage, y_train_Inner_ring_damage))
  print(f'\n Shape of train data: {X_train.shape}, {y_train.shape}')
  
  print('\n'+ '-'*100)

  h = [a for a in range(len(Healthy)) if a not in range(start_Healthy, start_Healthy+distance_Healthy)]
  
  X_test_Healthy = Healthy[h]
  y_test_Healthy = Healthy_label[h]
  if opt.PU_data_table_10_case_1:
    X_test_Healthy, y_test_Healthy = load_table_10_spe(X_test_Healthy, y_test_Healthy)
  print(f'\n Shape of the Health test data and label: {X_test_Healthy.shape}, {y_test_Healthy.shape}')
  
  k = [a for a in range(len(Outer_ring_damage)) if a not in range(start_Outer_ring_damage, start_Outer_ring_damage+distance_Outer_ring_damage)]
  X_test_Outer_ring_damage = Outer_ring_damage[k]
  y_test_Outer_ring_damage = Outer_ring_damage_label[k]
  if opt.PU_data_table_10_case_1:
    X_test_Outer_ring_damage, y_test_Outer_ring_damage = load_table_10_spe(X_test_Outer_ring_damage, y_test_Outer_ring_damage)
  print(f'\n Shape of the Outer ring damage test data and label: {X_test_Outer_ring_damage.shape}, {y_test_Outer_ring_damage.shape}')
  
  l = [a for a in range(len(Inner_ring_damage)) if a not in range(start_Inner_ring_damage, start_Inner_ring_damage+distance_Inner_ring_damage)]
  X_test_Inner_ring_damage = Inner_ring_damage[l]
  y_test_Inner_ring_damage = Inner_ring_damage_label[l]
  if opt.PU_data_table_10_case_1:
    X_test_Inner_ring_damage, y_test_Inner_ring_damage = load_table_10_spe(X_test_Inner_ring_damage, y_test_Inner_ring_damage)
  print(f'\n Shape of the Inner ring damage test data and label: {X_test_Inner_ring_damage.shape}, {y_test_Inner_ring_damage.shape}')
  
  X_test = np.concatenate((X_test_Healthy, X_test_Outer_ring_damage, X_test_Inner_ring_damage))
  y_test = np.concatenate((y_test_Healthy, y_test_Outer_ring_damage, y_test_Inner_ring_damage))
  print(f'\n Shape of test data: {X_test.shape}, {y_test.shape}')
  print('\n'+ '-'*100)
  
  if opt.faceNet:
    print('\n Train phase...')
    X_test = np.expand_dims(X_test, axis=-1).astype(np.float32)

    train_embs, test_embs = train(opt, X_train, y_train, X_test, y_test, CNN_C_trip) 
    
    print('\n Saving embedding phase...')   
    this_acc = []

    for each_ML in ['SVM', 'RandomForestClassifier', 'LogisticRegression', 'GaussianNB', 'euclidean', 'cosine']:
      model = FaceNetOneShotRecognitor(opt, X_train, y_train) 
      y_pred = model.predict(test_embs=test_embs, train_embs=train_embs, threshold=1, ML_method=each_ML)
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

      print(f'\n--------------Test accuracy: {acc} with the {each_ML} method----------------')
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

  print(color.GREEN + f'\n\t\t********* FINISHING ROUND {i} *********\n\n\n' + color.END)

print(color.CYAN + 'FINISH!\n' + color.END)
print(color.CYAN + f'Test accuracy: {np.mean(emb_accuracy_SVM)} with SVM' + color.END)
print(color.CYAN + f'Test accuracy: {np.mean(emb_accuracy_RandomForestClassifier)} with RandomForestClassifier' + color.END)
print(color.CYAN + f'Test accuracy: {np.mean(emb_accuracy_LogisticRegression)} with LogisticRegression' + color.END)
print(color.CYAN + f'Test accuracy: {np.mean(emb_accuracy_GaussianNB)} with GaussianNB' + color.END)
print(color.CYAN + f'Test accuracy: {np.mean(emb_accuracy_euclidean)} with euclidean' + color.END)
print(color.CYAN + f'Test accuracy: {np.mean(emb_accuracy_cosine)} with cosine' + color.END)
