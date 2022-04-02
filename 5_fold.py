from facenet import Trainer, parse_opt
from FaceNet_predict import FaceNetOneShotRecognitor
from load_data import Healthy, Outer_ring_damage, Inner_ring_damage 
from preprocessing.utils import invert_one_hot

from src.data import get_dataset
from scipy.spatial.distance import cosine, euclidean
from load_cases import get_data
from src.params import Params
from faceNet import parse_opt
from src.model  import face_model
import angular_grad
import tensorflow as tf
import glob 
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score

opt = parse_opt()

print('\t\t\t Loading labels...')
Healthy_label = [0]*len(Healthy)
Outer_ring_damage_label = [1]*len(Outer_ring_damage)
Inner_ring_damage_label = [2]*len(Inner_ring_damage)

for i in range(len(Healthy)):
  X_train_Healthy = Healthy[i: i+3]
  y_train_Healthy = Healthy_label[i: i+3]
  X_train_Healthy, y_train_Healthy = load_table_10_spe(X_train_Healthy, y_train_Healthy)
  print(f'\n Shape of the Health train data and label: {X_train_Healthy.shape}, {y_train_Healthy.shape}')
  
  X_train_Outer_ring_damage = Outer_ring_damage[i: i+3]
  y_train_Outer_ring_damage = Outer_ring_damage_label[i: i+3]
  X_train_Outer_ring_damage, y_train_Outer_ring_damage = load_table_10_spe(X_train_Outer_ring_damage, y_train_Outer_ring_damage)
  print(f'\n Shape of the Outer ring damage train data and label: {X_train_Outer_ring_damage.shape}, {y_train_Outer_ring_damage.shape}')
  
  X_train_Inner_ring_damage = Inner_ring_damage[i: i+3]
  y_train_Inner_ring_damage = Inner_ring_damage_label[i: i+3]
  X_train_Inner_ring_damage, y_train_Inner_ring_damage = load_table_10_spe(X_train_Inner_ring_damage, y_train_Inner_ring_damage)
  print(f'\n Shape of the Inner ring damage train data and label: {X_train_Inner_ring_damage.shape}, {y_train_Inner_ring_damage.shape}')
  
  X_train = np.concatenate((X_train_Healthy, X_train_Outer_ring_damage, X_train_Inner_ring_damage))
  y_train = np.concatenate((y_train_Healthy, y_train_Outer_ring_damage, y_train_Inner_ring_damage))
  print(f'\n Shape of train data: {X_train.shape}, {y_train.shape}')
  if len(X_train_Inner_ring_damage) != 3:
    break
  
  print('\n------------------------------------------------')
  X_test_Healthy = [i for i in Healthy if i not in X_train_Healthy]
  y_test_Healthy = [i for i in Healthy_label if i not in y_train_Healthy]
  X_test_Healthy, y_test_Healthy = load_table_10_spe(X_test_Healthy, y_test_Healthy)
  print(f'\n Shape of the Health test data and label: {X_test_Healthy.shape}, {y_test_Healthy.shape}')
  
  X_test_Outer_ring_damage = [i for i in Outer_ring_damage if i not in X_train_Outer_ring_damage]
  y_test_Outer_ring_damage = [i for i in Outer_ring_damage_label if i not in y_train_Outer_ring_damage]
  X_test_Outer_ring_damage, y_test_Outer_ring_damage = load_table_10_spe(X_test_Outer_ring_damage, y_test_Outer_ring_damage)
  print(f'\n Shape of the Outer ring damage test data and label: {X_test_Outer_ring_damage.shape}, {y_test_Outer_ring_damage.shape}')
  
  X_test_Inner_ring_damage = [i for i in Inner_ring_damage if i not in X_train_Inner_ring_damage]
  y_test_Inner_ring_damage = [i for i in Inner_ring_damage_label if i not in y_train_Inner_ring_damage]
  X_test_Inner_ring_damage, y_test_Inner_ring_damage = load_table_10_spe(X_test_Inner_ring_damage, y_test_Inner_ring_damage)
  print(f'\n Shape of the Inner ring damage test data and label: {X_test_Inner_ring_damage.shape}, {y_test_Inner_ring_damage.shape}')
  
  X_test = np.concatenate((X_test_Healthy, X_test_Outer_ring_damage, X_test_Inner_ring_damage))
  y_test = np.concatenate((y_test_Healthy, y_test_Outer_ring_damage, y_test_Inner_ring_damage))
  print(f'\n Shape of test data: {X_test.shape}, {y_test.shape}')
