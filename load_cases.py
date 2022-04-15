from sklearn.model_selection import train_test_split
from preprocessing.utils import convert_one_hot, choosing_features

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
import argparse
import numpy as np

def get_data(opt):
  if opt.data_normal:
    from load_data import Normal_0_train, Normal_0_test, Normal_1_train, Normal_1_test, Normal_2_train, Normal_2_test, Normal_3_train, Normal_3_test,\
                          Normal_0_label, Normal_1_label, Normal_2_label, Normal_3_label
  if opt.data_12k:
    from load_data import B007_0_train, B007_0_test, B007_0_label, B007_1_train, B007_1_test, B007_1_label, B007_2_train, B007_2_test, B007_2_label, B007_3_train, B007_3_test, B007_3_label,\
                          B014_0_train, B014_0_test, B014_0_label, B014_1_train, B014_1_test, B014_1_label, B014_2_train, B014_2_test, B014_2_label, B014_3_train, B014_3_test, B014_3_label,\
                          B021_0_train, B021_0_test, B021_0_label, B021_1_train, B021_1_test, B021_1_label, B021_2_train, B021_2_test, B021_2_label, B021_3_train, B021_3_test, B021_3_label,\
                          B028_0_train, B028_0_test, B028_0_label, B028_1_train, B028_1_test, B028_1_label, B028_2_train, B028_2_test, B028_2_label, B028_3_train, B028_3_test, B028_3_label,\
                          IR007_0_train, IR007_0_test, IR007_0_label, IR007_1_train, IR007_1_test, IR007_1_label, IR007_2_train, IR007_2_test, IR007_2_label, IR007_3_train, IR007_3_test, IR007_3_label,\
                          IR014_0_train, IR014_0_test, IR014_0_label, IR014_1_train, IR014_1_test, IR014_1_label, IR014_2_train, IR014_2_test, IR014_2_label, IR014_3_train, IR014_3_test, IR014_3_label,\
                          IR021_0_train, IR021_0_test, IR021_0_label, IR021_1_train, IR021_1_test, IR021_1_label, IR021_2_train, IR021_2_test, IR021_2_label, IR021_3_train, IR021_3_test, IR021_3_label,\
                          IR028_0_train, IR028_0_test, IR028_0_label, IR028_1_train, IR028_1_test, IR028_1_label, IR028_2_train, IR028_2_test, IR028_2_label, IR028_3_train, IR028_3_test, IR028_3_label,\
                          OR007_12_0_train, OR007_12_0_test, OR007_12_0_label, OR007_12_1_train, OR007_12_1_test, OR007_12_1_label, OR007_12_2_train, OR007_12_2_test, OR007_12_2_label, OR007_12_3_train, OR007_12_3_test, OR007_12_3_label,\
                          OR007_3_0_train, OR007_3_0_test, OR007_3_0_label, OR007_3_1_train, OR007_3_1_test, OR007_3_1_label, OR007_3_2_train, OR007_3_2_test, OR007_3_2_label, OR007_3_3_train, OR007_3_3_test, OR007_3_3_label,\
                          OR007_6_0_train, OR007_6_0_test, OR007_6_0_label, OR007_6_1_train, OR007_6_1_test, OR007_6_1_label, OR007_6_2_train, OR007_6_2_test, OR007_6_2_label, OR007_6_3_train, OR007_6_3_test, OR007_6_3_label,\
                          OR014_6_0_train, OR014_6_0_test, OR014_6_0_label, OR014_6_1_train, OR014_6_1_test, OR014_6_1_label, OR014_6_2_train, OR014_6_2_test, OR014_6_2_label, OR014_6_3_train, OR014_6_3_test, OR014_6_3_label,\
                          OR021_6_0_train, OR021_6_0_test, OR021_6_0_label, OR021_6_1_train, OR021_6_1_test, OR021_6_1_label, OR021_6_2_train, OR021_6_2_test, OR021_6_2_label, OR021_6_3_train, OR021_6_3_test, OR021_6_3_label,\
                          OR021_3_0_train, OR021_3_0_test, OR021_3_0_label, OR021_3_1_train, OR021_3_1_test, OR021_3_1_label, OR021_3_2_train, OR021_3_2_test, OR021_3_2_label, OR021_3_3_train, OR021_3_3_test, OR021_3_3_label,\
                          OR021_12_0_train, OR021_12_0_test, OR021_12_0_label, OR021_12_1_train, OR021_12_1_test, OR021_12_1_label, OR021_12_2_train, OR021_12_2_test, OR021_12_2_label, OR021_12_3_train, OR021_12_3_test, OR021_12_3_label
  
  if opt.data_48k:
    from load_data import B007_0_train, B007_0_test, B007_0_label, B007_1_train, B007_1_test, B007_1_label, B007_2_train, B007_2_test, B007_2_label, B007_3_train, B007_3_test, B007_3_label,\
                          IR007_0_train, IR007_0_test, IR007_0_label, IR007_1_train, IR007_1_test, IR007_1_label, IR007_2_train, IR007_2_test, IR007_2_label, IR007_3_train, IR007_3_test, IR007_3_label,\
                          OR007_12_0_train, OR007_12_0_test, OR007_12_0_label, OR007_12_1_train, OR007_12_1_test, OR007_12_1_label, OR007_12_2_train, OR007_12_2_test, OR007_12_2_label, OR007_12_3_train, OR007_12_3_test, OR007_12_3_label,\
                          OR007_3_0_train, OR007_3_0_test, OR007_3_0_label, OR007_3_1_train, OR007_3_1_test, OR007_3_1_label, OR007_3_2_train, OR007_3_2_test, OR007_3_2_label, OR007_3_3_train, OR007_3_3_test, OR007_3_3_label,\
                          OR007_6_0_train, OR007_6_0_test, OR007_6_0_label, OR007_6_1_train, OR007_6_1_test, OR007_6_1_label, OR007_6_2_train, OR007_6_2_test, OR007_6_2_label, OR007_6_3_train, OR007_6_3_test, OR007_6_3_label

  if opt.case_0_6:
    all_data_0_train = np.concatenate((Normal_0_train, IR007_0_train, B007_0_train, OR007_6_0_train, OR007_3_0_train, OR007_12_0_train))
    all_data_0_test = np.concatenate((Normal_0_test, IR007_0_test, B007_0_test, OR007_6_0_test, OR007_3_0_test, OR007_12_0_test))
    
    Normal_0_label_all_train = convert_one_hot(Normal_0_label) * Normal_0_train.shape[0]
    IR007_0_label_all_train = convert_one_hot(IR007_0_label) * IR007_0_train.shape[0]
    B007_0_label_all_train = convert_one_hot(B007_0_label) * B007_0_train.shape[0]
    OR007_6_0_label_all_train = convert_one_hot(OR007_6_0_label) * OR007_6_0_train.shape[0]
    OR007_3_0_label_all_train = convert_one_hot(OR007_3_0_label) * OR007_3_0_train.shape[0]
    OR007_12_0_label_all_train = convert_one_hot(OR007_12_0_label) * OR007_12_0_train.shape[0]
    all_labels_0_train = np.concatenate((Normal_0_label_all_train, IR007_0_label_all_train, B007_0_label_all_train, OR007_6_0_label_all_train, OR007_3_0_label_all_train, OR007_12_0_label_all_train))
    
    Normal_0_label_all_test = convert_one_hot(Normal_0_label) * Normal_0_test.shape[0] 
    IR007_0_label_all_test = convert_one_hot(IR007_0_label) * IR007_0_test.shape[0]
    B007_0_label_all_test = convert_one_hot(B007_0_label) * B007_0_test.shape[0]
    OR007_6_0_label_all_test = convert_one_hot(OR007_6_0_label) * OR007_6_0_test.shape[0]
    OR007_3_0_label_all_test = convert_one_hot(OR007_3_0_label) * OR007_3_0_test.shape[0]
    OR007_12_0_label_all_test = convert_one_hot(OR007_12_0_label) * OR007_12_0_test.shape[0]
    all_labels_0_test = np.concatenate((Normal_0_label_all_test, IR007_0_label_all_test, B007_0_label_all_test, OR007_6_0_label_all_test, OR007_3_0_label_all_test, OR007_12_0_label_all_test))

    X_train, X_test, y_train, y_test = all_data_0_train, all_data_0_test, all_labels_0_train, all_labels_0_test

  if opt.case_1_7:
    all_data_1_train = np.concatenate((Normal_1_train, IR007_1_train, B007_1_train, OR007_6_1_train, OR007_3_1_train, OR007_12_1_train))
    Normal_1_label_all_train = convert_one_hot(Normal_1_label) * Normal_1_train.shape[0]
    IR007_1_label_all_train = convert_one_hot(IR007_1_label) * IR007_1_train.shape[0]
    B007_1_label_all_train = convert_one_hot(B007_1_label) * B007_1_train.shape[0]
    OR007_6_1_label_all_train = convert_one_hot(OR007_6_1_label) * OR007_6_1_train.shape[0]
    OR007_3_1_label_all_train = convert_one_hot(OR007_3_1_label) * OR007_3_1_train.shape[0]
    OR007_12_1_label_all_train = convert_one_hot(OR007_12_1_label) * OR007_12_1_train.shape[0]
    all_labels_1_train = np.concatenate((Normal_1_label_all_train, IR007_1_label_all_train, B007_1_label_all_train, OR007_6_1_label_all_train, OR007_3_1_label_all_train, OR007_12_1_label_all_train))
  
    all_data_1_test = np.concatenate((Normal_1_test, IR007_1_test, B007_1_test, OR007_6_1_test, OR007_3_1_test, OR007_12_1_test))
    Normal_1_label_all_test = convert_one_hot(Normal_1_label) * Normal_1_test.shape[0]
    IR007_1_label_all_test = convert_one_hot(IR007_1_label) * IR007_1_test.shape[0]
    B007_1_label_all_test = convert_one_hot(B007_1_label) * B007_1_test.shape[0]
    OR007_6_1_label_all_test = convert_one_hot(OR007_6_1_label) * OR007_6_1_test.shape[0]
    OR007_3_1_label_all_test = convert_one_hot(OR007_3_1_label) * OR007_3_1_test.shape[0]
    OR007_12_1_label_all_test = convert_one_hot(OR007_12_1_label) * OR007_12_1_test.shape[0]
    all_labels_1_test = np.concatenate((Normal_1_label_all_test, IR007_1_label_all_test, B007_1_label_all_test, OR007_6_1_label_all_test, OR007_3_1_label_all_test, OR007_12_1_label_all_test))
    
    X_train, X_test, y_train, y_test = all_data_1_train, all_data_1_test, all_labels_1_train, all_labels_1_test
    
  if opt.case_2_8:
    all_data_2_train = np.concatenate((Normal_2_train, IR007_2_train, B007_2_train, OR007_6_2_train, OR007_3_2_train, OR007_12_2_train))
    Normal_2_label_all_train = convert_one_hot(Normal_2_label) * Normal_2_train.shape[0]
    IR007_2_label_all_train = convert_one_hot(IR007_2_label) * IR007_2_train.shape[0]
    B007_2_label_all_train = convert_one_hot(B007_2_label) * B007_2_train.shape[0]
    OR007_6_2_label_all_train = convert_one_hot(OR007_6_2_label) * OR007_6_2_train.shape[0]
    OR007_3_2_label_all_train = convert_one_hot(OR007_3_2_label) * OR007_3_2_train.shape[0]
    OR007_12_2_label_all_train = convert_one_hot(OR007_12_2_label) * OR007_12_2_train.shape[0]
    all_labels_2_train = np.concatenate((Normal_2_label_all_train, IR007_2_label_all_train, B007_2_label_all_train, OR007_6_2_label_all_train, OR007_3_2_label_all_train, OR007_12_2_label_all_train))
    
    all_data_2_test = np.concatenate((Normal_2_test, IR007_2_test, B007_2_test, OR007_6_2_test, OR007_3_2_test, OR007_12_2_test))
    Normal_2_label_all_test = convert_one_hot(Normal_2_label) * Normal_2_test.shape[0]
    IR007_2_label_all_test = convert_one_hot(IR007_2_label) * IR007_2_test.shape[0]
    B007_2_label_all_test = convert_one_hot(B007_2_label) * B007_2_test.shape[0]
    OR007_6_2_label_all_test = convert_one_hot(OR007_6_2_label) * OR007_6_2_test.shape[0]
    OR007_3_2_label_all_test = convert_one_hot(OR007_3_2_label) * OR007_3_2_test.shape[0]
    OR007_12_2_label_all_test = convert_one_hot(OR007_12_2_label) * OR007_12_2_test.shape[0]
    all_labels_2_test = np.concatenate((Normal_2_label_all_test, IR007_2_label_all_test, B007_2_label_all_test, OR007_6_2_label_all_test, OR007_3_2_label_all_test, OR007_12_2_label_all_test))
    
    X_train, X_test, y_train, y_test = all_data_2_train, all_data_2_test, all_labels_2_train, all_labels_2_test

  if opt.case_3_9:
    all_data_3_train = np.concatenate((Normal_3_train, IR007_3_train, B007_3_train, OR007_6_3_train, OR007_3_3_train, OR007_12_3_train))
    Normal_3_label_all_train = convert_one_hot(Normal_3_label) * Normal_3_train.shape[0]
    IR007_3_label_all_train = convert_one_hot(IR007_3_label) * IR007_3_train.shape[0]
    B007_3_label_all_train = convert_one_hot(B007_3_label) * B007_3_train.shape[0]
    OR007_6_3_label_all_train = convert_one_hot(OR007_6_3_label) * OR007_6_3_train.shape[0]
    OR007_3_3_label_all_train = convert_one_hot(OR007_3_3_label) * OR007_3_3_train.shape[0]
    OR007_12_3_label_all_train = convert_one_hot(OR007_12_3_label) * OR007_12_3_train.shape[0]
    all_labels_3_train = np.concatenate((Normal_3_label_all_train, IR007_3_label_all_train, B007_3_label_all_train, OR007_6_3_label_all_train, OR007_3_3_label_all_train, OR007_12_3_label_all_train))
    
    all_data_3_test = np.concatenate((Normal_3_test, IR007_3_test, B007_3_test, OR007_6_3_test, OR007_3_3_test, OR007_12_3_test))
    Normal_3_label_all_test = convert_one_hot(Normal_3_label) * Normal_3_test.shape[0]
    IR007_3_label_all_test = convert_one_hot(IR007_3_label) * IR007_3_test.shape[0]
    B007_3_label_all_test = convert_one_hot(B007_3_label) * B007_3_test.shape[0]
    OR007_6_3_label_all_test = convert_one_hot(OR007_6_3_label) * OR007_6_3_test.shape[0]
    OR007_3_3_label_all_test = convert_one_hot(OR007_3_3_label) * OR007_3_3_test.shape[0]
    OR007_12_3_label_all_test = convert_one_hot(OR007_12_3_label) * OR007_12_3_test.shape[0]
    all_labels_3_test = np.concatenate((Normal_3_label_all_test, IR007_3_label_all_test, B007_3_label_all_test, OR007_6_3_label_all_test, OR007_3_3_label_all_test, OR007_12_3_label_all_test))
    
    X_train, X_test, y_train, y_test = all_data_3_train, all_data_3_test, all_labels_3_train, all_labels_3_test

  if opt.case_4_10:
    all_data_4_train = np.concatenate((all_data_0_train, all_data_1_train, all_data_2_train, all_data_3_train))
    all_labels_4_train = np.concatenate((all_labels_0_train, all_labels_1_train, all_labels_2_train, all_labels_3_train))
    
    all_data_4_test = np.concatenate((all_data_0_test, all_data_1_test, all_data_2_test, all_data_3_test))
    all_labels_4_test = np.concatenate((all_labels_0_test, all_labels_1_test, all_labels_2_test, all_labels_3_test))
    
    X_train, X_test, y_train, y_test = all_data_4_train, all_data_4_test, all_labels_4_train, all_labels_4_test

  if opt.case_5_11:
    Normal_5_train = np.concatenate((Normal_0_train, Normal_1_train, Normal_2_train, Normal_3_train))
    IR007_5_train = np.concatenate((IR007_0_train, IR007_1_train, IR007_2_train, IR007_3_train))
    B007_5_train = np.concatenate((B007_0_train, B007_1_train, B007_2_train, B007_3_train))
    OR007_6_5_train = np.concatenate((OR007_6_0_train, OR007_6_1_train, OR007_6_2_train, OR007_6_3_train))
    OR007_3_5_train = np.concatenate((OR007_3_0_train, OR007_3_1_train, OR007_3_2_train, OR007_3_3_train))
    OR007_12_5_train = np.concatenate((OR007_12_0_train, OR007_12_1_train, OR007_12_2_train, OR007_12_3_train))

    all_data_5_train = np.concatenate((Normal_5_train, IR007_5_train, B007_5_train, OR007_6_5_train, OR007_3_5_train, OR007_12_5_train))

    Normal_5_label_all_train = convert_one_hot(Normal_0_label) * Normal_5_train.shape[0]
    IR007_5_label_all_train = convert_one_hot(IR007_0_label) * IR007_5_train.shape[0]
    B007_5_label_all_train = convert_one_hot(B007_0_label) * B007_5_train.shape[0]
    OR007_6_5_label_all_train = convert_one_hot(OR007_6_0_label) * OR007_6_5_train.shape[0]
    OR007_3_5_label_all_train = convert_one_hot(OR007_3_0_label) * OR007_3_5_train.shape[0]
    OR007_12_5_label_all_train = convert_one_hot(OR007_12_0_label) * OR007_12_5_train.shape[0]
    all_labels_5_train = np.concatenate((Normal_5_label_all_train, IR007_5_label_all_train, B007_5_label_all_train, OR007_6_5_label_all_train, OR007_3_5_label_all_train, OR007_12_5_label_all_train))

    
    Normal_5_test = np.concatenate((Normal_0_test, Normal_1_test, Normal_2_test, Normal_3_test))
    IR007_5_test = np.concatenate((IR007_0_test, IR007_1_test, IR007_2_test, IR007_3_test))
    B007_5_test = np.concatenate((B007_0_test, B007_1_test, B007_2_test, B007_3_test))
    OR007_6_5_test = np.concatenate((OR007_6_0_test, OR007_6_1_test, OR007_6_2_test, OR007_6_3_test))
    OR007_3_5_test = np.concatenate((OR007_3_0_test, OR007_3_1_test, OR007_3_2_test, OR007_3_3_test))
    OR007_12_5_test = np.concatenate((OR007_12_0_test, OR007_12_1_test, OR007_12_2_test, OR007_12_3_test))

    all_data_5_test = np.concatenate((Normal_5_test, IR007_5_test, B007_5_test, OR007_6_5_test, OR007_3_5_test, OR007_12_5_test))

    Normal_5_label_all_test = convert_one_hot(Normal_0_label) * Normal_5_test.shape[0]
    IR007_5_label_all_test = convert_one_hot(IR007_0_label) * IR007_5_test.shape[0]
    B007_5_label_all_test = convert_one_hot(B007_0_label) * B007_5_test.shape[0]
    OR007_6_5_label_all_test = convert_one_hot(OR007_6_0_label) * OR007_6_5_test.shape[0]
    OR007_3_5_label_all_test = convert_one_hot(OR007_3_0_label) * OR007_3_5_test.shape[0]
    OR007_12_5_label_all_test = convert_one_hot(OR007_12_0_label) * OR007_12_5_test.shape[0]
    all_labels_5_test = np.concatenate((Normal_5_label_all_test, IR007_5_label_all_test, B007_5_label_all_test, OR007_6_5_label_all_test, OR007_3_5_label_all_test, OR007_12_5_label_all_test))
    
    X_train, X_test, y_train, y_test = all_data_5_train, all_data_5_test, all_labels_5_train, all_labels_5_test
  
  if opt.case_12:
    all_data_12_train = np.concatenate((all_data_4_train, IR021_0_train, B021_0_train, OR021_6_0_train, OR021_3_0_train, OR021_12_0_train,
                                  IR021_1_train, B021_1_train, OR021_6_1_train, OR021_3_1_train, OR021_12_1_train,
                                  IR021_2_train, B021_2_train, OR021_6_2_train, OR021_3_2_train, OR021_12_2_train,
                                  IR021_3_train, B021_3_train, OR021_6_3_train, OR021_3_3_train, OR021_12_3_train))
    
    IR021_0_label_all_train = convert_one_hot(IR021_0_label) * IR021_0_train.shape[0]
    B021_0_label_all_train = convert_one_hot(B021_0_label) * B021_0_train.shape[0]
    OR021_6_0_label_all_train = convert_one_hot(OR021_6_0_label) * OR021_6_0_train.shape[0]
    OR021_3_0_label_all_train = convert_one_hot(OR021_3_0_label) * OR021_3_0_train.shape[0]
    OR021_12_0_label_all_train = convert_one_hot(OR021_12_0_label) * OR021_12_0_train.shape[0]
    
    IR021_1_label_all_train = convert_one_hot(IR021_1_label) * IR021_1_train.shape[0]
    B021_1_label_all_train = convert_one_hot(B021_1_label) * B021_1_train.shape[0]
    OR021_6_1_label_all_train = convert_one_hot(OR021_6_1_label) * OR021_6_1_train.shape[0]
    OR021_3_1_label_all_train = convert_one_hot(OR021_3_1_label) * OR021_3_1_train.shape[0]
    OR021_12_1_label_all_train = convert_one_hot(OR021_12_1_label) * OR021_12_1_train.shape[0]
    
    IR021_2_label_all_train = convert_one_hot(IR021_2_label) * IR021_2_train.shape[0]
    B021_2_label_all_train = convert_one_hot(B021_2_label) * B021_2_train.shape[0]
    OR021_6_2_label_all_train = convert_one_hot(OR021_6_2_label) * OR021_6_2_train.shape[0]
    OR021_3_2_label_all_train = convert_one_hot(OR021_3_2_label) * OR021_3_2_train.shape[0]
    OR021_12_2_label_all_train = convert_one_hot(OR021_12_2_label) * OR021_12_2_train.shape[0]
    
    IR021_3_label_all_train = convert_one_hot(IR021_3_label) * IR021_3_train.shape[0]
    B021_3_label_all_train = convert_one_hot(B021_3_label) * B021_3_train.shape[0]
    OR021_6_3_label_all_train = convert_one_hot(OR021_6_3_label) * OR021_6_3_train.shape[0]
    OR021_3_3_label_all_train = convert_one_hot(OR021_3_3_label) * OR021_3_3_train.shape[0]
    OR021_12_3_label_all_train = convert_one_hot(OR021_12_3_label) * OR021_12_3_train.shape[0]
    
    all_labels_12_train = np.concatenate((all_labels_4_train, IR021_0_label_all_train, B021_0_label_all_train, OR021_6_0_label_all_train, OR021_3_0_label_all_train, OR021_12_0_label_all_train,
                                    IR021_1_label_all_train, B021_1_label_all_train, OR021_6_1_label_all_train, OR021_3_1_label_all_train, OR021_12_1_label_all_train,
                                    IR021_2_label_all_train, B021_2_label_all_train, OR021_6_2_label_all_train, OR021_3_2_label_all_train, OR021_12_2_label_all_train,
                                    IR021_3_label_all_train, B021_3_label_all_train, OR021_6_3_label_all_train, OR021_3_3_label_all_train, OR021_12_3_label_all_train))
    
    
    all_data_12_test = np.concatenate((all_data_4_test, IR021_0_test, B021_0_test, OR021_6_0_test, OR021_3_0_test, OR021_12_0_test,
                                  IR021_1_test, B021_1_test, OR021_6_1_test, OR021_3_1_test, OR021_12_1_test,
                                  IR021_2_test, B021_2_test, OR021_6_2_test, OR021_3_2_test, OR021_12_2_test,
                                  IR021_3_test, B021_3_test, OR021_6_3_test, OR021_3_3_test, OR021_12_3_test))
    
    IR021_0_label_all_test = convert_one_hot(IR021_0_label) * IR021_0_test.shape[0]
    B021_0_label_all_test = convert_one_hot(B021_0_label) * B021_0_test.shape[0]
    OR021_6_0_label_all_test = convert_one_hot(OR021_6_0_label) * OR021_6_0_test.shape[0]
    OR021_3_0_label_all_test = convert_one_hot(OR021_3_0_label) * OR021_3_0_test.shape[0]
    OR021_12_0_label_all_test = convert_one_hot(OR021_12_0_label) * OR021_12_0_test.shape[0]
    
    IR021_1_label_all_test = convert_one_hot(IR021_1_label) * IR021_1_test.shape[0]
    B021_1_label_all_test = convert_one_hot(B021_1_label) * B021_1_test.shape[0]
    OR021_6_1_label_all_test = convert_one_hot(OR021_6_1_label) * OR021_6_1_test.shape[0]
    OR021_3_1_label_all_test = convert_one_hot(OR021_3_1_label) * OR021_3_1_test.shape[0]
    OR021_12_1_label_all_test = convert_one_hot(OR021_12_1_label) * OR021_12_1_test.shape[0]
    
    IR021_2_label_all_test = convert_one_hot(IR021_2_label) * IR021_2_test.shape[0]
    B021_2_label_all_test = convert_one_hot(B021_2_label) * B021_2_test.shape[0]
    OR021_6_2_label_all_test = convert_one_hot(OR021_6_2_label) * OR021_6_2_test.shape[0]
    OR021_3_2_label_all_test = convert_one_hot(OR021_3_2_label) * OR021_3_2_test.shape[0]
    OR021_12_2_label_all_test = convert_one_hot(OR021_12_2_label) * OR021_12_2_test.shape[0]
    
    IR021_3_label_all_test = convert_one_hot(IR021_3_label) * IR021_3_test.shape[0]
    B021_3_label_all_test = convert_one_hot(B021_3_label) * B021_3_test.shape[0]
    OR021_6_3_label_all_test = convert_one_hot(OR021_6_3_label) * OR021_6_3_test.shape[0]
    OR021_3_3_label_all_test = convert_one_hot(OR021_3_3_label) * OR021_3_3_test.shape[0]
    OR021_12_3_label_all_test = convert_one_hot(OR021_12_3_label) * OR021_12_3_test.shape[0]
    
    all_labels_12_test = np.concatenate((all_labels_4_test, IR021_0_label_all_test, B021_0_label_all_test, OR021_6_0_label_all_test, OR021_3_0_label_all_test, OR021_12_0_label_all_test,
                                    IR021_1_label_all_test, B021_1_label_all_test, OR021_6_1_label_all_test, OR021_3_1_label_all_test, OR021_12_1_label_all_test,
                                    IR021_2_label_all_test, B021_2_label_all_test, OR021_6_2_label_all_test, OR021_3_2_label_all_test, OR021_12_2_label_all_test,
                                    IR021_3_label_all_test, B021_3_label_all_test, OR021_6_3_label_all_test, OR021_3_3_label_all_test, OR021_12_3_label_all_test))
    
    X_train, X_test, y_train, y_test = all_data_12_train, all_data_12_test, all_labels_12_train, all_labels_12_test
    
  if opt.case_13:
    IR021_13_train    = np.concatenate((IR007_5_train, IR021_0_train, IR021_1_train, IR021_2_train, IR021_3_train))
    B021_13_train     = np.concatenate((B007_5_train, B021_0_train, B007_1_train, B021_2_train, B021_3_train))
    OR021_6_13_train  = np.concatenate((OR007_6_5_train, OR021_6_0_train, OR021_6_1_train, OR021_6_2_train, OR021_6_3_train))
    OR021_3_13_train  = np.concatenate((OR007_3_5_train, OR021_3_0_train, OR021_3_1_train, OR021_3_2_train, OR021_3_3_train))
    OR021_12_13_train = np.concatenate((OR007_12_5_train, OR021_12_0_train, OR021_12_1_train, OR021_12_2_train, OR021_12_3_train))

    all_data_13_train = np.concatenate((Normal_5_train, IR021_13_train, B021_13_train, OR021_6_13_train, OR021_3_13_train, OR021_12_13_train))
    
    IR021_13_label_all_train = convert_one_hot(IR007_0_label) * IR021_13_train.shape[0]
    B021_13_label_all_train = convert_one_hot(B007_0_label) * B021_13_train.shape[0]
    OR021_6_13_label_all_train = convert_one_hot(OR007_6_0_label) * OR021_6_13_train.shape[0]
    OR021_3_13_label_all_train = convert_one_hot(OR007_3_0_label) * OR021_3_13_train.shape[0]
    OR021_12_13_label_all_train = convert_one_hot(OR007_12_0_label) * OR021_12_13_train.shape[0]
    all_labels_13_train = np.concatenate((Normal_5_label_all_train, IR021_13_label_all_train, B021_13_label_all_train, OR021_6_13_label_all_train, OR021_3_13_label_all_train, OR021_12_13_label_all_train))

    
    IR021_13_test    = np.concatenate((IR007_5_test, IR021_0_test, IR021_1_test, IR021_2_test, IR021_3_test))
    B021_13_test     = np.concatenate((B007_5_test, B021_0_test, B007_1_test, B021_2_test, B021_3_test))
    OR021_6_13_test  = np.concatenate((OR007_6_5_test, OR021_6_0_test, OR021_6_1_test, OR021_6_2_test, OR021_6_3_test))
    OR021_3_13_test  = np.concatenate((OR007_3_5_test, OR021_3_0_test, OR021_3_1_test, OR021_3_2_test, OR021_3_3_test))
    OR021_12_13_test = np.concatenate((OR007_12_5_test, OR021_12_0_test, OR021_12_1_test, OR021_12_2_test, OR021_12_3_test))

    all_data_13_test = np.concatenate((Normal_5_test, IR021_13_test, B021_13_test, OR021_6_13_test, OR021_3_13_test, OR021_12_13_test))
    
    IR021_13_label_all_test = convert_one_hot(IR007_0_label) * IR021_13_test.shape[0]
    B021_13_label_all_test = convert_one_hot(B007_0_label) * B021_13_test.shape[0]
    OR021_6_13_label_all_test = convert_one_hot(OR007_6_0_label) * OR021_6_13_test.shape[0]
    OR021_3_13_label_all_test = convert_one_hot(OR007_3_0_label) * OR021_3_13_test.shape[0]
    OR021_12_13_label_all_test = convert_one_hot(OR007_12_0_label) * OR021_12_13_test.shape[0]
    all_labels_13_test = np.concatenate((Normal_5_label_all_test, IR021_13_label_all_test, B021_13_label_all_test, OR021_6_13_label_all_test, OR021_3_13_label_all_test, OR021_12_13_label_all_test))

    X_train, X_test, y_train, y_test = all_data_13_train, all_data_13_test, all_labels_13_train, all_labels_13_test
  
  if opt.case_14:
    all_data_14_train = np.concatenate((all_data_12_train, IR014_0_train, IR014_1_train, IR014_2_train, IR014_3_train,
                                  B014_0_train, B014_1_train, B014_2_train, B014_3_train,
                                  OR014_6_0_train, OR014_6_1_train, OR014_6_2_train, OR014_6_3_train,
                                  IR028_0_train, 	IR028_1_train, 	IR028_2_train, 	IR028_3_train,
                                  B028_0_train, B028_1_train, B028_2_train, B028_3_train))
    
    IR014_0_label_all_train = convert_one_hot(IR014_0_label) * IR014_0_train.shape[0]
    IR014_1_label_all_train = convert_one_hot(IR014_1_label) * IR014_1_train.shape[0]
    IR014_2_label_all_train = convert_one_hot(IR014_2_label) * IR014_2_train.shape[0]
    IR014_3_label_all_train = convert_one_hot(IR014_3_label) * IR014_3_train.shape[0]

    B014_0_label_all_train = convert_one_hot(B014_0_label) * B014_0_train.shape[0]
    B014_1_label_all_train = convert_one_hot(B014_1_label) * B014_1_train.shape[0]
    B014_2_label_all_train = convert_one_hot(B014_2_label) * B014_2_train.shape[0]
    B014_3_label_all_train = convert_one_hot(B014_3_label) * B014_3_train.shape[0]

    OR014_6_0_label_all_train = convert_one_hot(OR014_6_0_label) * OR014_6_0_train.shape[0]
    OR014_6_1_label_all_train = convert_one_hot(OR014_6_1_label) * OR014_6_1_train.shape[0]
    OR014_6_2_label_all_train = convert_one_hot(OR014_6_2_label) * OR014_6_2_train.shape[0]
    OR014_6_3_label_all_train = convert_one_hot(OR014_6_3_label) * OR014_6_3_train.shape[0]

    IR028_0_label_all_train = convert_one_hot(IR028_0_label) * IR028_0_train.shape[0]
    IR028_1_label_all_train = convert_one_hot(IR028_1_label) * IR028_1_train.shape[0]
    IR028_2_label_all_train = convert_one_hot(IR028_2_label) * IR028_2_train.shape[0]
    IR028_3_label_all_train = convert_one_hot(IR028_3_label) * IR028_3_train.shape[0]

    B028_0_label_all_train = convert_one_hot(B028_0_label) * B028_0_train.shape[0]
    B028_1_label_all_train = convert_one_hot(B028_1_label) * B028_1_train.shape[0]
    B028_2_label_all_train = convert_one_hot(B028_2_label) * B028_2_train.shape[0]
    B028_3_label_all_train = convert_one_hot(B028_3_label) * B028_3_train.shape[0]
    
    all_labels_14_train = np.concatenate((all_labels_12_train, IR014_0_label_all_train, IR014_1_label_all_train,  IR014_2_label_all_train, IR014_3_label_all_train,
                                          B014_0_label_all_train,    B014_1_label_all_train,    B014_2_label_all_train,    B014_3_label_all_train,
                                          OR014_6_0_label_all_train, OR014_6_1_label_all_train, OR014_6_2_label_all_train, OR014_6_3_label_all_train,
                                          IR028_0_label_all_train,   IR028_1_label_all_train,   IR028_2_label_all_train,   IR028_3_label_all_train,
                                          B028_0_label_all_train,    B028_1_label_all_train,    B028_2_label_all_train,    B028_3_label_all_train))
 
  
    all_data_14_test = np.concatenate((all_data_12_test, IR014_0_test, IR014_1_test, IR014_2_test, IR014_3_test,
                                  B014_0_test, B014_1_test, B014_2_test, B014_3_test,
                                  OR014_6_0_test, OR014_6_1_test, OR014_6_2_test, OR014_6_3_test,
                                  IR028_0_test, 	IR028_1_test, 	IR028_2_test, 	IR028_3_test,
                                  B028_0_test, B028_1_test, B028_2_test, B028_3_test))
    
    IR014_0_label_all_test = convert_one_hot(IR014_0_label) * IR014_0_test.shape[0]
    IR014_1_label_all_test = convert_one_hot(IR014_1_label) * IR014_1_test.shape[0]
    IR014_2_label_all_test = convert_one_hot(IR014_2_label) * IR014_2_test.shape[0]
    IR014_3_label_all_test = convert_one_hot(IR014_3_label) * IR014_3_test.shape[0]

    B014_0_label_all_test = convert_one_hot(B014_0_label) * B014_0_test.shape[0]
    B014_1_label_all_test = convert_one_hot(B014_1_label) * B014_1_test.shape[0]
    B014_2_label_all_test = convert_one_hot(B014_2_label) * B014_2_test.shape[0]
    B014_3_label_all_test = convert_one_hot(B014_3_label) * B014_3_test.shape[0]

    OR014_6_0_label_all_test = convert_one_hot(OR014_6_0_label) * OR014_6_0_test.shape[0]
    OR014_6_1_label_all_test = convert_one_hot(OR014_6_1_label) * OR014_6_1_test.shape[0]
    OR014_6_2_label_all_test = convert_one_hot(OR014_6_2_label) * OR014_6_2_test.shape[0]
    OR014_6_3_label_all_test = convert_one_hot(OR014_6_3_label) * OR014_6_3_test.shape[0]

    IR028_0_label_all_test = convert_one_hot(IR028_0_label) * IR028_0_test.shape[0]
    IR028_1_label_all_test = convert_one_hot(IR028_1_label) * IR028_1_test.shape[0]
    IR028_2_label_all_test = convert_one_hot(IR028_2_label) * IR028_2_test.shape[0]
    IR028_3_label_all_test = convert_one_hot(IR028_3_label) * IR028_3_test.shape[0]

    B028_0_label_all_test = convert_one_hot(B028_0_label) * B028_0_test.shape[0]
    B028_1_label_all_test = convert_one_hot(B028_1_label) * B028_1_test.shape[0]
    B028_2_label_all_test = convert_one_hot(B028_2_label) * B028_2_test.shape[0]
    B028_3_label_all_test = convert_one_hot(B028_3_label) * B028_3_test.shape[0]
    
    all_labels_14_test = np.concatenate((all_labels_12_test,       IR014_0_label_all_test,   IR014_1_label_all_test,  IR014_2_label_all_test, IR014_3_label_all_test,
                                    B014_0_label_all_test,    B014_1_label_all_test,    B014_2_label_all_test,    B014_3_label_all_test,
                                    OR014_6_0_label_all_test, OR014_6_1_label_all_test, OR014_6_2_label_all_test, OR014_6_3_label_all_test,
                                    IR028_0_label_all_test,   IR028_1_label_all_test,   IR028_2_label_all_test,   IR028_3_label_all_test,
                                    B028_0_label_all_test,    B028_1_label_all_test,    B028_2_label_all_test,    B028_3_label_all_test))
    
    X_train, X_test, y_train, y_test = all_data_14_train, all_data_14_test, all_labels_14_train, all_labels_14_test
    
  if opt.MFPT_data:
    from load_data import baseline_1, baseline_2, baseline_3, baseline_1_label, baseline_2_label, baseline_3_label,\
                          OuterRaceFault_1, OuterRaceFault_2, OuterRaceFault_3, OuterRaceFault_1_label, OuterRaceFault_2_label, OuterRaceFault_3_label,\
                          OuterRaceFault_vload_1, OuterRaceFault_vload_2, OuterRaceFault_vload_3, OuterRaceFault_vload_4, OuterRaceFault_vload_5, OuterRaceFault_vload_6, OuterRaceFault_vload_7,\
                          OuterRaceFault_vload_1_label, OuterRaceFault_vload_2_label, OuterRaceFault_vload_3_label, OuterRaceFault_vload_4_label, OuterRaceFault_vload_5_label, OuterRaceFault_vload_6_label, OuterRaceFault_vload_7_label,\
                          InnerRaceFault_vload_1, InnerRaceFault_vload_2, InnerRaceFault_vload_3, InnerRaceFault_vload_4, InnerRaceFault_vload_5, InnerRaceFault_vload_6, InnerRaceFault_vload_7,\
                          InnerRaceFault_vload_1_label, InnerRaceFault_vload_2_label, InnerRaceFault_vload_3_label, InnerRaceFault_vload_4_label, InnerRaceFault_vload_5_label, InnerRaceFault_vload_6_label, InnerRaceFault_vload_7_label
    
    baseline_1_label = convert_one_hot(baseline_1_label) * baseline_1.shape[0]
    baseline_2_label = convert_one_hot(baseline_2_label) * baseline_2.shape[0]
    baseline_3_label = convert_one_hot(baseline_3_label) * baseline_3.shape[0]

    OuterRaceFault_1_label = convert_one_hot(OuterRaceFault_1_label) * OuterRaceFault_1.shape[0]
    OuterRaceFault_2_label = convert_one_hot(OuterRaceFault_2_label) * OuterRaceFault_2.shape[0]
    OuterRaceFault_3_label = convert_one_hot(OuterRaceFault_3_label) * OuterRaceFault_3.shape[0]

    OuterRaceFault_vload_1_label = convert_one_hot(OuterRaceFault_vload_1_label) * OuterRaceFault_vload_1.shape[0]
    OuterRaceFault_vload_2_label = convert_one_hot(OuterRaceFault_vload_2_label) * OuterRaceFault_vload_2.shape[0]
    OuterRaceFault_vload_3_label = convert_one_hot(OuterRaceFault_vload_3_label) * OuterRaceFault_vload_3.shape[0]
    OuterRaceFault_vload_4_label = convert_one_hot(OuterRaceFault_vload_4_label) * OuterRaceFault_vload_4.shape[0]
    OuterRaceFault_vload_5_label = convert_one_hot(OuterRaceFault_vload_5_label) * OuterRaceFault_vload_5.shape[0]
    OuterRaceFault_vload_6_label = convert_one_hot(OuterRaceFault_vload_6_label) * OuterRaceFault_vload_6.shape[0]
    OuterRaceFault_vload_7_label = convert_one_hot(OuterRaceFault_vload_7_label) * OuterRaceFault_vload_7.shape[0]

    InnerRaceFault_vload_1_label = convert_one_hot(InnerRaceFault_vload_1_label) * InnerRaceFault_vload_1.shape[0]
    InnerRaceFault_vload_2_label = convert_one_hot(InnerRaceFault_vload_2_label) * InnerRaceFault_vload_2.shape[0]
    InnerRaceFault_vload_3_label = convert_one_hot(InnerRaceFault_vload_3_label) * InnerRaceFault_vload_3.shape[0]
    InnerRaceFault_vload_4_label = convert_one_hot(InnerRaceFault_vload_4_label) * InnerRaceFault_vload_4.shape[0]
    InnerRaceFault_vload_5_label = convert_one_hot(InnerRaceFault_vload_5_label) * InnerRaceFault_vload_5.shape[0]
    InnerRaceFault_vload_6_label = convert_one_hot(InnerRaceFault_vload_6_label) * InnerRaceFault_vload_6.shape[0]
    InnerRaceFault_vload_7_label = convert_one_hot(InnerRaceFault_vload_7_label) * InnerRaceFault_vload_7.shape[0]

    X_train = np.concatenate((baseline_1, baseline_2, OuterRaceFault_1, OuterRaceFault_2, OuterRaceFault_vload_1, OuterRaceFault_vload_2, OuterRaceFault_vload_4, OuterRaceFault_vload_5, OuterRaceFault_vload_7, InnerRaceFault_vload_1, InnerRaceFault_vload_2, InnerRaceFault_vload_4, InnerRaceFault_vload_5, InnerRaceFault_vload_7))
    y_train = np.concatenate((baseline_1_label, baseline_2_label, OuterRaceFault_1_label, OuterRaceFault_2_label, OuterRaceFault_vload_1_label, OuterRaceFault_vload_2_label, OuterRaceFault_vload_4_label, OuterRaceFault_vload_5_label, OuterRaceFault_vload_7_label, InnerRaceFault_vload_1_label, InnerRaceFault_vload_2_label, InnerRaceFault_vload_4_label, InnerRaceFault_vload_5_label, InnerRaceFault_vload_7_label))
    X_test = np.concatenate((baseline_3, baseline_2, OuterRaceFault_3, OuterRaceFault_vload_3, OuterRaceFault_vload_6, InnerRaceFault_vload_3, InnerRaceFault_vload_6))
    y_test = np.concatenate((baseline_3_label, baseline_2_label, OuterRaceFault_3_label, OuterRaceFault_vload_3_label, OuterRaceFault_vload_6_label, InnerRaceFault_vload_3_label, InnerRaceFault_vload_6_label))
  
  if opt.PU_data_table_8:
    from load_data import Healthy_train, Healthy_train_label, OR_Damage_train, OR_Damage_train_label, IR_Damage_train, IR_Damage_train_label,\
                          Healthy_test, Healthy_test_label, OR_Damage_test, OR_Damage_test_label, IR_Damage_test, IR_Damage_test_label
    
    Healthy_train_label = convert_one_hot(Healthy_train_label) * Healthy_train.shape[0]
    OR_Damage_train_label = convert_one_hot(OR_Damage_train_label) * OR_Damage_train.shape[0]
    IR_Damage_train_label = convert_one_hot(IR_Damage_train_label) * IR_Damage_train.shape[0]

    Healthy_test_label = convert_one_hot(Healthy_test_label) * Healthy_test.shape[0]
    OR_Damage_test_label = convert_one_hot(OR_Damage_test_label) * OR_Damage_test.shape[0]
    IR_Damage_test_label = convert_one_hot(IR_Damage_test_label) * IR_Damage_test.shape[0]

    X_train = np.concatenate((choosing_features(Healthy_train), choosing_features(OR_Damage_train), choosing_features(IR_Damage_train)))
    y_train = np.concatenate((Healthy_train_label, OR_Damage_train_label, IR_Damage_train_label))
    X_test = np.concatenate((Healthy_test, OR_Damage_test, IR_Damage_test))
    y_test = np.concatenate((Healthy_test_label, OR_Damage_test_label, IR_Damage_test_label))
  return X_train, X_test, y_train, y_test
