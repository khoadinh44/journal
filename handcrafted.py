from sklearn.model_selection import train_test_split
from preprocessing.utils import convert_one_hot

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
import argparse
import numpy as np

def main(opt):
  if opt.data_normal:
    from load_data import Normal_0, Normal_1, Normal_2, Normal_3,\
                          Normal_0_label, Normal_1_label, Normal_2_label, Normal_3_label
  if opt.data_12k:
    from load_data import B007_0, B007_0_label, B007_1, B007_1_label, B007_2, B007_2_label, B007_3, B007_3_label,\
                          B014_0, B014_0_label, B014_1, B014_1_label, B014_2, B014_2_label, B014_3, B014_3_label,\
                          B021_0, B021_0_label, B021_1, B021_1_label, B021_2, B021_2_label, B021_3, B021_3_label,\
                          B028_0, B028_0_label, B028_1, B028_1_label, B028_2, B028_2_label, B028_3, B028_3_label,\
                          IR007_0, IR007_0_label, IR007_1, IR007_1_label, IR007_2, IR007_2_label, IR007_3, IR007_3_label,\
                          IR014_0, IR014_0_label, IR014_1, IR014_1_label, IR014_2, IR014_2_label, IR014_3, IR014_3_label,\
                          IR021_0, IR021_0_label, IR021_1, IR021_1_label, IR021_2, IR021_2_label, IR021_3, IR021_3_label,\
                          IR028_0, IR028_0_label, IR028_1, IR028_1_label, IR028_2, IR028_2_label, IR028_3, IR028_3_label,\
                          OR007_12_0, OR007_12_0_label, OR007_12_1, OR007_12_1_label, OR007_12_2, OR007_12_2_label, OR007_12_3, OR007_12_3_label,\
                          OR007_3_0, OR007_3_0_label, OR007_3_1, OR007_3_1_label, OR007_3_2, OR007_3_2_label, OR007_3_3, OR007_3_3_label,\
                          OR007_6_0, OR007_6_0_label, OR007_6_1, OR007_6_1_label, OR007_6_2, OR007_6_2_label, OR007_6_3, OR007_6_3_label,\
                          OR0014_6_0, OR0014_6_0_label, OR0014_6_1, OR0014_6_1_label, OR0014_6_2, OR0014_6_2_label, OR0014_6_3, OR0014_6_3_label,\
                          OR0021_6_0, OR0021_6_0_label, OR0021_6_1, OR0021_6_1_label, OR0021_6_2, OR0021_6_2_label, OR0021_6_3, OR0021_6_3_label,\
                          OR0021_3_0, OR0021_3_0_label, OR0021_3_1, OR0021_3_1_label, OR0021_3_2, OR0021_3_2_label, OR0021_3_3, OR0021_3_3_label,\
                          OR0021_12_0, OR0021_12_0_label, OR0021_12_1, OR0021_12_1_label, OR0021_12_2, OR0021_12_2_label, OR0021_12_3, OR0021_12_3_label
  
  if opt.data_48k:
    from load_data import B007_0, B007_0_label, B007_1, B007_1_label, B007_2, B007_2_label, B007_3, B007_3_label,\
                          IR007_0, IR007_0_label, IR007_1, IR007_1_label, IR007_2, IR007_2_label, IR007_3, IR007_3_label,\
                          OR007_12_0, OR007_12_0_label, OR007_12_1, OR007_12_1_label, OR007_12_2, OR007_12_2_label, OR007_12_3, OR007_12_3_label,\
                          OR007_3_0, OR007_3_0_label, OR007_3_1, OR007_3_1_label, OR007_3_2, OR007_3_2_label, OR007_3_3, OR007_3_3_label,\
                          OR007_6_0, OR007_6_0_label, OR007_6_1, OR007_6_1_label, OR007_6_2, OR007_6_2_label, OR007_6_3, OR007_6_3_label

  if opt.case_0:
    all_data_0 = np.concatenate((Normal_0, IR007_0, B007_0, OR007_6_0, OR007_3_0, OR007_12_0))
    Normal_0_label_all = convert_one_hot(Normal_0_label) * Normal_0.shape[0]
    IR007_0_label_all = convert_one_hot(IR007_0_label) * IR007_0.shape[0]
    B007_0_label_all = convert_one_hot(B007_0_label) * B007_0.shape[0]
    OR007_6_0_label_all = convert_one_hot(OR007_6_0_label) * OR007_6_0.shape[0]
    OR007_3_0_label_all = convert_one_hot(OR007_3_0_label) * OR007_3_0.shape[0]
    OR007_12_0_label_all = convert_one_hot(OR007_12_0_label) * OR007_12_0.shape[0]
    all_labels_0 = np.concatenate((Normal_0_label_all, IR007_0_label_all, B007_0_label_all, OR007_6_0_label_all, OR007_3_0_label_all, OR007_12_0_label_all))
    X_train, X_test, y_train, y_test = train_test_split(all_data_0, all_labels_0, test_size=opt.test_rate, random_state=42)

  if opt.case_1:
    all_data_1 = np.concatenate((Normal_1, IR007_1, B007_1, OR007_6_1, OR007_3_1, OR007_12_1))
    Normal_1_label_all = convert_one_hot(Normal_1_label) * Normal_1.shape[0]
    IR007_1_label_all = convert_one_hot(IR007_1_label) * IR007_1.shape[0]
    B007_1_label_all = convert_one_hot(B007_1_label) * B007_1.shape[0]
    OR007_6_1_label_all = convert_one_hot(OR007_6_1_label) * OR007_6_1.shape[0]
    OR007_3_1_label_all = convert_one_hot(OR007_3_1_label) * OR007_3_1.shape[0]
    OR007_12_1_label_all = convert_one_hot(OR007_12_1_label) * OR007_12_1.shape[0]
    all_labels_1 = np.concatenate((Normal_1_label_all, IR007_1_label_all, B007_1_label_all, OR007_6_1_label_all, OR007_3_1_label_all, OR007_12_1_label_all))
    X_train, X_test, y_train, y_test = train_test_split(all_data_1, all_labels_1, test_size=opt.test_rate, random_state=42)

  if opt.case_2:
    all_data_2 = np.concatenate((Normal_2, IR007_2, B007_2, OR007_6_2, OR007_3_2, OR007_12_2))
    Normal_2_label_all = convert_one_hot(Normal_2_label) * Normal_2.shape[0]
    IR007_2_label_all = convert_one_hot(IR007_2_label) * IR007_2.shape[0]
    B007_2_label_all = convert_one_hot(B007_2_label) * B007_2.shape[0]
    OR007_6_2_label_all = convert_one_hot(OR007_6_2_label) * OR007_6_2.shape[0]
    OR007_3_2_label_all = convert_one_hot(OR007_3_2_label) * OR007_3_2.shape[0]
    OR007_12_2_label_all = convert_one_hot(OR007_12_2_label) * OR007_12_2.shape[0]
    all_labels_2 = np.concatenate((Normal_2_label_all, IR007_2_label_all, B007_2_label_all, OR007_6_2_label_all, OR007_3_2_label_all, OR007_12_2_label_all))
    X_train, X_test, y_train, y_test = train_test_split(all_data_2, all_labels_2, test_size=opt.test_rate, random_state=42)

  if opt.case_3:
    all_data_3 = np.concatenate((Normal_3, IR007_3, B007_3, OR007_6_3, OR007_3_3, OR007_12_3))
    Normal_3_label_all = convert_one_hot(Normal_3_label) * Normal_3.shape[0]
    IR007_3_label_all = convert_one_hot(IR007_3_label) * IR007_3.shape[0]
    B007_3_label_all = convert_one_hot(B007_3_label) * B007_3.shape[0]
    OR007_6_3_label_all = convert_one_hot(OR007_6_3_label) * OR007_6_3.shape[0]
    OR007_3_3_label_all = convert_one_hot(OR007_3_3_label) * OR007_3_3.shape[0]
    OR007_12_3_label_all = convert_one_hot(OR007_12_3_label) * OR007_12_3.shape[0]
    all_labels_3 = np.concatenate((Normal_3_label_all, IR007_3_label_all, B007_3_label_all, OR007_6_3_label_all, OR007_3_3_label_all, OR007_12_3_label_all))
    X_train, X_test, y_train, y_test = train_test_split(all_data_3, all_labels_3, test_size=opt.test_rate, random_state=42)

  if opt.case_4:
    all_data_4 = np.concatenate((all_data_0, all_data_1, all_data_2, all_data_3))
    all_labels_4 = np.concatenate((all_labels_0, all_labels_1, all_labels_2, all_labels_3))
    X_train, X_test, y_train, y_test = train_test_split(all_data_4, all_labels_4, test_size=opt.test_rate, random_state=42)

  if opt.case_5:
    Normal_5 = np.concatenate((Normal_0, Normal_1, Normal_2, Normal_3))
    IR007_5 = np.concatenate((IR007_0, IR007_1, IR007_2, IR007_3))
    B007_5 = np.concatenate((B007_0, B007_1, B007_2, B007_3))
    OR007_6_5 = np.concatenate((OR007_6_0, OR007_6_1, OR007_6_2, OR007_6_3))
    OR007_3_5 = np.concatenate((OR007_3_0, OR007_3_1, OR007_3_2, OR007_3_3))
    OR007_12_5 = np.concatenate((OR007_12_0, OR007_12_1, OR007_12_2, OR007_12_3))

    all_data = np.concatenate((Normal_5, IR007_5, B007_5, OR007_6_5, OR007_3_5, OR007_12_5))
    Normal_5_label_all = convert_one_hot(Normal_5_label) * Normal_5.shape[0]
    IR007_5_label_all = convert_one_hot(IR007_5_label) * IR007_5.shape[0]
    B007_5_label_all = convert_one_hot(B007_5_label) * B007_5.shape[0]
    OR007_6_5_label_all = convert_one_hot(OR007_6_5_label) * OR007_6_5.shape[0]
    OR007_3_5_label_all = convert_one_hot(OR007_3_5_label) * OR007_3_5.shape[0]
    OR007_12_5_label_all = convert_one_hot(OR007_12_5_label) * OR007_12_5.shape[0]
    all_labels_5 = np.concatenate((Normal_5_label_all, IR007_5_label_all, B007_5_label_all, OR007_6_5_label_all, OR007_3_5_label_all, OR007_12_5_label_all))

    X_train, X_test, y_train, y_test = train_test_split(all_data_5, all_labels_5, test_size=opt.test_rate, random_state=42)
  
  # model = RandomForestClassifier(n_estimators= 300, max_features = "sqrt", n_jobs = -1, random_state = 38)
  # model = LogisticRegression(random_state=1)
  model = SVC(kernel='rbf', probability=True)
  # Train the model
  model.fit(X_train, y_train)
  
  test_predictions = model.predict(X_test)
  print("Accuracy:", accuracy_score(y_test, test_predictions))
  
def parse_opt(known=False):
    parser = argparse.ArgumentParser()

    # Run case------------------------------------------------
    parser.add_argument('--case_0', default=True, type=bool)
    parser.add_argument('--case_1', default=False, type=bool)
    parser.add_argument('--case_2', default=False, type=bool)
    parser.add_argument('--case_3', default=False, type=bool)
    parser.add_argument('--case_4', default=False, type=bool)
    parser.add_argument('--case_5', default=False, type=bool)
    parser.add_argument('--case_6', default=False, type=bool)
    parser.add_argument('--case_7', default=False, type=bool)
    parser.add_argument('--case_8', default=False, type=bool)
    parser.add_argument('--case_9', default=False, type=bool)

    parser.add_argument('--data_normal', default=True, type=bool)
    parser.add_argument('--data_12k', default=False, type=bool)
    parser.add_argument('--data_48k', default=True, type=bool)

    # Parameters---------------------------------------------
    parser.add_argument('--save',       type=str,   default='model.h5', help='Position to save weights')
    parser.add_argument('--epochs',     type=int,   default=100,        help='Number of iterations for training')
    parser.add_argument('--batch_size', type=int,   default=32,         help='Number of batch size for training')
    parser.add_argument('--test_rate',  type=float, default=0.33,       help='rate of split data for testing')
    parser.add_argument('--use_type',   type=str,   default=None,       help='types of NN: use_CNN_A')
    
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

    
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
