import numpy as np
import pandas as pd
import scipy.io

'''
For all files, the following item in the variable name indicates:
    DE - drive end accelerometer data
    FE - fan end accelerometer data
    BA - base accelerometer data
    time - time series data
    RPM - rpm during testing

    Fault Diameter: 0.007"
    Motor Load (HP): 0
    Approx. Motor Speed (rpm): 1797
'''
num = 124600
n   = 200
Normal_0 = scipy.io.loadmat('./data/Normal_0.mat')
B007_0 = scipy.io.loadmat('./data/B007_0.mat')
IR007_0 = scipy.io.loadmat('./data/IR007_0.mat')
OR007_3_0 = scipy.io.loadmat('./data/OR007_3_0.mat')
OR007_6_0 = scipy.io.loadmat('./data/OR007_6_0.mat')
OR007_12_0 = scipy.io.loadmat('./data/OR007_12_0.mat')
all_labels = {0: 'Normal_0', 1: 'B007_0', 2: 'IR007_0', 3: 'OR007_3_0', 4: 'OR007_6_0', 5: 'OR007_12_0'}

Normal_0_X097_DE_time = Normal_0['X097_DE_time'][:num]
Normal_0_X097_FE_time = Normal_0['X097_FE_time'][:num]
Normal_0_X097RPM      = Normal_0['X097RPM']
Normal_0_group = np.concatenate((Normal_0_X097_DE_time.reshape(1, num), Normal_0_X097_FE_time.reshape(1, num)), axis=0)
Normal_0_name  = [[1, 0, 0, 0, 0, 0]]*n
Normal_0_reshape = Normal_0_group.reshape(n, int(num/n)*2)


B007_0_X122_DE_time = B007_0['X122_DE_time'][:num]
B007_0_X122_FE_time = B007_0['X122_FE_time'][:num]
B007_0_X122RPM      = B007_0['X122RPM']
B007_0_group = np.concatenate((B007_0_X122_DE_time.reshape(1, num), B007_0_X122_FE_time.reshape(1, num)), axis=0)
B007_0_name  = [[0, 1, 0, 0, 0, 0]]*n
B007_0_reshape = B007_0_group.reshape(n, int(num/n)*2)


IR007_0_X122_DE_time = IR007_0['X109_DE_time'][:num]
IR007_0_X122_FE_time = IR007_0['X109_FE_time'][:num]
IR007_0_X122RPM      = IR007_0['X109RPM']
IR007_0_group = np.concatenate((IR007_0_X122_DE_time.reshape(1, num), IR007_0_X122_FE_time.reshape(1, num)), axis=0)
IR007_0_name  = [[0, 0, 1, 0, 0, 0]]*n
IR007_0_reshape = IR007_0_group.reshape(n, int(num/n)*2)


OR007_3_0_X122_DE_time = OR007_3_0['X148_DE_time'][:num]
OR007_3_0_X122_FE_time = OR007_3_0['X148_FE_time'][:num]
OR007_3_0_X122RPM      = OR007_3_0['X148RPM']
OR007_3_0_group = np.concatenate((OR007_3_0_X122_DE_time.reshape(1, num), OR007_3_0_X122_FE_time.reshape(1, num)), axis=0)
OR007_3_0_name  = [[0, 0, 0, 1, 0, 0]]*n
OR007_3_0_reshape = OR007_3_0_group.reshape(n, int(num/n)*2)


OR007_6_0_X122_DE_time = OR007_6_0['X135_DE_time'][:num]
OR007_6_0_X122_FE_time = OR007_6_0['X135_FE_time'][:num]
OR007_6_0_X122RPM      = OR007_6_0['X135RPM']
OR007_6_0_group = np.concatenate((OR007_6_0_X122_DE_time.reshape(1, num), OR007_6_0_X122_FE_time.reshape(1, num)), axis=0)
OR007_6_0_name  = [[0, 0, 0, 0, 1, 0]]*n
OR007_6_0_reshape = OR007_6_0_group.reshape(n, int(num/n)*2)


OR007_12_0_X122_DE_time = OR007_12_0['X161_DE_time'][:num]
OR007_12_0_X122_FE_time = OR007_12_0['X161_FE_time'][:num]
OR007_12_0_X122RPM      = OR007_12_0['X161RPM']
OR007_12_0_group = np.concatenate((OR007_12_0_X122_DE_time.reshape(1, num), OR007_12_0_X122_FE_time.reshape(1, num)), axis=0)
OR007_12_0_name  = [[0, 0, 0, 0, 0, 1]]*n
OR007_12_0_reshape = OR007_12_0_group.reshape(n, int(num/n)*2)

DE_time = np.concatenate((Normal_0_X097_DE_time.reshape(n, int(num/n)), B007_0_X122_DE_time.reshape(n, int(num/n)), IR007_0_X122_DE_time.reshape(n, int(num/n)), OR007_3_0_X122_DE_time.reshape(n, int(num/n)), OR007_6_0_X122_DE_time.reshape(n, int(num/n)), OR007_12_0_X122_DE_time.reshape(n, int(num/n))))
FE_time = np.concatenate((Normal_0_X097_FE_time.reshape(n, int(num/n)), B007_0_X122_FE_time.reshape(n, int(num/n)), IR007_0_X122_FE_time.reshape(n, int(num/n)), OR007_3_0_X122_FE_time.reshape(n, int(num/n)), OR007_6_0_X122_FE_time.reshape(n, int(num/n)), OR007_12_0_X122_FE_time.reshape(n, int(num/n))))

merge_data = np.concatenate((DE_time, FE_time), axis=1)
label = np.concatenate((Normal_0_name, B007_0_name, IR007_0_name, OR007_3_0_name, OR007_6_0_name, OR007_12_0_name))
# print(merge_data[:, :623].shape)
