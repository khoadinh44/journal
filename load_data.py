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
Normal_0 = scipy.io.loadmat('Normal_0.mat')
B007_0 = scipy.io.loadmat('B007_0.mat')
IR007_0 = scipy.io.loadmat('IR007_0.mat')
OR007_3_0 = scipy.io.loadmat('OR007_3_0.mat')
OR007_6_0 = scipy.io.loadmat('OR007_6_0.mat')
OR007_12_0 = scipy.io.loadmat('OR007_12_0.mat')

num = 124602
Normal_0_X097_DE_time = Normal_0['X097_DE_time'][:num]
Normal_0_X097_FE_time = Normal_0['X097_FE_time'][:num]
Normal_0_X097RPM      = Normal_0['X097RPM']
Normal_0_group = np.concatenate((Normal_0_X097_DE_time.reshape(1, num), Normal_0_X097_FE_time.reshape(1, num)), axis=0)
Normal_0_name  = np.array(['X097_DE_time', 'X097_FE_time']).reshape(2, 1)

B007_0_X122_DE_time = B007_0['X122_DE_time'][:num]
B007_0_X122_FE_time = B007_0['X122_FE_time'][:num]
B007_0_X122RPM      = B007_0['X122RPM']
B007_0_group = np.concatenate((B007_0_X122_DE_time.reshape(1, num), B007_0_X122_FE_time.reshape(1, num)), axis=0)
B007_0_name  = np.array(['X122_DE_time', 'X122_FE_time']).reshape(2, 1)

IR007_0_X122_DE_time = IR007_0['X109_DE_time'][:num]
IR007_0_X122_FE_time = IR007_0['X109_FE_time'][:num]
IR007_0_X122RPM      = IR007_0['X109RPM']
IR007_0_group = np.concatenate((IR007_0_X122_DE_time.reshape(1, num), IR007_0_X122_FE_time.reshape(1, num)), axis=0)
IR007_0_name  = np.array(['X109_DE_time', 'X109_FE_time']).reshape(2, 1)

OR007_3_0_X122_DE_time = OR007_3_0['X148_DE_time'][:num]
OR007_3_0_X122_FE_time = OR007_3_0['X148_FE_time'][:num]
OR007_3_0_X122RPM      = OR007_3_0['X148RPM']
OR007_3_0_group = np.concatenate((OR007_3_0_X122_DE_time.reshape(1, num), OR007_3_0_X122_FE_time.reshape(1, num)), axis=0)
OR007_3_0_name  = np.array(['X148_DE_time', 'X148_FE_time']).reshape(2, 1)

OR007_6_0_X122_DE_time = OR007_6_0['X135_DE_time'][:num]
OR007_6_0_X122_FE_time = OR007_6_0['X135_FE_time'][:num]
OR007_6_0_X122RPM      = OR007_6_0['X135RPM']
OR007_6_0_group = np.concatenate((OR007_6_0_X122_DE_time.reshape(1, num), OR007_6_0_X122_FE_time.reshape(1, num)), axis=0)
OR007_6_0_name  = np.array(['X135_DE_time', 'X135_FE_time']).reshape(2, 1)

OR007_12_0_X122_DE_time = OR007_12_0['X161_DE_time'][:num]
OR007_12_0_X122_FE_time = OR007_12_0['X161_FE_time'][:num]
OR007_12_0_X122RPM      = OR007_12_0['X161RPM']
OR007_12_0_group = np.concatenate((OR007_12_0_X122_DE_time.reshape(1, num), OR007_12_0_X122_FE_time.reshape(1, num)), axis=0)
OR007_12_0_name  = np.array(['X161_DE_time', 'X161_FE_time']).reshape(2, 1)
