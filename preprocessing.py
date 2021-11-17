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
