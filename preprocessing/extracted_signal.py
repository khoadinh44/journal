import numpy as np

def RMS(x):
  return np.sqrt(np.mean(x**2))

def Variance(x):
  data_RMS = RMS(x)
  return np.mean((x-data_RMS)**2)

def Peak_value(x):
  return np.max(np.abs(x))

def Crest_factor(x):
  data_PvT = Peak_value(x)
  data_RMS = RMS(x)
  return data_PvT/data_RMS

def Kurtosis(x):
  var = Variance(x)
  return np.sum((x-np.mean(x))**4)/(float(len(x))*var**2)

def Clearance_factor(x):
  data_PvT = Peak_value(x)
  return data_PvT/np.mean(x)**2

def Impulse_factor(x):
  data_PvT = Peak_value(x)
  return data_PvT/np.mean(x)

def shape_factor(x):
  data_RMS = RMS(x)
  return data_RMS/np.mean(x)

def Line_integral(x):
  a = 0.
  for i in range(len(x)-1):
    a += abs(x[i+1]-x[i])
  return a

def Peak_peak_value(x):
  return max(x)-min(x)

def Skewness(x):
  mean = np.mean(x)
  return np.mean((x-mean)**3) / (np.sqrt(np.mean((x-mean)**2)))**3

def extracted_feature_of_signal(signals):
  data = []
  for signal in signals:
    features = [RMS(signal), Variance(signal), Peak_value(signal), Crest_factor(signal), \
                Kurtosis(signal), Clearance_factor(signal), Impulse_factor(signal),  \
                shape_factor(signal), Line_integral(signal), Peak_peak_value(signal), Skewness(signal)]
    data.append(features)
  return np.array(data)
