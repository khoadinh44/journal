# Denoise signal with Fourier
import numpy as np
import matplotlib.pyplot as plt
import pywt
from matplotlib.image import imread


def Fourier(f, num, plot_all=False, get_result=False, get_PSD=False, thres=10):
  t = np.linspace(0, 1, num=num).astype(np.float32)
  f = f.reshape(num, )

  ## Compute the Fast Fourier Transform (FFT)
  fhat = np.fft.fft(f, num)                     # Compute the FFT
  PSD = fhat * np.conj(fhat) / num             # Power spectrum (power per freq)
  L = np.arange(0, np.floor(num/2),dtype='int') # Only plot the first half of freqs
  freq = np.arange(num)

  ## Use the PSD to filter out noise
  indices = PSD > thres       # Find all freqs with large power
  PSDclean = PSD * indices  # Zero out all others
  fhat = indices * fhat     # Zero out small Fourier coeffs. in Y
  ffilt = np.fft.ifft(fhat) # Inverse FFT for filtered time signal

  if get_PSD:
    return PSD.real 
    # plt.plot(PSD.real)
    # plt.title('Noise')
    # plt.show()
    # plt.rcParams["figure.figsize"] = (20,3)
    # plt.savefig('plot_noise.png')
  if plot_all:
    fig,axs = plt.subplots(3,1)

    plt.sca(axs[0])
    plt.plot(t, f, color='r', LineWidth=1.5, label='Noisy')
    plt.xlim(t[0], t[-1])
    plt.legend()

    plt.sca(axs[1])
    plt.plot(t, ffilt, color='b', LineWidth=2, label='Filtered')
    plt.xlim(t[0], t[-1])
    plt.legend()

    plt.sca(axs[2])
    plt.plot(freq[L], PSD[L], color='r', LineWidth=2, label='Noisy')
    plt.plot(freq[L], PSDclean[L], color='b', LineWidth=1.5, label='Filtered')
    plt.xlim(freq[L[0]], freq[L[-1]])
    plt.legend()

    plt.show()
    plt.rcParams["figure.figsize"] = (20,3)
    plt.savefig('plot_all.png')
  
  if get_result:
    return ffilt.real

def SVD_denoise(Xnoisy):
    sigma = 1
    U, S, VT = np.linalg.svd(Xnoisy, full_matrices=0)
    N = Xnoisy.shape[1]
    cutoff = (4/np.sqrt(3)) * np.sqrt(N) * sigma # Hard threshold
    r = np.max(np.where(S > cutoff)) # Keep modes w/ sig > cutoff 

    Xclean = U[:,:(r+1)] @ np.diag(S[:(r+1)]) @ VT[:(r+1),:]
    return Xclean

def Wavelet(X):
    '''
    n = 124600
    [0]: (31150, 1)

    [1][0]: (31150, 1)
    [1][1]: (31150, 1)
    [1][2]: (31150, 1)

    [2][0]: (62300, 1)
    [2][1]: (62300, 1)
    [2][2]: (62300, 1)
    '''
    n = 2
    w = 'db1' #
    coeffs = pywt.wavedec2(X, wavelet=w, level=n)

    coeffs_0 = coeffs[0]
    coeffs_1 = np.concatenate((coeffs[1][0], coeffs[1][1], coeffs[1][2]), axis=1)
    coeffs_2 = np.concatenate((coeffs[2][0], coeffs[2][1], coeffs[2][2]), axis=1)
    return coeffs_0, coeffs_1, coeffs_2

def Wavelet_denoise(X):
  # Create wavelet object and define parameters
  X = X.reshape(int(X.shape[0]),)
  w = pywt.Wavelet('sym4')
  maxlev = 2 # Override if desired
  print("maximum level is " + str(maxlev))
  threshold = 0.04 # Threshold for filtering

  # Decompose into wavelet components, to the level selected:
  coeffs = pywt.wavedec(X, 'sym4', level=maxlev)

  #cA = pywt.threshold(cA, threshold*max(cA))
  for i in range(1, len(coeffs)):
      coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))

  datarec = pywt.waverec(coeffs, 'sym4')
  return datarec
