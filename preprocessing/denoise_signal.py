# Denoise signal with Fourier
import numpy as np
import matplotlib.pyplot as plt
import pywt
from matplotlib.image import imread
import numpy as np
from math import factorial
from sklearn.cluster import KMeans

def f_to_mels(f):
    return 2595*np.log10(1+f/700)
def mels_to_f(mels):
    return 700*(np.power(10, mels/2595)-1)

def savitzky_golay(y=None, window_size=None, order=None, deriv=0, rate=1, range_y=None):
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    
    https://gist.github.com/krvajal/1ca6adc7c8ed50f5315fee687d57c3eb
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    y = y.reshape(range_y, )
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def Fourier(f, num, plot_all=False, get_result=False, get_PSD=False, thres=10):
  t = np.linspace(0, 1, num=num).astype(np.float32)
  f = f.reshape(num, )
  f = f_to_mels(f)

  ## Compute the Fast Fourier Transform (FFT)
  fhat = np.fft.fft(f, num)                     # Compute the FFT
  PSD = fhat * np.conj(fhat) / num             # Power spectrum (power per freq)
  L = np.arange(0, np.floor(num/2),dtype='int') # Only plot the first half of freqs
  freq = np.arange(num)

#   # Use the PSD to filter out noise
#   indices = PSD > thres       # Find all freqs with large power
#   PSDclean = PSD * indices  # Zero out all others
#   fhat = indices * fhat     # Zero out small Fourier coeffs. in Y
#   flat = np.where(thres, fhat, 0)
    
  PSD_real = PSD.real.reshape(-1, 1)
  kmeans = KMeans(n_clusters=2, random_state=0).fit(PSD_real)
  thres = np.array(kmeans.labels_).astype(np.int32) > 0.5
  PSDclean = PSD * thres
  fhat = thres * fhat

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
    return mels_to_f(ffilt.real)

def SVD_denoise(Xnoisy):
    m, n = Xnoisy.shape
    if m > n:
        n, m = Xnoisy.shape
        
    belta = m/n
    sigma = 1
    
    lambda_ = np.sqrt(2*(belta+1) + (8*belta)/((belta + 1) + np.sqrt(belta**2 + 14*belta + 1)))
    cutoff = lambda_ * np.sqrt(n) * sigma
    
    U, S, VT = np.linalg.svd(Xnoisy, full_matrices=0)
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
