# Denoise signal with Fourier
import numpy as np
import matplotlib.pyplot as plt


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
