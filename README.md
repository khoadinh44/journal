# Journal 1

NOTE:
use_DNN_A, use_DNN_B, use_CNN_A, use_CNN_B are represented for DNN A, DNN B, CNN A, CNN B model
<!-- if the DNN A is choosen: -->
    --use_DNN_B True
    
DFK, Wavelet_denoise, SVD, savitzky_golay, None are represented for denoising methods. In those methods, DFK is our proposal
<!-- if DFK is choosen: -->
    --denoise DFK
    
Training with 7 ls diameter faults data:
    %cd /signal_machine
    !python train.py --use_DNN_B True --denoise DFK
