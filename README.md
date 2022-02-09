# Journal 1

**NOTE:**
- use_DNN_A, use_DNN_B, use_CNN_A, use_CNN_B, use_CNN_C are represented for DNN A, DNN B, CNN A, CNN B model
- DFK, Wavelet_denoise, SVD, savitzky_golay, None are represented for denoising methods. In those methods.\
**DFK** is our proposal, **None** means that nothing denoising method is used to denoise

## For example
### if the CNN A is choosen:
    --use_DNN_B True
    
#### if DFK is choosen:
    --denoise DFK
    
## Training with 7 ls diameter faults data:
    %cd /signal_machine
    !python train.py --use_CNN_C True --denoise DFK

# Run WaveNet
### 1. Clone github:
    !git clone https://github.com/khoadinh44/signal_machine.git

### 2. Go into signal_machine folder:
    %cd /signal_machine
    
### 3. Install library:
    pip install -r requirements.txt
    
### 4. Training WaveNet:
    !python train.py --case_0_6 True --case_1_7 True --case_2_8 True --case_3_9 True --case_4_10 True --case_12 True --case_14 True --data_12k True --use_wavenet True 

### 5. Training WaveNet + Multihead Self Attention:
    !python train.py --case_0_6 True --case_1_7 True --case_2_8 True --case_3_9 True --case_4_10 True --case_12 True --case_14 True --data_12k True --use_wavenet True --multi_head True


