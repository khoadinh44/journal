from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

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
                          OR0014_6_0, OR0014_6_0_label, OR0014_6_1, OR0014_6_1_label
    
    
  
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    
    # Models and denoising methods--------------------------
    parser.add_argument('--data_normal', default=True, type=bool)
    parser.add_argument('--data_12k', default=True, type=bool)
    parser.add_argument('--data_48k', default=False, type=bool)    
    # Parameters---------------------------------------------
    parser.add_argument('--save',       type=str,   default='model.h5', help='Position to save weights')
    parser.add_argument('--epochs',     type=int,   default=100,        help='Number of iterations for training')
    parser.add_argument('--batch_size', type=int,   default=32,         help='Number of batch size for training')
    parser.add_argument('--test_rate',  type=float, default=0.25,       help='rate of split data for testing')
    parser.add_argument('--use_type',   type=str,   default=None,       help='types of NN: use_CNN_A')
    
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

    
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
