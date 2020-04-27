import numpy as np
import argparse

parser = argparse.ArgumentParser(description='convert npy to csv')
parser.add_argument('--input', type=str, default='13.Zeisel_LTMG_0.1_features.npy',
                    help='input')
parser.add_argument('--output', type=str, default='13.csv',
                    help='output')

args = parser.parse_args()

t=np.load(args.input,allow_pickle=True)
# np.savetxt(args.output,t, delimiter=',',fmt='%d')
np.savetxt(args.output,t, delimiter=',',fmt='%f')

