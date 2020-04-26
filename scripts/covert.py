import numpy as np
parser = argparse.ArgumentParser(description='convert npy to csv')
parser.add_argument('--input', type=str, default='13.Zeisel_LTMG_0.1_features.npy',
                    help='input')
parser.add_argument('--output', type=str, default='13.csv',
                    help='output')

args = parser.parse_args()

t=np.load(args.input,allow_pickle=True)
t=t.tolist()
t1=t.todense()
np.savetxt(args.output,t1, delimiter=',',fmt='%d')

