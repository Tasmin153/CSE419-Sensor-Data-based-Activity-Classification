import os
import io
import argparse



def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('-gpu-id', type=int, default=0)
    parser.add_argument('-data-dir',type=str,default='./data')
    parser.add_argument('-epoch', type=int, default=100)
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-use-cuda', default=True, action='store_true')

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    print(args.epoch)
