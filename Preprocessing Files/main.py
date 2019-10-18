import os
import io
import argparse

from data_preprocessing import


def get_args():

    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('-data_dir',type=str,default='./H:/CSE 419/casas-dataset')

    # model 
    parser.add_argument('-gpu-id', type=int, default=0)
    parser.add_argument('-epoch', type=int, default=100)
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-use-cuda', default=True, action='store_true')

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    print(args.data_dir)




