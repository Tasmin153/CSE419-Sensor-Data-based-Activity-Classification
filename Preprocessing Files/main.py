import os
import io
import argparse


from utils import folder_finder
from data_preprocessing import dataset
from models import model_init
from train import train_model



def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('-gpu-id', type=int, default=0)
    parser.add_argument('-data-dir',type=str,default='./data')

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
    print(args.epoch)

    # print('Epoch: ',args.epoch)
    # print('lr: ',args.lr)
    # print(args.data_dir)

    folder_list, file_list = folder_finder(args.data_dir)
    # print('Folder:',len(folder_list))
    # print('Total files:',len(file_list))


    X_train, X_test, y_train, y_test = dataset(file_list[0])
    
    model = model_init()
    train_model(model, X_train, y_train)













