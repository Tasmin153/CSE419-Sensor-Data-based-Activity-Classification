import os
import io
import argparse


from utils import folder_finder
from data_preprocessing import dataset
from models import model_init
from train import train_model, test_model, model_evalution



def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('-gpu-id', type=int, default=0)

    # dataset
    parser.add_argument('-data_dir',type=str,default='../Dataset/')

    parser.add_argument('-epoch', type=int, default=100)
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-use-cuda', default=True, action='store_true')

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    file_list = folder_finder(args.data_dir)

    # print('Epoch: ', args.epoch)
    # print('lr: ', args.lr)
    # print('Dataset Path: ', args.data_dir)
    print('Total files:',len(file_list))


    model = model_init()

    for i in range(len(file_list)):

        X_train, X_test, y_train, y_test = dataset(file_list[i])
        
        print("File: ",file_list[i])
        print("X_train: ", X_train.shape)
        print("y_train: ", y_train.shape)

        print("X_test: ", X_test.shape)
        print("y_test: ", y_test.shape)

        clf_tree = train_model(model, X_train, y_train)
        pred_tree = test_model(clf_tree, X_test)

        model_evalution(y_test,pred_tree)













