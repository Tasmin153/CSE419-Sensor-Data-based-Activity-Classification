import os
import io
import argparse
import pickle

from utils import folder_finder
from data_preprocessing import data_loader
from models import model_init
from train import train_model, test_model, model_evalution

from datetime import datetime

def get_args():

    parser = argparse.ArgumentParser()

    # parser.add_argument('-gpu-id', type=int, default=0)

    # dataset
    parser.add_argument('-data_dir',type=str,default='../Dataset/')
    parser.add_argument('-pretrain',type=bool,default=False)

    # parser.add_argument('-epoch', type=int, default=100)
    # parser.add_argument('-lr', type=float, default=0.001)
    # parser.add_argument('-use-cuda', default=True, action='store_true')

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    file_list = folder_finder(args.data_dir)

    # print('Epoch: ', args.epoch)
    # print('lr: ', args.lr)
    # print('Dataset Path: ', args.data_dir)
    print(file_list)
    print('Total files:',len(file_list))


    model_arch = model_init()
    save_model_name = "pretrain/"+datetime.now().strftime('time_%H_%M_%S__date_%Y-%m-%d')+".pkl"

    # if args.pretrain:
    #     print('Loading pretrain model...')
    #     model = pickle.load(open(save_model_name, 'rb'))

    x_test_bulk = []
    y_test_bulk = []

    for i in range(len(file_list)):

        X_train, X_test, y_train, y_test = data_loader(file_list[i])

        model = train_model(model_arch, X_train, y_train)
        pickle.dump(model, open(save_model_name, 'wb'))

        x_test_bulk.append(X_test)
        y_test_bulk.append(y_test)
        # pred_tree = test_model(model, X_test)
        # model_evalution(y_test,pred_tree)
        #dataset_info(file_list)

    for i in range(len(y_test_bulk)):
        model = pickle.load(open(save_model_name, 'rb'))
        pred_tree = test_model(model, x_test_bulk[i])
        model_evalution(y_test_bulk[i],pred_tree)















