import os
import io
import argparse

import pickle
from datetime import datetime
from data_preprocessing import dataset
from train import test_model, model_evalution
from utils import folder_finder

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir',type=str,default='../Dataset/')
    parser.add_argument('-weights',type=str,default='pretrain/time_15_04_27__date_2019-12-23.pkl')
    return parser.parse_args()


if __name__ == '__main__':

	args = get_args()
	file_list = folder_finder(args.data_dir)
	weight_path = args.weights

	for i in range(len(file_list)):
		X_train, X_test, y_train, y_test = dataset(file_list[i])
		model = pickle.load(open(weight_path, 'rb'))
		pred_tree = test_model(model, X_test)
		model_evalution(y_test,pred_tree)