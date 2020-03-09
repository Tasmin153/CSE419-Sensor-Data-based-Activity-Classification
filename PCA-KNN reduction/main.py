import os
import io
import argparse
import pickle

import numpy as np

from utils import folder_finder, dataset_info, file_train_test_wise_info
from data_preprocessing import data_loader, full_dataset
from models import model_init
from train import train_model, test_model, model_evalution
from plotter import input_data_plot

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pylab import figure, axes, pie, title, show


import sklearn 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from datetime import datetime

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
                  KNeighborsClassifier(3),
                  SVC(kernel="linear", C=0.025),
                  SVC(gamma=2, C=1),
                  GaussianProcessClassifier(1.0 * RBF(1.0)),
                  DecisionTreeClassifier(max_depth=5),
                  RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
                  MLPClassifier(alpha=1, max_iter=1000),
                  AdaBoostClassifier(),
                  GaussianNB(),
                  QuadraticDiscriminantAnalysis()]

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
    # for i in range(2):

        X_train, X_test, y_train, y_test,xx,yy = data_loader(file_list[i])
        file_train_test_wise_info(file_list[i],y_train,y_test)
        
        model = train_model(model_arch, X_train, y_train)
        #pickle.dump(model, open(save_model_name, 'wb'))

        x_test_bulk.append(X_test)
        y_test_bulk.append(y_test)
        #model = model_init()

    input_data_plot(X_train,X_test,xx,y_test,y_train,yy)


    # #dataset_info(file_list)

    for i in range(len(y_test_bulk)):
        #model = pickle.load(open(save_model_name, 'rb'))
        pred_tree = test_model(model, x_test_bulk[i])
        model_evalution(y_test_bulk[i],pred_tree)


    



















