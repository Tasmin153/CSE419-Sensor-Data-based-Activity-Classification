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


     # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(file_list), len(classifiers) + 1, i)
    
    ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')
    # Plot the testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
               edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(file_list), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        Z = xx

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        #else:
            #Z = clf.predict_proba(Xpred)

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # Plot the testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   edgecolors='k', alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        
        ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1

    plt.tight_layout()
    plt.show()


    # #dataset_info(file_list)

    for i in range(len(y_test_bulk)):
        #model = pickle.load(open(save_model_name, 'rb'))
        pred_tree = test_model(model, x_test_bulk[i])
        model_evalution(y_test_bulk[i],pred_tree)


    



















