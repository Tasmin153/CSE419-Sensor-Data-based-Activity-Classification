
import pandas as pd
import numpy as np

from glob import glob
from utils import read_file, string_to_index, pd_to_np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def data_loader(file_path, split=0.3):

    X_train =  X_test =  y_train =  y_test = []
    x = y = []
    pd_data = read_file(file_path)


    feature_list = []

    for i in pd_data:
        feature_list.append(i)

  
    selectData = pd_data.loc[:, feature_list[:-1]]
    activityLabel = pd_data.loc[:, ['activity']]


    x = pd_to_np(selectData)
    y = string_to_index(activityLabel)
    y = np.asarray(y) 
    y = y.astype('int32')

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = split,random_state=42)

    # Feature Scaling
    # sc = StandardScaler()
    # X_train = sc.fit_transform(X_train)
    # X_test = sc.transform(X_test)

    return X_train, X_test, y_train, y_test


def full_dataset(file_list):

    X_train =  X_test =  y_train =  y_test = np.asarray([])
    X_train_temp =  X_test_temp =  y_train_temp =  y_test_temp = []

    # for i in range(len(file_list)):
    for i in range(3):
        X_train_temp, X_test_temp, y_train_temp, y_test_temp = data_loader(file_list[i])

        if i == 0:
            X_train = X_train_temp
            X_test  = X_test_temp
            y_train = y_train_temp
            y_test  = y_test_temp
        else:
            X_train = np.concatenate([X_train,X_train_temp],axis=0)
            X_test  = np.concatenate([X_test ,X_test_temp],axis=0)
            y_train = np.concatenate([y_train,y_train_temp],axis=0)
            y_test  = np.concatenate([y_test ,y_test_temp],axis=0)
    # print(X_train_temp.shape)
    # print(X_test_temp.shape)
    # print(y_train_temp.shape)
    # print(y_test_temp.shape) 

    return X_train, X_test, y_train, y_test



