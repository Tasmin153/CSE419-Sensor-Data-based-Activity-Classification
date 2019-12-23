
import pandas as pd
import numpy as np
from glob import glob


# find all files path and make a list
def folder_finder(path):

    file_list = []

    for j in glob(path+"*.csv"):
        file_list.append(j)

    return file_list


# read csv file
def read_file(path):

    data = pd.read_csv(path) 
    return data


# pandas to numpy 
def pd_to_np(data):
    
    if type(data) == np.ndarray:
      print('Data is already in numpy format!')
    else:
      data = data.values
      #print('Pandas to Numpy done!')

    return data


# string to id
def string_to_index(activity_label):

    har_class={} 
    activity_class =[]

    activity_label = pd_to_np(activity_label)
    unique, counts = np.unique(activity_label, return_counts=True)

    # string to index dict
    for i in range(len(unique)):
        har_class[unique[i]]=i

    # activity class tranform into indexes
    for i in range(len(activity_label)):
        activity_class.append(har_class[activity_label[i][0]])

    return activity_class


def dataset_info(file_list):

    for file in file_list:

        X_train, X_test, y_train, y_test = dataset(file)
        print("File: ",file)
        print("Size: ",y_train.shape[0])
        print("X_train: ", X_train.shape)
        print("y_train: ", y_train.shape)

        print("X_test: ", X_test.shape)
        print("y_test: ", y_test.shape)


