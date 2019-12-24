
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

    activity_class =[]
    har_class = {
                    'Cook':0,
                    'Eat':1,
                    'Phone':2,
                    'Read':3,
                    'Watch_TV':4
                }

    activity_label = pd_to_np(activity_label)

    # activity class tranform into indexes
    for label in activity_label:
        activity_class.append(har_class[label[0]])

    return activity_class


def dataset_info(file_list):

    count = 0

    for file in file_list:
        temp = read_file(file)
        print("File: ",file)
        print("Size: ",temp.shape[0])
        count+=temp.shape[0]

    print("Total datapoints: ",count)

def file_train_test_wise_info(file,y_train,y_test):
    
    har_k2i = ['Cook','Eat','Phone','Read','Watch_TV']

    print("--------------------------------------------")
    print("Filename: ",file[11:-4])
    print("--------------------------------------------")
    unique, count1 = np.unique(y_train, return_counts=True)
    #print("Train class",unique,count1)
    unique, count2 = np.unique(y_test, return_counts=True)
    #print("Test class",unique,count2)

    class_count = np.sum([count1,count2],axis=0)

    print('{:10s} {:10s} {:10s} {:10s}'.format("Class","Train","Test","Total"))
    if len(class_count) == len(har_k2i):
        for i in range(len(class_count)):
            print('{:10s} {:10s} {:10s} {:10s}'.format(har_k2i[i],str(count1[i]),str(count2[i]),str(class_count[i])))
            # print(har_k2i[i],count1[i],count2[i],class_count[i])

    print("--------------------------------------------")



