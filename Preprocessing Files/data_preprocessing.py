
import pandas as pd
import numpy as np

from glob import glob
from utils import read_file, string_to_index, pd_to_np

from sklearn.model_selection import train_test_split



def dataset(file_path, split=0.3):

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

    # y = activityClass
    y = np.asarray(y) 
    y = y.astype('int32')

    #print(y)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = split)

    return X_train, X_test, y_train, y_test
