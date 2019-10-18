
import pandas as pd
import numpy as np

from glob import glob
from utils import read_file, string_to_index, pd_to_np

from sklearn.model_selection import train_test_split



def dataset(file_path, split=0.3):

    pd_data = read_file(file_path)
  
    selectData = pd_data.loc[:, ['lastSensorEventSeconds',
                                'lastSensorDayOfWeek',
                                'lastSensorLocation',
                                'lastMotionLocation',
                                'complexity',
                                'activityChange']]
    activityLabel = pd_data.loc[:, ['activity']]


    x = pd_to_np(selectData)
    y = string_to_index(activityLabel)

    # y = activityClass
    # y = np.asarray(y) 
    # y = y.astype('int32')

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = split)

    return X_train, X_test, y_train, y_test
