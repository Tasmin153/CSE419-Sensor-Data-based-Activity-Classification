import pandas as pd
import numpy as np

# read csv file
def read_file(path):
    data = pd.read_csv(path) 
    return data, data.values
    

# Activity classes list
def find_class(data):
    
    if type(data) == np.ndarray:
      print('do nothing')
    else:
      data = data.values

    row = len(data)
    col = len(data[0])
    
    activity_class = []

    for i in range(row):
        activity = data[i][col-1]
        activity_class.append(activity)
    
    return activity_class


# counting the activity classes
def count_class(activity_class):
    unique, counts = np.unique(activity_class, return_counts=True)    
    return unique, counts