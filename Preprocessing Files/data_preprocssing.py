
import pandas as pd
import numpy as np

import sklearn 
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from pandas.plotting import scatter_matrix

from glob import glob
from utils import read_file, find_class , count_class


# ---------------------------------------file finder---------------------------------------------------
folder_list = []
file_list = []

for i in glob('./casas-dataset/*'):
    folder_list.append(i)
    
    for j in glob(i+'\*ann.features.csv'):
        file_list.append(j)
        #print(j)

print('Folder:',len(folder_list))
print('Total files:',len(file_list))
# ---------------------------------------------------------------------------------------------------------



# total_class = read_file(file_list[0])
# unique, counts = count_class(total_class)
# print(unique)
# print(counts)


# ---------------------------------------file finder---------------------------------------------------
pd_data, np_data = read_file(file_list[0])
selectData =pd_data.loc[:, ['lastSensorEventSeconds',
                            'lastSensorDayOfWeek',
                            'lastSensorLocation',
                            'lastMotionLocation',
                            'complexity',
                            'activityChange']]
activityLabel =pd_data.loc[:, ['activity']]


x =  selectData.values
y_old= activityLabel.values

# ---------------------------------------------------------------------------------------------------------

har_class={} 
activityClass =[]

total_class = find_class(np_data)
unique, counts = count_class(total_class)


for i in range(len(unique)):
    har_class[unique[i]]=i

for i in range(len(y_old)):
    activityClass.append(har_class[y_old[i][0]])

 
y = activityClass
y = np.asarray(y) 
y = y.astype('int32')  

X_train, X_test, y_train, y_test =train_test_split(x, y, test_size=0.3)
