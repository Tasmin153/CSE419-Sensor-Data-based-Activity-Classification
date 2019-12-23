# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 11:43:21 2019

@author: N M Shihab Islam
"""

import pandas as pd

def data_import(a,b):
    for i in range(a,b): 	# iterate 1 to 5
        print(i)         	# print the iteration
        list_name =[]      	# declare a empty list
        df = pd.read_csv('csh'+str(i)+'/csh'+str(i)+'.ann.features.csv')  # read data
        list_name.append(df)
        return list_name

list_1 = data_import(101,106)
df = pd.concat(list_1,axis = 0)

list_2 = data_import(106,111)
df_1 = pd.concat(list_2,axis = 0)

list_3 = data_import(111,115)
df_2 = pd.concat(list_3,axis = 0)

list_4 = data_import(115,121)
df_3 = pd.concat(list_4,axis = 0)

list_5 = data_import(121,127)
df_4 = pd.concat(list_5,axis = 0)

list_6 = data_import(127,130)
df_5 = pd.concat(list_6,axis = 0)

print(df.activity.value_counts())
print(df_2.activity.value_counts())

def collect_data(df,a,b,c,d,e):
    return df[(df['activity']==a)|(df['activity']==b)|(df['activity']==c)|(df['activity']==d)|(df['activity']==e)]


data_1 = collect_data(df,'Eat','Phone','Cook','Watch_TV','Read')
data_2 = collect_data(df_1,'Eat','Phone','Cook','Watch_TV','Read')
data_3 = collect_data(df_2,'Eat','Phone','Cook','Watch_TV','Read')
data_4 = collect_data(df_3,'Eat','Phone','Cook','Watch_TV','Read')
data_5 = collect_data(df_4,'Eat','Phone','Cook','Watch_TV','Read')
data_6 = collect_data(df_5,'Eat','Phone','Cook','Watch_TV','Read')


data_1.to_csv('dataset_1.csv',index=False)
data_2.to_csv('dataset_2.csv',index=False)
data_3.to_csv('dataset_3.csv',index=False)
data_4.to_csv('dataset_4.csv',index=False)
data_5.to_csv('dataset_5.csv',index=False)
data_6.to_csv('dataset_6.csv',index=False)


train_final_1.to_csv('Mod_2_tran.csv',index=False)
