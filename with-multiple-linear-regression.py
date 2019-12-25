# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 12:18:01 2019

@author: Taoseef
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Datasets\dataset_2.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 46].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
X = sc_X.fit_transform(X)

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#experiment
#np.savetxt('E:\\temporary working files\\Section 5 - Multiple Linear Regression\\vulnability test\\fname.csv', X_train, delimiter=',', fmt='%d')
#X_train.tofile('E:\\temporary working files\\Section 5 - Multiple Linear Regression\\vulnability test\\foo.csv',sep=',',format='%10.5f')
pd.DataFrame(X).to_csv("Datasets\data_2.csv")

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
model = regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

print ("Score:") 
model.score(X_test, y_test)

#Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm 
X = np.append(arr = np.ones((300,1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1,	2,	3,	4,	5,	6,	7,	8,	9,	10,	11,	12,	13,	14,	15,	16,	17,	18,	19,	20,	21,	22,	23,	24,	25,	26,	27,	28,	29,	30,	31,	32,	33,	34,	35,	36,	37,	38,	39,	40,	41,	42,	43,	44,	45]]
X_opt = X[:, [0, 1,	2,	3,	4,	5,	6,	7,	8,	9,	10,	11,	12,	13,	15,	16,	17,	18,	19,	20,	21,	22,	23,	24,	25,	26,	27,	28,	29,	30,	31,	32,	33,	34,	36,	37,	38,	39,	41,	44,	45]]
#X_opt = X[:, [0, 1,	2,	3,	4,	5,	6,	7,	8,	9,	10,	11,	12,	13,	14,	15,	16,	17,	18,	19,	20,	21,	22,	23,	24,	25,	26,	27,	28,	29,	30,	31,	32,	33,	34,	35,	36,	37,	38,	39,	40,	41,	42,	43,	44,	45, 46, 47,	48,	49,	50,	51,	52,	53,	54,	55,	56,	57,	58,	59,	60,	61,	62,	63,	64,	65,	66,	67,	68,	69,	70,	71,	72,	73,	74,	75,	76,	77,	78,	79,	80,	81,	82,	83,	84,	85,	86,	87,	88,	89,	90,	91,	92,	93,	94,	95,	96,	97,	98,	99,	100,	101,	102,	103,	104,	105,	106,	107,	108,	109,	110,	111,	112,	113,	114,	115,	116,	117,	118,	119,	120,	121,	122,	123,	124,	125,	126,	127,	128,	129,	130,	131,	132,	133,	134,	135,	136,	137,	138,	139,	140,	141,	142,	143,	144,	145,	146,	147,	148,	149,	150,	151,	152,	153,	154,	155,	156,	157,	158,	159,	160,	161,	162,	163,	164,	165,	166,	167,	168,	169,	170,	171,	172,	173,	174,	175,	176,	177,	178,	179,	180,	181,	182,	183,	184,	185,	186,	187,	188,	189,	190,	191, 192]]
#X_opt = X[:, [0, 1,	2,	3,	4,	5,	7,	8,	9,	10,	11,	12,	13,	14,	15,	16,	17,	18,	19,	20,	21,	22,	23,	24,	25,	26,	27,	28,	29,	30,	31,	32,	33,	34,	35,	37,	39,	40,	41,	42,	43,	44,	70,	76,	77,	78,	81,	86,	89,	90,	92,	93,	96,	97,	99,	100, 102,	103,	106,	107,	108,	109,	111,	112,	113,	114,	116,	120,	121,	124,	125,	126,	127,	128,	129,	130,	137,	138,	139,	140,	141,	144,	145,	146,	147,	148,	156,	159,	160,	161,	162,	184,	185,	188,	189]]
#X_opt = X[:, [0, 1, 2,	3,	4,	5,	7,	8,	10,	11,	12,	13,	14,	15,	16,	17,	18,	19,	20,	21,	22,	23,	24,	25,	26,	27,	28,	29,	30,	31,	32,	33,	34,	35,	37,	39,	40,	41,	42,	43,	44,	70,	76,	77,	78,	81,	86,	89,	90,	92,	93,	96,	97,	99,	100,  102,	103,	106,	107,	108,	109,	111,	112,	113,	114,	116,	120,	125,	126,	128,	130,	137,	140,	141,	145,	147,	148,	156,	160,	161,	162,	184,	185,	189]]
#X_opt = X[:, [0,  1,   2,	3,	4, 	5,	7,	8,	10,	11,	12,	13,	14,	15,	16,	17,	18,	19,	20,	21,	22,	23,	24,	25,	26,	27,	28,	29,	30,	31,	32,	33,	34,	35,	37,	39,	40,	41,	42,	43,	44,	70,	76,	77,	78,	81,	86,	89,	90,	92,	93,	96,	97,	99,	100,  102,	103,	106,	107,	108,	109,	111,	112,	113,	114,	116,	120,	125,	126,	128,	130,	137,	140,	141,	145,	147,	148,	156,	160,	161,	162,	184,	185,	189]]
#X_opt = X[:, [0,  1,	2,	3,	4, 	5,	7,	8,	10,	11,	12,	13,	14,	15,	16,	17,	18,	19,	21,	22,	24,	25,	26,	27,	28,	29,	30,	31,	32,	33,	34,	35,	37,	39,	40,	41,	43,	44,	70,	76,	77,	78,	81,	86,	89,	90,	92,	93,	96,	97,	99,	100,	102,	103,	106,	107,	108,	109,	111,	112,	113,	114,	116,	120,	125,	126,	128,	130,	140,	147,	148,	160,	161,	162,	184,	189]]

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0,3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()