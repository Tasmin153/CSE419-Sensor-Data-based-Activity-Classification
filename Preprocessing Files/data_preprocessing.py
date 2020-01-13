import pandas as pd
import numpy as np

from glob import glob
from utils import read_file, string_to_index, pd_to_np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline


def data_loader(file_path, split=0.4):

    h = .02  # step size in the mesh

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

    x = StandardScaler().fit_transform(x)

    #Fitting the PCA algorithm with our Data
    pca = PCA().fit(x)
    #Plotting the Cumulative Summation of the Explained Variance
    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)') #for each component
    plt.title('CASAS dataset Explained Variance')
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = split,random_state=42,stratify=y)

    # print(X_train_temp.shape)

    # Feature Scaling
    # sc = StandardScaler()
    # X_train = sc.fit_transform(X_train)
    # X_test = sc.transform(X_test)

    x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
    y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    n_neighbors = 3


    dim = len(x[0])
    n_classes = len(np.unique(y))

    # pca = PCA(n_components=30)
    # dataset = pca.fit_transform(x)

    # Reduce dimension to 2 with PCA
    pca =PCA(n_components=5)
    pca.fit(x)
    X_pca=pca.transform(x) 
    print ("shape of pca", X_pca.shape)
    ex_variance=np.var(X_pca,axis=0)
    ex_variance_ratio = ex_variance/np.sum(ex_variance)
    print (ex_variance_ratio) 
    print("\nSum of ex_variance_ratio : ", np.sum(ex_variance_ratio)) 

    pca = make_pipeline(StandardScaler(),
                        PCA(n_components=5, random_state=42))

    # Use a nearest neighbor classifier to evaluate the methods
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Make a list of the methods to be compared
    dim_reduction_methods = [('PCA', pca)]

    '''pc_train = pca.fit_transform(X_train)
    pcaDf = pd.DataFrame(data = pc_train, columns = ['pc 1', 'pc 2'])
    pcaDf['Target'] = y_train'''

    plt.figure()
    for i, (name, model) in enumerate(dim_reduction_methods):
        plt.figure()
        plt.subplot(1, 3, i + 1, aspect=1)

        # Fit the method's model
        model.fit(X_train, y_train)

        # Fit a nearest neighbor classifier on the embedded training set
        knn.fit(model.transform(X_train), y_train)

        # Compute the nearest neighbor accuracy on the embedded test set
        acc_knn = knn.score(model.transform(X_test), y_test)

        # Embed the data set in 2 dimensions using the fitted model
        x_embedded = model.transform(x)

        # Plot the projected points and show the evaluation score
        plt.scatter(x_embedded[:, 0], x_embedded[:, 1], c=y, s=30, cmap='Set1')
        plt.title("{}, KNN (k={})\nVariance Range = {:.2f}".format(name,n_neighbors,acc_knn))
        #plt.title("{}, KNN (k={})\nTest accuracy = {:.2f}".format(name,n_neighbors,acc_knn))
    plt.show()
    plt.savefig('dim.png')

    return X_train, X_test, y_train, y_test, xx, yy


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