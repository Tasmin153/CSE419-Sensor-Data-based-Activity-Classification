
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

def train_model(model, X_train, y_train):
	#return model.partial_fit(X_train, y_train)
    return model.fit(X_train, y_train)


def test_model(model,X_test):
	pred = model.predict(X_test)
	return pred
    

# For model evolution
def model_evalution(y_test, y_pred):
    #print("------------------- Model evaluation ----------------\n\n")
    #print("Confusion Matrix : \n",confusion_matrix(y_test, y_pred))
    #print("--------------------------------------------")
    #print("Accuracy Score : ",accuracy_score(y_test,y_pred))
    #print("Classification Report : \n",classification_report(y_test, y_pred))
    #print("--------------------------------------------")
    print("")



def train_fulldataset():
    pass
    #X_train, X_test, y_train, y_test = full_dataset(file_list)

     # print(X_train.shape)
    # print(X_test.shape)
    # print(y_train.shape)
    # print(y_test.shape)

    # file_train_test_wise_info("full_dataset",y_train,y_test)
    # model = train_model(model_arch, X_train, y_train)
    # pred_tree = test_model(model, X_test)
    # model_evalution(y_test,pred_tree)