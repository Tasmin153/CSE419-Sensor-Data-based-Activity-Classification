
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

def train_model(model, X_train, y_train):
	return model.fit(X_train, y_train)


def test_model(model,X_test):
	pred = model.predict(X_test)
	return pred
    

# For model evolution
def model_evalution(y_test, y_pred):
    print("------------------- Model evaluation ----------------\n\n")
    print("Confusion Matrix : \n",confusion_matrix(y_test, y_pred))
    print("\nAccuracy Score : ",accuracy_score(y_test,y_pred),'\n')
    print("Classification Report : \n",classification_report(y_test, y_pred))