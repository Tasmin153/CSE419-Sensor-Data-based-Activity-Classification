
import sklearn 
from sklearn.ensemble import RandomForestClassifier


def model_init():

	model = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)

	return model