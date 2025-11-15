import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
#from sklearn.preprocessing import StandardScaler
#from sklearn.pipeline import make_pipeline
from mlxtend.data import wine_data
from mlxtend.plotting import scatterplotmatrix
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.plotting import plot_decision_regions



wine = load_wine()
x, y = wine.data, wine.target

x_train, x_test , y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42, shuffle= True, stratify=y)

print(f"Number of training points: {len(x_train)}")
print(f"Number of test points {len(x_test)}")

knn = KNeighborsClassifier()

params = {'n_neighbors' : [1, 3, 5, 11], 'p' : [1 ,2]}

grid = GridSearchCV(knn, params, cv = 5)

grid.fit(x_train, y_train)
#print (grid.cv_results_)
print ('Best params:', grid.best_params_)
print ('Best score', grid.best_score_)
print('Best estimator', grid.best_estimator_)

clf = grid.best_estimator_
grid.fit(x_train,y_train)
print('Test accuracy: %.2f%%' % (clf.score(x_test, y_test) * 100))
