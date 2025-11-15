"""
🍷 Wine Quality Prediction - Decision Tree Model

Autor: Dorota Nalewajek
Cel: Klasyfikacja win na podstawie cech chemicznych.
"""


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import pandas as pd
from sklearn.cluster import KMeans


white_wine = pd.read_csv("/PUT YOUR FILE PATH/winequality-white.csv")
red_wine= pd.read_csv("/PUT YOUR FILE PATH/winequality-red.csv")

# Wczytaj dane z pliku CSV (dostosuj separator, jeśli potrzeba)
df_white = pd.read_csv('/PUT YOUR FILE PATH/winequality-white.csv', delimiter=';')
df_red = pd.read_csv('/PUT YOUR FILE PATH/winequality-red.csv', delimiter=';')


df_white['wine_type'] = 'white'
df_red['wine_type'] = 'red'

df_combined = pd.concat([df_white, df_red], ignore_index=True)

y = df_combined['quality'].values.reshape(-1, 1)

# Klasteryzacja na 3 grupy
kmeans = KMeans(n_clusters=3, random_state=42)
df_combined['wine_class'] = kmeans.fit_predict(y)

print("Podział na klasy przez KMeans:")
print(df_combined['wine_class'].value_counts())

#Podgląd wynikowego zbioru danych
# print(df_combined.head())
# print(f'Łączna liczba próbek: {df_combined.shape[0]}')

# Konwersja DataFrame na tablicę numpy
wine_numpy = df_combined.to_numpy()
wine_numpy = wine_numpy[:,:-1]

# Wyświetlenie pierwszych 5 wierszy tablicy
# print("Tablica danych (pierwsze 5 wierszy):")
#print(wine_numpy[:5])

# Rozmiar danych (wiersze, kolumny)
# print("\nKształt tablicy:", wine_numpy.shape)

# Sprawdzenie typów danych
# print("\nTypy danych w tablicy:", wine_numpy.dtype

x= df_combined.drop(columns=['wine_type'])  # cechy (wszystkie kolumny oprócz etykiety)
y = df_combined['wine_class']

wine_numpy = df_combined.to_numpy()

np.set_printoptions(suppress=True)

print( 'wine features', np.unique(x))
print ('class labels' , np.unique (y))

pipe = make_pipeline(StandardScaler(), DecisionTreeClassifier())

x_train , x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle= True ,stratify=y)

print(f"Liczba cech w danych: {x_train.shape[1]}")
print(f"Liczba nazw cech: {len(x)}")
print(f"Number of training points: {len(x_train)}")
print(f"Number of test points {len(x_test)}")

# print ('Labels counts in y:' , np.bincount(y))
# print('Label counts in y_train:', np.bincount(y_train))
# print('Label counts in y_test:', np.bincount(y_test))

dtc = DecisionTreeClassifier(random_state=42)

params = {'max_depth': range(1, 21)}  

grid = GridSearchCV(dtc, param_grid=params ,cv = 5)
grid.fit(x_train,y_train)
#print (grid.cv_results_)
print ('Best params:', grid.best_params_)
print('Best score: {:.2f}%'.format(grid.best_score_ * 100))
print('Best estimator', grid.best_estimator_)

clf = DecisionTreeClassifier(criterion='entropy', max_depth= 2, random_state=42)
clf.fit(x_train,y_train)
print('Test accuracy: %.2f%%' % (clf.score(x_test, y_test) * 100))


plt.figure(figsize=(15,10))
plot_tree(clf, 
          filled=True, 
          rounded=True, 
          class_names=['Class 0', 'Class 1', 'Class 2'], 
          feature_names = wine_numpy )
plt.title('Wine quality - Decision Tree / classification')
plt.show()



