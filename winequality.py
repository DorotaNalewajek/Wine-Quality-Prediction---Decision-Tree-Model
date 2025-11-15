from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


wine= load_wine()
wine_data , wine_target = wine.data, wine.target
print (wine)

#print(wine.DESCR)
# print("Nazwy cech:", wine.feature_names)
# print("Klasa wina:", wine.target_names)

def DecisionTreeClassifier ( )


def split_train_test(wine_data , wine_target, test_size = 0.2, random_state =42):
    wine_data_train , wine_data_test , wine_target_train, wine_target_test = train_test_split(wine_data, wine_target, test_size = test_size, random_state= random_state)
    print (f'Traning set : {wine_data_train.shape} \n'f'Test set : {wine_data_test.shape}')
    return wine_data_train, wine_data_test, wine_target_train, wine_target_test

#def decision_tree (wine_data_train, wine_data_test, wine_target_train, wine_target_test):

def best_depth(wine_data_train, wine_target_train):
    wine_data , wine_target = wine.data, wine.target
    dtree = DecisionTreeClassifier(random_state=42)
    param_grid = {'Max depth' : range (1, 11)}
    grid_search = GridSearchCV(dtree,param_grid, cv = 5, scoring= 'accurracy')
    grid_search.fit(wine_data_train, wine_target_train)

    best_depth = grid_search.best_params_['Max depth']
    best_score = grid_search.best_score_
    print (f'Best depth is {best_depth} \n'f'fBest score is {best_score:4f}')
 



if __name__ == '__main__':
    split_train_test(wine_data , wine_target )
    best_depth()
