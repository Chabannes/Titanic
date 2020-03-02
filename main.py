import pandas as pd
import matplotlib.pyplot as plt

from helper import *
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn import ensemble
from sklearn.neural_network import MLPClassifier
from sklearn import tree



def main():

    data_train = pd.read_csv('train.csv', encoding='utf-8')
    data_test = pd.merge(pd.read_csv('test.csv', encoding='utf-8'),
                         pd.read_csv('gender_submission.csv', encoding='utf-8'))
    Visualize_data = False  # plots
    Create_Features = True
    Evaluate_features = False  # plots
    Drop_Poor_Features = True
    Scale_Data = True
    Choosing_Estimator = True  # plots
    Tuning_Hyperparameters = False
    Evaluate_Performence = False

    list_classifiers = [MLPClassifier(solver='lbfgs', alpha=1e-3),
                        tree.DecisionTreeClassifier(),
                        linear_model.SGDClassifier(max_iter=5, tol=None),
                        Perceptron(max_iter=5),
                        GaussianNB(),
                        KNeighborsClassifier(n_neighbors=3),
                        ensemble.GradientBoostingRegressor(),
                        RandomForestClassifier(n_estimators=500),
                        SVC(gamma=2, C=1),
                        AdaBoostClassifier()]

    data_train, data_test = cleaning_data(data_train), cleaning_data(data_test)
    classifier = RandomForestClassifier(min_weight_fraction_leaf=0, max_features=2,
                                        min_samples_split=2, min_samples_leaf=2, n_estimators=500)

    if Visualize_data:
        visualizing_data(data_train)

    if Create_Features:
        data_train, data_test = creating_new_features(data_train), creating_new_features(data_test)

    if Evaluate_features:
        evaluate_features(data_train)

    if Drop_Poor_Features:
        data_train, data_test = drop_poor_features(data_train, features=['SibSp', 'Parch']), \
                                drop_poor_features(data_test, features=['SibSp', 'Parch'])

    if Scale_Data:
        data_train, data_test = scale_data(data_train), scale_data(data_test)

    if Choosing_Estimator:
        estimator_study(list_classifiers, data_train)

    if Tuning_Hyperparameters:
        grid_search_RandomForest(data_train)

    if Evaluate_Performence:
        plot_ROC_curve(classifier, data_train, data_test)
        LearningCurve(classifier, data_train, data_test)

    plt.show()




if __name__ == '__main__':
    main()
