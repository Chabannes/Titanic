import pandas as pd
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as metrics
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import linear_model
from sklearn import ensemble
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import re
import seaborn as sns


tunned_estimator = RandomForestClassifier(min_weight_fraction_leaf=0, max_features=2,
                                          min_samples_split=15, min_samples_leaf=2, n_estimators=100)

list_classifiers = [MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2)),
                    tree.DecisionTreeClassifier(),
                    linear_model.SGDClassifier(max_iter=5, tol=None),
                    Perceptron(max_iter=5),
                    GaussianNB(),
                    KNeighborsClassifier(n_neighbors=3),
                    ensemble.GradientBoostingRegressor(),
                    RandomForestClassifier(n_estimators=100),
                    tunned_estimator]


def remove_letter(chaine):
    try:
        chaine = chaine.replace(' ', '')
        chaine = chaine.replace('/', '')
        chaine = chaine.replace('.', '')
        chaine = re.sub(r'[A-Z]+', '', chaine, re.I)
        chaine = re.sub(r'[a-z]+', '', chaine, re.I)
        number = int(chaine)
        return number
    except:
        return np.nan


# ----------------------------------------------------------------------------------------------------------------------
#                                             TRAINING DATASET
# ----------------------------------------------------------------------------------------------------------------------

data_train_pd = pd.read_csv('train.csv', encoding='utf-8')


# DROP IRRELEVANT TABLES
data_train_pd = data_train_pd.drop('Name', 1)
data_train_pd = data_train_pd.drop('Cabin', 1)

# CONVERT STR INTO BOOLEAN
data_train_pd.loc[data_train_pd['Sex'] == 'female', 'Sex'] = 0
data_train_pd.loc[data_train_pd['Sex'] == 'male', 'Sex'] = 1

data_train_pd.loc[data_train_pd['Embarked'] == 'Q', 'Embarked'] = 0
data_train_pd.loc[data_train_pd['Embarked'] == 'S', 'Embarked'] = 1
data_train_pd.loc[data_train_pd['Embarked'] == 'C', 'Embarked'] = 2


# CONVERT STRING TO FLOAT
for i in range(np.shape(data_train_pd)[0]):
    s = data_train_pd['Ticket'].iloc[i]
    data_train_pd['Ticket'].iloc[i] = remove_letter(s)


# DROP NAN COLUMNS

# data_train_pd.columns = ['PassengerId','Pclass','Sex','Age','SibSp','Parch','Ticket','Fare','Embarked','Survived']


data_train_pd = data_train_pd.dropna(subset = ['PassengerId','Pclass','Sex','Age','SibSp','Parch','Ticket','Fare','Embarked','Survived'])
y_train_pd = data_train_pd['Survived']
X_train_pd = data_train_pd.drop('Survived', 1)

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
#                                             TESTING DATASET
# ----------------------------------------------------------------------------------------------------------------------

test_pd = pd.read_csv('test.csv', encoding='utf-8')
id_pd = pd.read_csv('gender_submission.csv', encoding='utf-8')
data_test_pd = pd.merge(test_pd, id_pd, on='PassengerId')


# DROP IRRELEVANT TABLES
data_test_pd = data_test_pd.drop('Name', 1)
data_test_pd = data_test_pd.drop('Cabin', 1)

# CONVERT STR INTO BOOLEAN
data_test_pd.loc[data_test_pd['Sex'] == 'female', 'Sex'] = 0
data_test_pd.loc[data_test_pd['Sex'] == 'male', 'Sex'] = 1

data_test_pd.loc[data_test_pd['Embarked'] == 'Q', 'Embarked'] = 0
data_test_pd.loc[data_test_pd['Embarked'] == 'S', 'Embarked'] = 1
data_test_pd.loc[data_test_pd['Embarked'] == 'C', 'Embarked'] = 2


# CONVERT STRING TO FLOAT
for i in range (np.shape(data_test_pd)[0]):
    s = data_test_pd['Ticket'].iloc[i]
    data_test_pd['Ticket'].iloc[i] = remove_letter(s)

# DROP NAN COLUMNS

data_test_pd = data_test_pd.dropna(subset = ['PassengerId','Pclass','Sex','Age','SibSp','Parch','Ticket','Fare','Embarked','Survived'])


def split_data(data_train, data_test):
    ytrain = data_train['Survived']
    Xtrain = data_train.drop('Survived', 1)
    ytest = data_test['Survived']
    Xtest = data_test.drop('Survived', 1)
    return(Xtrain, ytrain, Xtest, ytest)

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
#                                                   ANALYSING DATA
# ----------------------------------------------------------------------------------------------------------------------

# class stat
stat_class = data_train_pd[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# sex stat
sex_stat = data_train_pd[["Sex", "Survived"]].groupby(['Sex']
                                                     , as_index=False).mean().sort_values(by='Survived', ascending=False)
# SIBling or spouses
sib_stat = data_train_pd[["SibSp", "Survived"]].groupby(['SibSp']
                                                       , as_index=False).mean().sort_values(by='Survived', ascending=False)

# PARent or CHild
parch_stat = data_train_pd[["Parch", "Survived"]].groupby(['Parch'],
                                                   as_index=False).mean().sort_values(by='Survived', ascending=False)


def plot_gender():
    data = data_train_pd
    survived = 'survived'
    not_survived = 'not survived'
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
    women = data[data['Sex'] == 0]
    men = data[data['Sex'] == 1]
    ax = sns.distplot(women[women['Survived'] == 1].Age.dropna(), bins=18, label=survived, ax=axes[0], kde=False)
    ax = sns.distplot(women[women['Survived'] == 0].Age.dropna(), bins=40, label=not_survived, ax=axes[0], kde=False)
    ax.legend()
    ax.set_title('Female')
    ax = sns.distplot(men[men['Survived'] == 1].Age.dropna(), bins=18, label=survived, ax=axes[1], kde=False)
    ax = sns.distplot(men[men['Survived'] == 0].Age.dropna(), bins=40, label=not_survived, ax=axes[1], kde=False)
    ax.legend()
    ax.set_title('Male')


def plot_embarked():
    data = data_train_pd.copy() #copy of pandas dataframe
    data.loc[data['Sex'] == 0, 'Sex'] = 'Female'
    data.loc[data['Sex'] == 1, 'Sex'] = 'Male'
    data.loc[data['Embarked'] == 0, 'Embarked'] = 'Queenstown'
    data.loc[data['Embarked'] == 1, 'Embarked'] = 'Southampton'
    data.loc[data['Embarked'] == 2, 'Embarked'] = 'Cherbourg'
    FacetGrid = sns.FacetGrid(data, row='Embarked', size=2, aspect=1.6)
    FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=None, order=None, hue_order=None)
    FacetGrid.add_legend()


def plot_class():
    plt.figure()
    ax = sns.barplot(x='Pclass', y='Survived', data=data_train_pd)
    ax.set_title('Survival probability and classes')


# -----------------------------------------------------------------------------------------------------------------------
#                                             ENGINEERING NEW FEATURES
# -----------------------------------------------------------------------------------------------------------------------

# -----------Is under 10-----------#

# update testing dataset
data_test_pd['Under 10'] = data_test_pd['Survived']          #creating new column
data_test_pd.loc[data_test_pd['Age'] <= 10, 'Under 10'] = 1
data_test_pd.loc[data_test_pd['Age'] > 10, 'Under 10'] = 0

# update training dataset
data_train_pd['Under 10'] = data_train_pd['Survived']          #creating new column
data_train_pd.loc[data_train_pd['Age'] <= 10, 'Under 10'] = 1
data_train_pd.loc[data_train_pd['Age'] > 10, 'Under 10'] = 0

under10_stat = data_train_pd[["Under 10", "Survived"]].groupby(['Under 10'],
                                                   as_index=False).mean().sort_values(by='Survived', ascending=False)

# ----------Family Size---------#

# update testing dataset
data_test_pd['Family Size'] = data_test_pd['Survived']          # creating new column
data_test_pd['Family Size'] = data_test_pd['SibSp'] + data_test_pd['Parch']

# update testing dataset
data_train_pd['Family Size'] = data_train_pd['Survived']          # creating new column
data_train_pd['Family Size'] = data_train_pd['SibSp'] + data_train_pd['Parch']

family_stat = data_train_pd[["Family Size", "Survived"]].groupby(['Family Size'],
                                                   as_index=False).mean().sort_values(by='Survived', ascending=False)

# ---------IS ALONE--------#

# update testing dataset
data_test_pd['Is Alone'] = data_test_pd['Survived']          # creating new column
data_test_pd.loc[data_test_pd['Family Size'] == 0, 'Is Alone'] = 1
data_test_pd.loc[data_test_pd['Family Size'] != 0, 'Is Alone'] = 0

# update training dataset
data_train_pd['Is Alone'] = data_train_pd['Survived']          # creating new column
data_train_pd.loc[data_train_pd['Family Size'] == 0, 'Is Alone'] = 1
data_train_pd.loc[data_train_pd['Family Size'] != 0, 'Is Alone'] = 0

alone_stat = data_train_pd[["Is Alone", "Survived"]].groupby(['Is Alone'],
                                                   as_index=False).mean().sort_values(by='Survived', ascending=False)


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------



# ----------------------------------------------------------------------------------------------------------------------
#                                                 FEATURE SELECTION
# ----------------------------------------------------------------------------------------------------------------------

X_train_pd, y_train_pd, X_test_pd, y_test_pd = split_data(data_train_pd,data_test_pd)


def cross_plot():
    corrmat = data_train_pd.corr()
    k = len(data_train_pd.columns)
    plt.figure(figsize=(15, 15))
    cols = corrmat.nlargest(k, 'Survived')['Survived'].index
    cm = np.corrcoef(data_train_pd[cols].values.T)
    sns.set(font_scale=1.25)
    sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
                     annot_kws={'size': 10},
                     cmap=plt.cm.viridis,
                     yticklabels=cols.values,
                     linecolor='white',
                     xticklabels=cols.values)
    plt.title('Pearson correlation matrix using HeatMap')



def plot_best_features():
    plt.figure()
    y = data_train_pd['Survived']
    X = data_train_pd.drop('Survived', 1)
    n_feature = len(X.columns)
    selector = SelectKBest(f_classif, k=5)
    selector.fit(X, y)
    # Get the raw p-values for each feature, and transform from p-values into scores
    scores = -np.log10(selector.pvalues_)
    s = ' Score : \n'
    i = 0
    for col in X:
        s = s + str(col)  +' : %f \n' %  scores[i]
        i = i + 1
    # Plot the scores.  See how "Pclass", "Sex", "Title", and "Fare" are the best?
    plt.bar(range(n_feature), scores)
    plt.xticks(range(n_feature), X_train_pd, rotation='vertical')
    plt.title('Features Scores')
    plt.text(10,18,s,fontsize = 9)


# -------------------- drop unrelevent feature and update datasets ---------------------- #

data_train_pd = data_train_pd.drop('PassengerId', 1)
data_test_pd = data_test_pd.drop('PassengerId', 1)

data_train_pd = data_train_pd.drop('Ticket', 1)
data_test_pd = data_test_pd.drop('Ticket', 1)
#
# data_train_pd = data_train_pd.drop('SibSp', 1)
# data_test_pd = data_test_pd.drop('SibSp', 1)
#
# data_train_pd = data_train_pd.drop('Family Size', 1)
# data_test_pd = data_test_pd.drop('Family Size', 1)

# data_train_pd = data_train_pd.drop('Sex', 1)
# data_test_pd = data_test_pd.drop('Sex', 1)

X_train_pd, y_train_pd, X_test_pd, y_test_pd = split_data(data_train_pd,data_test_pd)
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------



# ----------------------------------------------------------------------------------------------------------------------
#                                                 VALIDATION
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------- CROSS PLOT -----------------------------------------------------------
def plot_cross_validation(classifiers):
    '''
    :param classifiers: list
    :return: none (plot results
    sqrt(len(classifier)) must be round
    '''
    fig, axs = plt.subplots(int(len(classifiers)**0.5), int(len(classifiers)**0.5),
                            figsize=(20, 20), facecolor='w', edgecolor='k')
    fig.suptitle("Cross Validation Scores", fontsize=11)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    axs = axs.ravel()
    for k in range(len(classifiers)):
        scores = cross_val_score(classifiers[k], X_train_pd, y_train_pd, cv=10)
        clf_str = str(classifiers[k]).partition("(")[0]
        title = "Cross Validation of classifier " + clf_str + " : " + str(round(np.mean(scores),3))
        axs[k].set_title(title, fontsize = 8)
        sns.distplot(pd.Series(scores), color='r', ax=axs[k])


# -------------------------------- CONFUSION MATRIX (CLASSIFICATION REPORT) --------------------------------------------

def classification(classifier):
    '''
    :param classifier: chosen classifier from sklearn
    :return: classification report, type array
    '''
    clf = classifier
    clf.fit(X_train_pd,y_train_pd)
    y_pred = clf.predict(X_test_pd)
    return(classification_report(y_test_pd, y_pred))


def plot_ROC_curve(classifier, name_classifier):
    '''
    :param classifier: chosen classifier from sklearn
    :return: none (plot ROC curve)
    '''
    clf = classifier
    clf.fit(X_train_pd,y_train_pd)
    probs = clf.predict_proba(X_test_pd)
    preds = probs[:, 1]
    fpr, tpr, threshold = metrics.roc_curve(y_test_pd, preds)
    roc_auc = metrics.auc(fpr, tpr)

    plt.figure()
    title = 'Receiver Operating Characteristic for classifier ' + str(classifier).partition("(")[0] + " "
    plt.title(title + name_classifier)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')


# ----------------------------------------------------------------------------------------------------------------------
#                                             TUNING HYPERPARAMETERS
# ----------------------------------------------------------------------------------------------------------------------


def grid_search_RandomForest():
    n_estimators  = [5, 10, 50, 100, 500, 1000]
    min_samples_split = [2, 4, 15]
    min_samples_leaf = [2, 4, 15]
    min_weight_fraction_leaf = [0, 0.2, 0.5]
    max_features = [2, 4, 6, len(X_train_pd.columns)]
    param_grid = dict(n_estimators=n_estimators,
                      min_samples_split=min_samples_split,
                      min_samples_leaf=min_samples_leaf,
                      min_weight_fraction_leaf=min_weight_fraction_leaf,
                      max_features=max_features)
    grid = GridSearchCV(estimator=RandomForestClassifier(),
                        param_grid=param_grid,
                        scoring='roc_auc',
                        verbose=1,
                        n_jobs=-1)
    grid_result = grid.fit(X_train_pd, y_train_pd)
    print('Best Score: ', grid_result.best_score_)
    print('Best Params: ', grid_result.best_params_)

# grid_search_RandomForest()
# result: Best Score:  0.8606034343356287
# Best Params:  {'min_weight_fraction_leaf': 0, 'max_features': 2, 'min_samples_split': 15, 'min_samples_leaf': 2, 'n_estimators': 100}



# ----- Test performence upgraded estimator -------#
tunned_estimator = RandomForestClassifier(min_weight_fraction_leaf = 0, max_features = 2,
                                          min_samples_split = 15, min_samples_leaf = 2, n_estimators = 100)


# ----------------------------------------------------------------------------------------------------------------------
#                                            LEARNING CURVE
# ----------------------------------------------------------------------------------------------------------------------


def LearningCurve(classifier, classifier_name):
    plt.figure()
    data_set = pd.concat([data_train_pd, data_test_pd], axis=0)
    Y = data_set['Survived']
    X = data_set.drop('Survived', 1)
    train_sizes, train_scores, test_scores = learning_curve(classifier,
                                                            X,
                                                            Y,
                                                            # Number of folds in cross-validation
                                                            cv=10,
                                                            # Evaluation metric
                                                            scoring='accuracy',
                                                            # Use all computer cores
                                                            n_jobs=-1,
                                                            # 50 different sizes of the training set
                                                            train_sizes=np.linspace(0.01, 1.0, 50))

    # Create means and standard deviations of training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Create means and standard deviations of test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Draw lines
    plt.plot(train_sizes, train_mean, color='r',  label="Training score")
    plt.plot(train_sizes, test_mean, color='b', label="Cross-validation score")

    # Draw bands
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='r', alpha=0.6)
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color='b', alpha=0.6)

    # Create plot
    plt.title("Learning Curve " + classifier_name)
    plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
    plt.tight_layout()

# ----------------------------------------------------------------------------------------------------------------------
#                                                  RESULTS
# ----------------------------------------------------------------------------------------------------------------------

plot_gender()
plot_embarked()
plot_class()

print(cross_plot())
print(plot_best_features())

print(plot_cross_validation(list_classifiers))

# print(" Before Tunning",classification(RandomForestClassifier(n_estimators=100)))
# print(" After Tunning",classification(tunned_estimator))
# 
# print(plot_ROC_curve(RandomForestClassifier(n_estimators=100), 'Before Tunning'))
# print(plot_ROC_curve(tunned_estimator, 'After Tunning'))
#
# print(LearningCurve(RandomForestClassifier(n_estimators=100), 'Before Tunning'))
# print(LearningCurve(tunned_estimator, 'After Tunning'))

plt.show()
