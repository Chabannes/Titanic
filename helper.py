import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn import ensemble
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as metrics
from sklearn.model_selection import learning_curve




#                                                 CLEANING THE DATA
# ----------------------------------------------------------------------------------------------------------------------


def encode_columns(data):
    # convert male/female and departure harbours into integers
    le = preprocessing.LabelEncoder()
    data['Sex'] = le.fit_transform(data['Sex'].astype(str))  # S = 2 C = 0 Q = 1
    data['Embarked'] = le.fit_transform(data['Embarked'].astype(str))
    return data


def drop_unrelevent_columns(data):
    data.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
    return data


def clean_nan_values(data):
    for i in range(0, len(data)):
        if np.isnan(data['Age'][i]):
            data.at[i, 'Age'] = data['Age'].mean()  # centered distribution
        elif np.isnan(data.at[i, 'Fare']):
            data.at[i, 'Fare'] = data['Fare'].median()  #  baised distribution
    data = data.dropna(subset=['Embarked'])
    return data


def cleaning_data(data):
    data = drop_unrelevent_columns(data)
    data = clean_nan_values(data)
    data = encode_columns(data)

    return data


#                                             VISUALIZING THE DATA
# ----------------------------------------------------------------------------------------------------------------------

def plot_gender(data):
    """

    :param data: dataset, type PandaFrame
    :return: None
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
    women = data[data['Sex'] == 0]
    men = data[data['Sex'] == 1]
    ax = sns.distplot(women[women['Survived'] == 1].Age.dropna(), bins=18, label='survived', ax=axes[0], kde=False)
    ax = sns.distplot(women[women['Survived'] == 0].Age.dropna(), bins=18, label='not survived', ax=axes[0], kde=False)
    ax.legend()
    ax.set_title('Female')
    ax = sns.distplot(men[men['Survived'] == 1].Age.dropna(), bins=18, label='survived', ax=axes[1], kde=False)
    ax = sns.distplot(men[men['Survived'] == 0].Age.dropna(), bins=18, label='not survived', ax=axes[1], kde=False)
    ax.legend()
    ax.set_title('Male')


def plot_factors(data):
    """

    :param data: dataset, type PandaFrame
    :return: None
    """
    row = 0
    factors = ['Pclass', 'Sex', 'Embarked']
    prediction = data['Survived']
    sns.set(style="whitegrid")
    # values for the factors.
    plots = plt.subplots(len(factors), 1, figsize=(5, 10), squeeze=False)
    # It should
    for factor in factors:
        # Then print the total for each prediction
        sns.barplot(x=factor, y=prediction, data=data, ax=plots[1][row][0])
        row = row + 1
    plt.tight_layout()


def plot_age_distribution(data):
    """

    :param data: dataset, type PandaFrame
    :return: None
    """
    plt.figure()
    # plots a kernel density estimate of the subset of the 1st class passangers's age
    data.Age[data.Pclass == 1].plot.kde()
    data.Age[data.Pclass == 2].plot.kde()
    data.Age[data.Pclass == 3].plot.kde()
    plt.legend(('First Class', 'Second Class', 'Third Class'), loc='best')
    plt.xlabel("Age")
    plt.title("Age Distribution in the different classes")


def plot_age_groups(data):
    """

    :param data: dataset, type PandaFrame
    :return: None
    """
    plt.figure()
    df = data.copy()
    df['Age'] = df['Age'].astype(int)
    df.loc[df['Age'] <= 10, 'Age'] = 0
    df.loc[(df['Age'] > 10) & (df['Age'] <= 20), 'Age'] = 1
    df.loc[(df['Age'] > 20) & (df['Age'] <= 25), 'Age'] = 2
    df.loc[(df['Age'] > 25) & (df['Age'] <= 29), 'Age'] = 3
    df.loc[(df['Age'] > 29) & (df['Age'] <= 35), 'Age'] = 4
    df.loc[(df['Age'] > 35) & (df['Age'] <= 44), 'Age'] = 5
    df.loc[(df['Age'] > 44) & (df['Age'] <= 69), 'Age'] = 6
    df.loc[df['Age'] > 69, 'Age'] = 6
    plt.title('Survivals within the different age groups')
    sns.barplot(x=df['Age'], y=df['Survived'])


def plot_fare(data):
    """

    :param data: dataset, type PandaFrame
    :return: None
    """
    plt.scatter(data.Survived, data.Fare, alpha=0.1)
    plt.ylabel("Fare")
    plt.grid(axis='y')
    plt.title("Survival by fare")
    plt.xlim(-1, 2)


def plot_relatives(data):
    """

    :param data: dataset, type PandaFrame
    :return: None
    """
    fig = plt.figure()
    data['Relatives'] = data['SibSp'] + data['Parch']
    df_alone = data.Survived[data.Relatives == 0].value_counts().sort_index()
    df_not_alone = data.Survived[data.Relatives > 0].value_counts().sort_index()
    ax1 = fig.add_subplot(121)
    (df_alone / float(df_alone.sum())).plot(kind='barh', color='blue', label='Alone', alpha=0.5)
    (df_not_alone / float(df_not_alone.sum())).plot(kind='barh', color='red', label='With relatives', alpha=0.5)
    plt.title("Impact of the number of relatives on chances of survival ")
    plt.legend(loc='best')
    ax1.set_ylim(-1, 2)


def visualizing_data(data):
    """

    :param data: dataset, type PandaFrame
    :return: None
    """
    plot_factors(data)
    plot_gender(data)
    plot_age_distribution(data)
    plot_fare(data)
    plot_relatives(data)
    plot_age_groups(data)


#                                             CREATING NEW FEATURES
# ----------------------------------------------------------------------------------------------------------------------


def creating_new_features(data):
    """

    :param data: dataset, type PandaFrame
    :return: None
    """
    data['A*C'] = data['Pclass'] ** 3 * data['Age']
    data.loc[data['SibSp'] + data['Parch'] > 0, 'Is_Alone'] = 0
    data.loc[data['SibSp'] + data['Parch'] == 0, 'Is_Alone'] = 1

    deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
    data['Cabin'] = data['Cabin'].fillna("U0")
    data['Deck'] = data['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    data['Deck'] = data['Deck'].map(deck)
    data['Deck'] = data['Deck'].fillna(0)
    data['Deck'] = data['Deck'].astype(int)  # we can now drop the cabin feature
    data = data.drop(['Cabin'], axis=1)
    return data


def cross_plot(data):
    """

    :param data: dataset, type PandaFrame
    :return: None
    """
    corrmat = data.corr()
    k = len(data.columns)
    plt.figure(figsize=(15, 15))
    cols = corrmat.nlargest(k, 'Survived')['Survived'].index
    cm = np.corrcoef(data[cols].values.T)
    sns.set(font_scale=0.5)
    sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
                annot_kws={'size': 10}, cmap=plt.cm.viridis, yticklabels=cols.values,
                linecolor='white', xticklabels=cols.values)
    plt.title('Pearson correlation matrix using HeatMap')


def plot_best_features(data):
    """

    :param data: dataset, type PandaFrame
    :return: None
    """
    plt.figure()
    y = data['Survived']
    x = data.drop('Survived', 1)
    n_feature = len(x.columns)
    selector = SelectKBest(f_classif, k=5)
    selector.fit(x, y)
    # get the raw p-values for each feature, transform from p-values into scores
    scores = -np.log10(selector.pvalues_)
    s = 'Score : \n'
    i = 0
    for col in x:
        s = s + str(col) + ' : %f \n' % scores[i]
        i = i + 1
    plt.bar(range(n_feature), scores)
    plt.xticks(range(n_feature), x, rotation='vertical')
    plt.title('Features Scores')
    plt.text(4, 15, s, fontsize=10)


def RF_feature_importance(data):
    """

    :param data: dataset, type PandaFrame
    :return: None
    """
    X = data.drop('Survived', 1)
    y = data['Survived']
    forest = RandomForestClassifier(n_estimators=500)
    forest.fit(X, y)
    importances = pd.DataFrame(
        {'feature': X.columns, 'importance': np.round(forest.feature_importances_, 3)})

    importances = importances.sort_values('importance', ascending=False).set_index('feature')
    importances.head(15)
    importances.plot.bar(title='Random Forest Features Importance')


def evaluate_features(data):
    """

    :param data: dataset, type PandaFrame
    :return: None
    """
    cross_plot(data)
    plot_best_features(data)
    RF_feature_importance(data)


def drop_poor_features(data, features):
    """

    :param data: dataset, type PandaFrame
    :param features: list of features to evaluate, type list
    :return: dataset without irrelevant features, type PandaFrame
    """
    for feature in features:
        data.drop(feature, 1)
    return data


#                                             SCALING DATA
# ----------------------------------------------------------------------------------------------------------------------
def scale_data(data):
    """

    :param data: dataset, type PandaFrame
    :return: scaled data, type PandaFrame
    """
    scaled_features = data.copy()
    try:
        col_names = ['Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'A*C']
        features = scaled_features[col_names]
        scaler = StandardScaler().fit(features.values)
        scaled_features[col_names] = features
        features = scaler.transform(features.values)
        scaled_features[col_names] = features
        return scaled_features
    except:
        print("Error in scale_data. \nYou must apply scale_data() after dropping \
        the unrelevant features and adding the customized ones")


#                                        CHOOSING AN ESTIMATOR
# ----------------------------------------------------------------------------------------------------------------------


def estimator_study(classifiers, data):
    '''
    :param classifiers: list of classifiers, type list
    :return: none (plot results
    sqrt(len(classifier)) must be round
    '''
    y = data['Survived']
    x = data.drop('Survived', 1)
    fig, axs = plt.subplots(int(len(classifiers) ** 0.5), int(len(classifiers) ** 0.5),
                            figsize=(20, 20), facecolor='w', edgecolor='k')
    fig.suptitle("Cross Validation Scores", fontsize=11)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    axs = axs.ravel()
    for k in range(len(classifiers) - 1):
        scores = cross_val_score(classifiers[k], x, y, cv=20)
        clf_str = str(classifiers[k]).partition("(")[0]
        title = "Cross Validation of classifier " + clf_str + " : " + str(round(np.mean(scores), 3))
        axs[k].set_title(title, fontsize=8)
        sns.distplot(pd.Series(scores), color='r', bins=5, ax=axs[k])


#                                       TUNING HYPERPARAMETERS
# ----------------------------------------------------------------------------------------------------------------------


def grid_search_RandomForest(data):
    """

    :param data: dataset, type PandaFrame
    :return: None
    """
    y = data['Survived']
    X = data.drop('Survived', 1)
    n_estimators = [100, 200, 500, 800, 1000]
    min_samples_split = [2, 4]
    min_samples_leaf = [1, 2, 4]
    min_weight_fraction_leaf = [0, 0.2, 0.5]
    max_features = [2, 4, 6, len(X.columns)]
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
    grid_result = grid.fit(X, y)
    print('Best Score: ', grid_result.best_score_)
    print('Best Params: ', grid_result.best_params_)


#                                          EVALUATING PERFORMENCES
# ----------------------------------------------------------------------------------------------------------------------


def plot_ROC_curve(classifier, data_train, data_test):
    '''
    :param classifier: chosen classifier from sklearn
    :return: none (plot ROC curve)
    '''
    y_train = data_train['Survived']
    X_train = data_train.drop('Survived', 1)
    y_test = data_test['Survived']
    X_test = data_test.drop('Survived', 1)
    name_classifier = str(classifier).partition("(")[0]
    clf = classifier
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)
    preds = probs[:, 1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)

    plt.figure()
    title = 'Receiver Operating Characteristic for classifier ' + name_classifier
    plt.title(title)
    plt.plot(fpr, tpr, 'b', label='ROC_AUC SCORE = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')


def LearningCurve(classifier, data_train, data_test):
    """

    :param classifier: Object classifier
    :param data_train: train data
    :param data_test: test data
    :return: none
    """
    plt.figure()
    classifier_name = str(classifier).partition("(")[0]
    data_set = pd.concat([data_train, data_test], axis=0)
    y = data_set['Survived']
    X = data_set.drop('Survived', 1)
    train_sizes, train_scores, test_scores = learning_curve(classifier,
                                                            X,
                                                            y,
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
    plt.plot(train_sizes, train_mean, color='r', label="Training score")
    plt.plot(train_sizes, test_mean, color='b', label="Cross-validation score")

    # Draw bands
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='r', alpha=0.6)
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color='b', alpha=0.6)

    # Create plot
    plt.title("Learning Curve " + classifier_name)
    plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
    plt.tight_layout()
