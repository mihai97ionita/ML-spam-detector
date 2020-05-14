import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

from sklearn.metrics import confusion_matrix
import math
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
import pandas as pd
import numpy as np
import pickle

models = {
    'RandomForestClassifier': RandomForestClassifier(random_state=0),
    # ,'K-Nearest-Neighbors Classifier': KNeighborsClassifier(),
    # 'LogisticRegression': LogisticRegression(solver='liblinear'),
    # 'Bernoulli Naive Bayes':BernoulliNB(),
    # 'Multinomial Naive Bayes':MultinomialNB(),
    # 'Support vector machines with linear kernel':LinearSVC(),
    # 'Support vector machines with polynomial kernel':SVC(kernel='poly'),
    'Decision trees Classifier': DecisionTreeClassifier(random_state=0),
    'Decision trees Regressor': DecisionTreeRegressor(random_state=0),
    'Decision trees Regressor  AdaBoost': AdaBoostRegressor(DecisionTreeRegressor(random_state=0),
                                                            n_estimators=10, random_state=0)
    # ,'Support vector machines with Gaussian kernel':SVC(kernel='sigmoid')
}
clf = models.copy()
# LR=logistic Regression
# NB-B=Bernoulli Naive Bayes
# NB-M=Multinomial Naive Bayes
# RF=RandomForestClassifier
# SVM-L=Support vector machines with linear kernel
# SVM-P=Support vector machines with polynomial kernel
powers = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0, 1e+1, 1e+2, 1e+3, 1e+4, 1e+5]
# the optimisation parameters for each of the above models

params = {
    'RandomForestClassifier': {"n_estimators": np.arange(10, 110, 10)},
    'K-Nearest-Neighbors Classifier': {'n_neighbors': [1, 3, 5]},
    'LogisticRegression': {'C': powers},
    'Bernoulli Naive Bayes': {'alpha': powers},
    'Multinomial Naive Bayes': {'alpha': powers},
    'Support vector machines with linear kernel': {'C': powers},
    'Support vector machines with polynomial kernel': {'C': powers, 'gamma': powers},
    'Decision trees Classifier': {},
    'Decision trees Regressor': {},
    'Decision trees Regressor  AdaBoost': {},
    'Support vector machines with Gaussian kernel': {'C': powers, 'gamma': powers}
}

from sklearn.model_selection import GridSearchCV

#scores = ['accuracy', 'recall']  # recall se traduce in rata de prindere a spamului
scores = ['accuracy']  # recall se traduce in rata de prindere a spamului


def get_node_depths(tree):
    """
    Get the node depths of the decision tree

    >>> d = DecisionTreeClassifier()
    >>> d.fit([[1,2,3],[4,5,6],[7,8,9]], [1,2,3])
    >>> get_node_depths(d.tree_)
    array([0, 1, 1, 2, 2])
    """

    def get_node_depths_(current_node, current_depth, l, r, depths):
        depths += [current_depth]
        if l[current_node] != -1 and r[current_node] != -1:
            get_node_depths_(l[current_node], current_depth + 1, l, r, depths)
            get_node_depths_(r[current_node], current_depth + 1, l, r, depths)

    depths = []
    get_node_depths_(0, 0, tree.children_left, tree.children_right, depths)
    return np.array(depths)


import datetime
import time


def DT_boosted_fit(X_train, y_train, X_test, y_test, dataset_name):
    """
    fits the list of models to the training data, thereby obtaining in each
    case an evaluation score after GridSearchCV cross-validation
    """
    print("-------------------------------------------------")
    ts = time.time()
    time_stamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%dT%H %M %S')
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    rez = pd.DataFrame(columns=['Data', 'Algoritm', 'Metrica', 'ACC', 'SC', 'BH', 'F1', 'MCC', 'Mean time of predict'])
    list_method_name = ["Decision trees Classifier", 'Decision trees Regressor','Decision trees Regressor  AdaBoost']
    logs_file = open(".\\train_results\\" + "Decision trees" + " " + time_stamp + ".txt", 'w+')
    for method_name in list_method_name:
        print(method_name + " ")
        for score in scores:
            est = models[method_name]
            est_params = params[method_name]
            abori = []
            gscv = GridSearchCV(estimator=est, param_grid=est_params, cv=10, scoring=score, iid=True)  # iid ->True
            gscv.fit(X_train, y_train)
            print(score + " best parameters :{}".format(gscv.best_estimator_))
            # print("cu scorul (accuracy) : "+format(gscv.best_estimator_.score(X_test, y_test))+" \n")
            y_true = y_test
            clf[method_name] = gscv.best_estimator_
            y_pred = gscv.best_estimator_.predict(X_test)

            #abori.append(max(get_node_depths(gscv.best_estimator_.tree_)))
            # print("Cea mai mare adancime:" + format(max(abori)))
            # print("Cea mai mica adancime:" + format(min(abori)))
            # print("Adancimea medie:" + format(np.mean(abori)))
            cm = confusion_matrix(y_true, y_pred)
            TP = cm[0][0]
            FP = cm[0][1]
            FN = cm[1][0]
            TN = cm[1][1]
            # TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
            # tn, fp, fn, tp
            MCC = matthews_corrcoef(y_true, y_pred)
            F1 = f1_score(y_true, y_pred)
            ACC = accuracy_score(y_true, y_pred)
            SC = recall_score(y_true, y_pred)
            BH = float(FN) / float(TP + FN)  # FNR

            model_file = open(".\\models\\" + method_name + " " + time_stamp + ".pkl" , "wb")
            pickle.dump(gscv.best_estimator_, model_file)
            model_file.close()

            n = 1000
            time_list = pd.DataFrame(columns=['RandomForestClassifier',
                                           'Decision trees'],
                                  index=np.arange(n))
            for index in range(1, n):
                start = time.clock()
                y_pred = gscv.best_estimator_.predict(X_test)
                stop = time.clock()
                timp = stop - start
                time_list.loc[index, method_name] = timp
            time_mean = time_list[method_name].mean()
            rez = rez.append(pd.Series([dataset_name, method_name, score, ACC, SC, BH, F1, MCC, time_mean], index=rez.columns),
                             ignore_index=True)

            logs_file.write(dataset_name + "-data set name \n ")
            logs_file.write(score + " best parameters :{}".format(gscv.best_estimator_))
            logs_file.write("\n Confusion matrix:\n")
            logs_file.write("| TP:" + format(TP) + " FP:" + format(FP) + " |\n")
            logs_file.write("| FN:" + format(FN) + " TN:" + format(TN) + " |\n")
            logs_file.write("Acc :" + format(ACC))
            logs_file.write("\n SC :" + format(SC))
            logs_file.write("\n BH :" + format(BH))
            logs_file.write("\n F-measure :" + format(F1))
            logs_file.write("\n MCC :" + format(MCC))
            logs_file.write("\n Mean time of predict :" + format(time_mean))
            logs_file.write(" \n \n")
    logs_file.write("\n table \n")
    logs_file.write(format(rez))
    logs_file.close()
    return rez

