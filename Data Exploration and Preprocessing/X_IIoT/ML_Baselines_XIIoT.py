#%%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
#Load Custom preprocessed data
X_train = np.load('../../Datasets/X-IIoT-pre-processed/Custom PreProcessed/X_train.npy')
X_val = np.load('../../Datasets/X-IIoT-pre-processed/Custom PreProcessed/X_val.npy')
X_test = np.load('../../Datasets/X-IIoT-pre-processed/Custom PreProcessed/X_test.npy')
Y_1_train = np.load('../../Datasets/X-IIoT-pre-processed/Custom PreProcessed/Y_1_train.npy')
Y_1_val = np.load('../../Datasets/X-IIoT-pre-processed/Custom PreProcessed/Y_1_val.npy')
Y_1_test = np.load('../../Datasets/X-IIoT-pre-processed/Custom PreProcessed/Y_1_test.npy')

#Print shapes
print(X_train.shape)
print(X_val.shape)
print(X_test.shape)
print(Y_1_train.shape)
print(Y_1_val.shape)
print(Y_1_test.shape)

#%%
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier

algorithms = [ "LR", "SVM", "KNN"]

for alorithm in algorithms:
    if alorithm == "DT":
        clf= tree.DecisionTreeClassifier()
    elif alorithm == "NB":
        clf = GaussianNB()
    elif alorithm == "RF":
        clf = RandomForestClassifier()
    elif alorithm == "SVM":
        clf = SVC()
    elif alorithm == "KNN":
        clf = KNeighborsClassifier()
    elif alorithm =="LR":
        clf = LogisticRegression()
    else:
        print("Invalid Algorithm")
        break
    print("Algorithm: ", alorithm)
    clf.fit(X_train, Y_1_train)
    
    #Evaluate the model
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix, classification_report

    Y_1_train_pred = clf.predict(X_train)
    Y_1_test_pred = clf.predict(X_test)

    print("Train Accuracy: ", accuracy_score(Y_1_train, Y_1_train_pred))

    print("Test Accuracy: ", accuracy_score(Y_1_test, Y_1_test_pred))

    # classification report
    print(classification_report(Y_1_test, Y_1_test_pred))

    # save classification report
    report = classification_report(Y_1_test, Y_1_test_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv('./results/'+alorithm+'_classification_report.csv')


# %%


