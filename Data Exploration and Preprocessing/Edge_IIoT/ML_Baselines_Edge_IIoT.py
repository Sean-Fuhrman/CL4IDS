#%%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
#Load Custom preprocessed data
x_train = np.load('../../Datasets/Edge-IIoT-pre-processed/Default Preprocessing/Multi/X_train.npy')
y_train = np.load('../../Datasets/Edge-IIoT-pre-processed/Default Preprocessing/Multi/Y_train.npy')
x_val = np.load('../../Datasets/Edge-IIoT-pre-processed/Default Preprocessing/Multi/X_val.npy')
y_val = np.load('../../Datasets/Edge-IIoT-pre-processed/Default Preprocessing/Multi/Y_val.npy')
x_test = np.load('../../Datasets/Edge-IIoT-pre-processed/Default Preprocessing/Multi/X_test.npy')
y_test = np.load('../../Datasets/Edge-IIoT-pre-processed/Default Preprocessing/Multi/Y_test.npy')

#Print shapes
print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)
print(x_test.shape)
print(y_test.shape)

y_train = np.argmax(y_train, axis=1)
y_val = np.argmax(y_val, axis=1)
y_test = np.argmax(y_test, axis=1)


#Print shapes
print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)
print(x_test.shape)
print(y_test.shape)
#%%

from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier

algorithms = [ "RF","LR", "SVM", "KNN"]

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
    clf.fit(x_train, y_train)
    
    #Evaluate the model
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix, classification_report

    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)

    print("Train Accuracy: ", accuracy_score(y_train, y_train_pred))

    print("Test Accuracy: ", accuracy_score(y_test, y_test_pred))

    # classification report
    print(classification_report(y_test, y_test_pred))

    # save classification report
    report = classification_report(y_test, y_test_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv('./results/'+alorithm+'_classification_report.csv')


# %%


