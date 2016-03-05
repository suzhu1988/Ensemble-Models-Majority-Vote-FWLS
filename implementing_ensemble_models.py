'''
Ensemble models:  Implement majority vote classifier, and a FWLS logistic regression with the Iris dataset
'''

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split

#Load the iris dataset and split it into training and test sets 60:40
iris = datasets.load_iris()  #Note that the classes are in the target array, and the features are in the data array 
x = iris.data[:]
y = iris.target[:]
y = LabelEncoder().fit_transform(y)  #Encode class labels as integers (setosa=0, versicolor=2, virginica=3)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=1)

#Train 3 classifiers: SVM, Random Forest, and KNN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

#SVM pipeline
pipe_svc = Pipeline([('scl', StandardScaler()), ('clf', SVC(random_state=1, probability=True))])
#Specify values for the parameter c, which determines how large the margin is for the SVM
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
#List the hyperparameters to tune for the SVM
param_grid = [{'clf__C': param_range, 'clf__kernel': ['linear']},
                 {'clf__C': param_range, 'clf__gamma': param_range, 'clf__kernel': ['rbf']}]
#Perform grid search cross validation on the parameters listed in param_grid, using accuracy as the measure of fit and number of folds (CV) = 5
gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring='accuracy', cv=5)
#Fit the cross validated SVM to the training dataset and view the parameters of the best model
gs.fit(x_train, y_train)
best_SVM_params = gs.best_params_
print("Optimal SVM parameters are: ", best_SVM_params)
scores = cross_val_score(gs, x_train, y_train, scoring='accuracy', cv=5)
print("Average SVM Accuracy: %.3f +/- %.3f" % (np.mean(scores), np.std(scores)))
#Build SVM with optimized parameters listed above
classifier1 = SVC(C=gs.best_params_['clf__C'], kernel=gs.best_params_['clf__kernel'], gamma=gs.best_params_['clf__gamma'])

#Random forest pipeline - scaling not needed here
pipe_rf = Pipeline([('clf', RandomForestClassifier(criterion='entropy', random_state=1, n_jobs=-1))])
#Specify values for the number of trees in the forest and the max features to use for each split
param_range_trees = [10, 50, 100, 200, 500]
param_range_features = [2, 3, 4]
param_grid = [{'clf__n_estimators': param_range_trees, 'clf__max_features': param_range_features}]
#Perform grid search cross validation on the parameters listed in param_grid, using accuracy as the measure of fit and number of folds (CV) = 5
gs = GridSearchCV(estimator=pipe_rf, param_grid=param_grid, scoring='accuracy', cv=5)
#Fit the cross validated RF to the training dataset and view the parameters of the best model
gs.fit(x_train, y_train)
best_rf_params = gs.best_params_
print("Optimal RF parameters are: ", best_rf_params)
scores = cross_val_score(gs, x_train, y_train, scoring='accuracy', cv=2)
print("Average RF Accuracy: %.3f +/- %.3f" % (np.mean(scores), np.std(scores)))
#Build SVM with optimized parameters listed above
classifier2 = RandomForestClassifier(n_estimators=gs.best_params_['clf__n_estimators'], max_features=gs.best_params_['clf__max_features'], criterion='entropy', random_state=1, n_jobs=-1)

#KNN pipeline
pipe_knn = Pipeline([('scl', StandardScaler()), ('clf', KNeighborsClassifier())])
#Specify values for number of neighbors to tune
param_range = [1, 2, 5, 9]
param_grid = [{'clf__n_neighbors': param_range}]
#Perform grid search cross validation on the parameters listed in param_grid, using accuracy as the measure of fit and number of folds (CV) = 5
gs = GridSearchCV(estimator=pipe_knn, param_grid=param_grid, scoring='accuracy', cv=5)
#Fit the cross validated KNN to the training dataset and view the parameters of the best model
gs.fit(x_train, y_train)
best_KNN_params = gs.best_params_
print("Optimal KNN parameters are: ", best_KNN_params)
scores = cross_val_score(gs, x_train, y_train, scoring='accuracy', cv=5)
print("Average KNN Accuracy: %.3f +/- %.3f" % (np.mean(scores), np.std(scores)))
#Build SVM with optimized parameters listed above
classifier3 = KNeighborsClassifier(n_neighbors=gs.best_params_['clf__n_neighbors'])


"""
Majority vote ensemble classifier begins here
"""

import majority_vote_classifier as mvc

#Apply standarized scaling to the SVM and KNN models
pipe1 = Pipeline([['sc', StandardScaler()], ['clf', classifier1]])
pipe3 = Pipeline([['sc', StandardScaler()], ['clf', classifier3]])

#Set labels for ease of distinguishing models
clf_labels = ['SVM', 'Random Forest', 'KNN', 'Majority Vote Classifier']

#Now implement the majority voting classifier
mv_clf = mvc.MajorityVoteClassifier(classifiers=[pipe1, classifier2, pipe3])

#(Optional) Run 10 fold cross validation on the individual models and the ensemble
all_clf = [pipe1, classifier2, pipe3, mv_clf]
for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf, X=x_train, y=y_train, scoring='accuracy', cv=10)
    print("Average accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

#Fit the majority vote ensemble model to the training set
mv_clf.fit(x_train, y_train)
#Predict classifications of test set observations
y_test_pred = mv_clf.predict(x_test)

#Build a confusion matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

confmat = confusion_matrix(y_true=y_test, y_pred=y_test_pred)
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i,j], va='center', ha='center')
plt.xlabel('predicted values')
plt.ylabel('true values')
plt.title('Confusion Matrix: Majority Vote Ensemble Classifier')
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.show()


"""
FWLS ensemble classifier begins here
"""

import fwls_logistic_reg_classifier as fwls
from sklearn.linear_model import LogisticRegression

#Apply standarized scaling to the SVM and KNN models
pipe1 = Pipeline([['sc', StandardScaler()], ['clf', classifier1]])
pipe3 = Pipeline([['sc', StandardScaler()], ['clf', classifier3]])

#Now implement the FWLS classifier
fwls_clf = fwls.FWLS_Classifier(classifiers=[pipe1, classifier2, pipe3])

#Fit the component models to the training set
fwls_clf.fit(x_train, y_train)
#Predict classifications of training set observations for each component model
y_train_pred = fwls_clf.predict(x_train)
#Predict classifications of test set observations for each component model
y_test_pred = fwls_clf.predict(x_test)

#Fit a logistic regression to the predicted y values for each model
pipe_lr = Pipeline([('clf', LogisticRegression(random_state=1))])
pipe_lr.fit(y_train_pred, y_train)
#Predict the y values of the test set using the fitted FWLS ensemble model (the prediction is made with the composite predictions of the component models)
y_pred = pipe_lr.predict(y_test_pred)

#View the test accuracy of the FWLS ensemble model                  
print('Test Accuracy of FWLS ensemble model: %.3f' % pipe_lr.score(y_test_pred, y_test))

#Build a confusion matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i,j], va='center', ha='center')
plt.xlabel('predicted values')
plt.ylabel('true values')
plt.title('Confusion Matrix: FWLS Ensemble Classifier')
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.show()

