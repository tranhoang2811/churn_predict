import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, StratifiedKFold
import pandas as pd
import numpy as np

# Normal testing
credit_data = pd.read_csv('../../dataset/smote/balance_data_without_cluster.csv')
y = credit_data['Churn_Label']
X = credit_data.drop(columns='Churn_Label')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
classifier = LogisticRegression(solver='liblinear', penalty='l1')
classifier.fit(X_train, y_train)
y_predict = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_predict)*100
print('Accuracy: %.2f %%' % accuracy)

# Testing with stratified
stratified = StratifiedKFold(n_splits=10)
score = cross_val_score()
# Testing normal split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’} not multiclass problems => use one of liblinear
# classifier = LogisticRegression(solver='liblinear', penalty='l1', max_iter=200)
# classifier.fit(X_train, y_train)
# predict = classifier.predict(X_test)
# print('Accuracy: %.2f %%' % (accuracy_score(y_test, predict)*100))
#
# # Evaluate model performance with stratified k fold
# stratified_kfold = StratifiedKFold(n_splits=5)
# scores = cross_val_score(classifier, X_train, y_train, cv=stratified_kfold)
# print('Average cross-validation score %.2f %%' % (scores.mean()*100))

# Visualize model accuracy and cross-validation
# stratified_kfold = StratifiedKFold(n_splits=5)
# predicts_accuracy = []
# cross_mean_score = []
#
# for test_index in range(1, 21):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#     clf = LogisticRegression(solver='liblinear', penalty='l2', max_iter=200, C=1e6)
#     clf.fit(X_train, y_train)
#     predict = clf.predict(X_test)
#     predicts_accuracy.append(accuracy_score(y_test, predict))
#     scores = cross_val_score(clf, X_train, y_train, cv=stratified_kfold)
#     cross_mean_score.append(scores.mean())
#
# cross_test_mean = np.mean(cross_mean_score)
# predict_test_mean = np.mean(predicts_accuracy)
# print('Cross validation mean after 20 loop: %.2f %%' % (cross_test_mean*100))
# print('Max and min cross validation score: %.2f %.2f %%' % (max(cross_mean_score)*100, min(cross_mean_score)*100))
# print('Test predict accuracy mean after 20 loop: %.2f %%' % (predict_test_mean*100))
# print('Max and min predict accuracy: %.2f %.2f %%' % (max(predicts_accuracy)*100, min(predicts_accuracy)*100))
#
# plt.plot(range(1, 21), predicts_accuracy, label='predict accuracy')
# plt.plot(range(1, 21), cross_mean_score, label='cross validation accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('test index')
# plt.legend()
# plt.show()
