import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, StratifiedKFold
import pandas as pd
import matplotlib.pyplot as plt


def point_weight(distance):
    return np.exp(-distance**2/0.4)


# Model testing
credit_data = pd.read_csv('../../dataset/smote/balance_data_without_cluster.csv')
y = credit_data['Churn_Label']
X = credit_data.drop(columns='Churn_Label')

# # Normal testing
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# classifier = KNeighborsClassifier(n_neighbors=7, p=2, weights=point_weight)
# classifier.fit(X_train, y_train)
# y_predict = classifier.predict(X_test)
# accuracy = accuracy_score(y_test, y_predict)*100
# print('Accuracy: %.2f %%' % accuracy)

# Testing with stratified
stratified = StratifiedKFold(n_splits=10, shuffle=False)

neighbors = range(1, 21)
accuracies_mean = []

for neighbor in neighbors:
    classifier_with_stratified = KNeighborsClassifier(n_neighbors=neighbor, p=1, weights='distance')
    cross_score = round(cross_val_score(classifier_with_stratified, X, y, cv=stratified).mean()*100, 2)
    accuracies_mean.append(cross_score)

print(f'Max score: {max(accuracies_mean)}\nNeighbor: {accuracies_mean.index(max(accuracies_mean))+1}')
print(f'Min score: {min(accuracies_mean)}\nNeighbor: {accuracies_mean.index(min(accuracies_mean))+1}')

plt.plot(neighbors, accuracies_mean)
plt.grid(True)
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()
