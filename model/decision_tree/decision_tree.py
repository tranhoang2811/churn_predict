from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

credit_data = pd.read_csv('../../dataset/extracted.csv')
y = credit_data['Churn_Label']
X = credit_data.drop(columns='Churn_Label')
X.drop(columns='CLIENTNUM', inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

predict = classifier.predict(X_test)
print('Accuracy: %.2f %%' % (accuracy_score(y_test, predict)*100))
