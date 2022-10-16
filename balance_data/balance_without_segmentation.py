from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.metrics import accuracy_score

credit_data = pd.read_csv('../dataset/original/extracted.csv')

y = credit_data['Churn_Label']
print(y)
X = credit_data.drop(columns='Churn_Label')
X.drop(columns='CLIENTNUM', inplace=True)

smote = SMOTE()
X_smote, y_smote = smote.fit_resample(X, y)
print(y_smote.value_counts())
balance_data = pd.concat([X_smote, y_smote], axis=1)
balance_data.to_csv('../dataset/smote/balance_data_without_cluster.csv', index=False)
