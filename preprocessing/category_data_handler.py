from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd


# Transform category feature
def transform_feature(data):
    return LabelEncoder().fit_transform(data)


# Encoding category feature function
def feature_encoding(feature_name, credit_data):
    feature_label_encoder = LabelEncoder()
    feature_label_encoder.fit(credit_data[feature_name])
    feature_one_hot_encoder = OneHotEncoder()
    feature_array = feature_one_hot_encoder.fit_transform(credit_data[[feature_name]]).toarray()
    feature_label_names = []
    for label_name in list(feature_label_encoder.classes_):
        feature_label_names.append(f'{feature_name}_{label_name}')
    features = pd.DataFrame(feature_array, columns=feature_label_names)
    return features


def preprocess_category_data(credit_data):
    # Replace College = Graduate & Divorced = Single
    credit_data.loc[credit_data['Education_Level'] == 'College', 'Education_Level'] = 'Graduate'
    credit_data.loc[credit_data['Marital_Status'] == 'Divorced', 'Marital_Status'] = 'Single'
    # Transform Gender and Attrition flag
    credit_data['Gender_Label'] = transform_feature(credit_data['Gender'])
    credit_data['Churn_Label'] = transform_feature(credit_data['Attrition_Flag'])
    credit_data.reset_index(drop=True, inplace=True)
    # Set Churn = 1, Stay = 0
    for index, row in credit_data.iterrows():
        credit_data.loc[index, 'Churn_Label'] = 0 if credit_data.loc[index, 'Churn_Label'] == 1 else 1
    # Encode Card feature and concat into original dataframe
    credit_data = pd.concat([credit_data, feature_encoding('Card_Category', credit_data)], axis=1)
    # Drop unnecessary data
    credit_data.drop('Card_Category', axis=1, inplace=True)
    credit_data.drop('Gender', axis=1, inplace=True)
    credit_data.drop('Attrition_Flag', axis=1, inplace=True)
    return credit_data
