import pandas as pd
from numeric_data_handler import preprocess_numeric_data
from category_data_handler import preprocess_category_data, feature_encoding

# Preprocessing numeric and category data
credit_data = pd.read_csv('../dataset/original/original.csv')
credit_data = preprocess_numeric_data(credit_data)
credit_data = preprocess_category_data(credit_data)
# Split Unknown value into another file
valid_data_index = credit_data[(credit_data['Education_Level'] != 'Unknown') &
                          (credit_data['Marital_Status'] != 'Unknown') &
                          (credit_data['Income_Category'] != 'Unknown')].index
unknown_data = pd.DataFrame(credit_data.drop(labels=list(valid_data_index), axis=0))
valid_data = credit_data.iloc[valid_data_index]
unknown_data.to_csv('../dataset/original/unknown_data.csv', index=False)
valid_data.to_csv('../dataset/original/valid_data.csv', index=False)

# Save valid data for model testing
valid_data.reset_index(drop=True, inplace=True)
category_feature = ['Education_Level', 'Marital_Status', 'Income_Category']
for feature in category_feature:
    valid_data = pd.concat([valid_data, feature_encoding(feature, valid_data)], axis=1)
    valid_data.drop(columns=feature, axis=1, inplace=True)
valid_data.to_csv('../dataset/original/extracted.csv', index=False)








