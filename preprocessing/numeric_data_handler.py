from sklearn.preprocessing import MinMaxScaler, StandardScaler


# Scale numeric data
def preprocess_numeric_data(credit_data):
    standard_scaler = StandardScaler().fit_transform(credit_data[['Customer_Age', 'Months_on_book']])
    min_max_scaler = MinMaxScaler().fit_transform(credit_data[['Credit_Limit', 'Total_Revolving_Bal', 'Avg_Open_To_Buy',
                                                               'Total_Trans_Amt', 'Total_Trans_Ct']])

    credit_data[['Customer_Age', 'Months_on_book']] = standard_scaler
    credit_data[['Credit_Limit', 'Total_Revolving_Bal', 'Avg_Open_To_Buy', 'Total_Trans_Amt', 'Total_Trans_Ct']] \
        = min_max_scaler
    return credit_data
