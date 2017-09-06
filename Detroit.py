
# coding: utf-8
# 
# **Data fields**
# 
# train.csv & test.csv
# 
#     ticket_id - unique identifier for tickets
#     agency_name - Agency that issued the ticket
#     inspector_name - Name of inspector that issued the ticket
#     violator_name - Name of the person/organization that the ticket was issued to
#     violation_street_number, violation_street_name, violation_zip_code - Address where the violation occurred
#     mailing_address_str_number, mailing_address_str_name, city, state, zip_code, non_us_str_code, country - Mailing address of the violator
#     ticket_issued_date - Date and time the ticket was issued
#     hearing_date - Date and time the violator's hearing was scheduled
#     violation_code, violation_description - Type of violation
#     disposition - Judgment and judgement type
#     fine_amount - Violation fine amount, excluding fees
#     admin_fee - $20 fee assigned to responsible judgments
# state_fee - $10 fee assigned to responsible judgments
#     late_fee - 10% fee assigned to responsible judgments
#     discount_amount - discount applied, if any
#     clean_up_cost - DPW clean-up or graffiti removal cost
#     judgment_amount - Sum of all fines and fees
#     grafitti_status - Flag for graffiti violations
#     
# train.csv only
# 
#     payment_amount - Amount paid, if any
#     payment_date - Date payment was made, if it was received
#     payment_status - Current payment status as of Feb 1 2017
#     balance_due - Fines and fees still owed
#     collection_status - Flag for payments in collections
#     compliance [target variable for prediction] 
#      Null = Not responsible
#      0 = Responsible, non-compliant
#      1 = Responsible, compliant
#     compliance_detail - More information on why each ticket was marked compliant or non-compliant
# 
# 
# ___

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

def blight_model():
    # Training Data
    train = pd.read_csv('train.csv',encoding='iso-8859-1')
    train = train.set_index('ticket_id')
    train.dropna(subset=['compliance'],inplace=True)
    test = pd.read_csv('test.csv')
    ticket_id_test  = np.array(test['ticket_id'])
    #Test data
    test = test.set_index('ticket_id')
    train.dropna(subset=['compliance'], inplace=True)
    nafe = ['inspector_name', 'violator_name','violation_zip_code', 'violation_street_number', 'violation_street_name','mailing_address_str_number', 'mailing_address_str_name', 'city','state', 'zip_code', 'non_us_str_code', 'country','violation_description','admin_fee', 'state_fee', 'late_fee']
    #Removing non relevent fields
    for feature in nafe:
        test.drop(feature, axis=1, inplace=True)
        train.drop(feature, axis=1, inplace=True)
    #Date Time Fields
    for column_name in ['ticket_issued_date', 'hearing_date']:
    # test
        day_time = pd.to_datetime(test[column_name])
        test.drop(column_name, axis=1, inplace=True)
        test[column_name+'_month'] = np.array(day_time.dt.month)
        test[column_name+'_year'] = np.array(day_time.dt.year)
        test[column_name+'_day'] = np.array(day_time.dt.day)
        test[column_name+'_dayofweek'] = np.array(day_time.dt.dayofweek) 
    # train
        day_time = pd.to_datetime(train[column_name])
        train.drop(column_name, axis=1, inplace=True)
        train[column_name+'_month'] = np.array(day_time.dt.month)
        train[column_name+'_year'] = np.array(day_time.dt.year)
        train[column_name+'_day'] = np.array(day_time.dt.day)
        train[column_name+'_dayofweek'] = np.array(day_time.dt.dayofweek)
        category_cols = test.select_dtypes(exclude=['float', 'int']).columns
        len_train = len(train)
    temp_concat = pd.concat((train[category_cols], test[category_cols]), axis=0)

#  filtering on violation_code to make it more manageable
    temp_concat['violation_code'] = temp_concat['violation_code'].apply(lambda x: x.split(' ')[0])
    temp_concat['violation_code'] = temp_concat['violation_code'].apply(lambda x: x.split('(')[0])
    temp_concat['violation_code'][temp_concat['violation_code'].apply(lambda x: x.find('-')<=0)] = np.nan

# Make all codes with < 10 occurrences null
    counts = temp_concat['violation_code'].value_counts()
    temp_concat['violation_code'][temp_concat['violation_code'].isin(counts[counts < 10].index)] = np.nan
    for column_name in category_cols:
        dummies = pd.get_dummies(temp_concat[column_name])
        temp_concat[dummies.columns] = dummies
        temp_concat.drop(column_name, axis=1, inplace=True)
        train.drop(column_name, axis=1, inplace=True)
        test.drop(column_name, axis=1, inplace=True)
    #Training and test data 
    train[temp_concat.columns] = temp_concat.loc[train.index]
    test[temp_concat.columns] = temp_concat.loc[test.index]
    X = train[test.columns]
    y = np.array(train[['compliance']]).ravel()
    X = X.replace([np.inf, -np.inf], np.nan)
    X[pd.isnull(X)] = 0
    Xtest = test[test.columns]
    Xtest = Xtest.replace([np.inf, -np.inf], np.nan)
    Xtest[pd.isnull(Xtest)] = 0
    #Logistic Regression
    clf = LogisticRegression(solver='sag', tol=1e-1, C=1.e4 / train[test.columns].shape[0]).fit(X,y)
    #Predicting Probabaility for test set
    y_pred = clf.predict_proba(Xtest)
    s = pd.Series(y_pred[:,1],index = ticket_id_test)
    s.index.name = 'ticket_id'
    s.name = 'compliance'
    return s.astype('float32') # Your answer here


# In[ ]:




# In[ ]:



