
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.0** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-machine-learning/resources/bANLa) course resource._
# 
# ---

# ## Assignment 4 - Understanding and Predicting Property Maintenance Fines
# 
# This assignment is based on a data challenge from the Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)). 
# 
# The Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)) and the Michigan Student Symposium for Interdisciplinary Statistical Sciences ([MSSISS](https://sites.lsa.umich.edu/mssiss/)) have partnered with the City of Detroit to help solve one of the most pressing problems facing Detroit - blight. [Blight violations](http://www.detroitmi.gov/How-Do-I/Report/Blight-Complaint-FAQs) are issued by the city to individuals who allow their properties to remain in a deteriorated condition. Every year, the city of Detroit issues millions of dollars in fines to residents and every year, many of these fines remain unpaid. Enforcing unpaid blight fines is a costly and tedious process, so the city wants to know: how can we increase blight ticket compliance?
# 
# The first step in answering this question is understanding when and why a resident might fail to comply with a blight ticket. This is where predictive modeling comes in. For this assignment, your task is to predict whether a given blight ticket will be paid on time.
# 
# All data for this assignment has been provided to us through the [Detroit Open Data Portal](https://data.detroitmi.gov/). **Only the data already included in your Coursera directory can be used for training the model for this assignment.** Nonetheless, we encourage you to look into data from other Detroit datasets to help inform feature creation and model selection. We recommend taking a look at the following related datasets:
# 
# * [Building Permits](https://data.detroitmi.gov/Property-Parcels/Building-Permits/xw2a-a7tf)
# * [Trades Permits](https://data.detroitmi.gov/Property-Parcels/Trades-Permits/635b-dsgv)
# * [Improve Detroit: Submitted Issues](https://data.detroitmi.gov/Government/Improve-Detroit-Submitted-Issues/fwz3-w3yn)
# * [DPD: Citizen Complaints](https://data.detroitmi.gov/Public-Safety/DPD-Citizen-Complaints-2016/kahe-efs3)
# * [Parcel Map](https://data.detroitmi.gov/Property-Parcels/Parcel-Map/fxkw-udwf)
# 
# ___
# 
# We provide you with two data files for use in training and validating your models: train.csv and test.csv. Each row in these two files corresponds to a single blight ticket, and includes information about when, why, and to whom each ticket was issued. The target variable is compliance, which is True if the ticket was paid early, on time, or within one month of the hearing data, False if the ticket was paid after the hearing date or not at all, and Null if the violator was found not responsible. Compliance, as well as a handful of other variables that will not be available at test-time, are only included in train.csv.
# 
# Note: All tickets where the violators were found not responsible are not considered during evaluation. They are included in the training set as an additional source of data for visualization, and to enable unsupervised and semi-supervised approaches. However, they are not included in the test set.
# 
# <br>
# 
# **File descriptions** (Use only this data for training your model!)
# 
#     train.csv - the training set (all tickets issued 2004-2011)
#     test.csv - the test set (all tickets issued 2012-2016)
#     addresses.csv & latlons.csv - mapping from ticket id to addresses, and from addresses to lat/lon coordinates. 
#      Note: misspelled addresses may be incorrectly geolocated.
# 
# <br>
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
# 
# ## Evaluation
# 
# Your predictions will be given as the probability that the corresponding blight ticket will be paid on time.
# 
# The evaluation metric for this assignment is the Area Under the ROC Curve (AUC). 
# 
# Your grade will be based on the AUC score computed for your classifier. A model which with an AUROC of 0.7 passes this assignment, over 0.75 will recieve full points.
# ___
# 
# For this assignment, create a function that trains a model to predict blight ticket compliance in Detroit using `train.csv`. Using this model, return a series of length 61001 with the data being the probability that each corresponding ticket from `test.csv` will be paid, and the index being the ticket_id.
# 
# Example:
# 
#     ticket_id
#        284932    0.531842
#        285362    0.401958
#        285361    0.105928
#        285338    0.018572
#                  ...
#        376499    0.208567
#        376500    0.818759
#        369851    0.018528
#        Name: compliance, dtype: float32

# In[3]:

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



