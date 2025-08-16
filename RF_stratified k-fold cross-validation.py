#Importing Required Libraries:

from statistics import mean, stdev
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn import linear_model


import time
import os

import types

import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier



############################################################  CIC-DDoS2019_CSV  from original   - -  All files Available to test ................................

#Target_file_loc = r"E:\Cic-DDos2019 Original\03-11\MSSQL_Pre.csv"

Target_file_loc = r"E:\Cic-DDos2019 Original\03-11\MSSQL_undersampling.csv"


#Target_file_loc = r"E:\Cic-DDos2019 Original\03-11\UDP_Pre.csv"

#Target_file_loc =  r"E:\Cic-DDos2019 Original\03-11\UDP_undersampling.csv"

 
###########################################################################################


Data_target_df = pd.read_csv(Target_file_loc)

Data_target_df.info()
print(" >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>. .. .. .. .")

#get file size  # Conversion to kilobytes, megabytes  .. .

file_size_bytes = os.path.getsize(Target_file_loc)
file_size_kb = file_size_bytes / 1024
file_size_mb = file_size_kb / 1024


print("File Size is :", file_size_mb, "MB")


#The problem is with csv files in CICDDos 2019, there are leading whistespaces in column names, because of which the key error is coming.
Data_target_df.columns = Data_target_df.columns.str.strip()

print("analyze class distribution ", Data_target_df.groupby("Label").size())


print(" **************************************")



# X .. features , y .. target


############ X,y ...   CIC-DDos-2019 from Original: 

X = Data_target_df[[ 'Packet Length Mean', 'Average Packet Size', 'Bwd Packet Length Min', 'Fwd Packets/s' , 'Min Packet Length', 'Down/Up Ratio']]


#X = Data_target_df[[ 'Packet Length Mean', 'Average Packet Size', 'Bwd Packet Length Min', 'Fwd Packets/s' ]]


#X = Data_target_df[[ 'Packet Length Mean', 'Bwd Packet Length Min', 'Fwd Packets/s']]  # # 3Features  from Second  Approach  .. .


#X = Data_target_df[[ 'Packet Length Mean', 'Average Packet Size']]   #  2 Features  from the First Approach .. .



y = Data_target_df['Label']  

###############################################################

print(" **************************************")


#Feature Scaling (Normalization):
'''
    MinMaxScaler(): scales features to a range between 0 and 1
    fit_transform(x): fits scaler on data and applies transformation
    '''

scaler = preprocessing.MinMaxScaler()
x_scaled = scaler.fit_transform(X)

#Model and K-Fold Object Setup:

forest = RandomForestClassifier ()


skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
lst_accu_stratified = []


#Applying Stratified K-Fold and Training Model:

'''    skf.split(x, y): splits dataset into stratified train-test indices
    x_train_fold, x_test_fold: features for training and testing
    y_train_fold, y_test_fold: labels for training and testing  '''

start_train = time.time()

for train_index, test_index in skf.split(X, y): 
	x_train_fold, x_test_fold = x_scaled[train_index], x_scaled[test_index]
	y_train_fold, y_test_fold = y[train_index], y[test_index]
	forest.fit(x_train_fold, y_train_fold)
	lst_accu_stratified.append(forest.score(x_test_fold, y_test_fold))

training_time = time.time() - start_train
print(f'Training_time = {training_time}')


#Printing Accuracy Results
'''
    max(): highest accuracy
    min(): lowest accracy
    mean(): average accuracy
 '''

print('List of possible accuracy:', lst_accu_stratified)
print('\nMaximum Accuracy That can be obtained from this model is:',
	max(lst_accu_stratified)*100, '%')
print('\nMinimum Accuracy:',
	min(lst_accu_stratified)*100, '%')
print('\nOverall Accuracy:',
	mean(lst_accu_stratified)*100, '%')
print('\nStandard Deviation is:', stdev(lst_accu_stratified))




















