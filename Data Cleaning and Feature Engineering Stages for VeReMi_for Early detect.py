#This code excute Data Cleaning and Feature Engineering Stages .. . 

# The libraries we need .. . 


import os
import pandas as pd
import numpy as np


def is_float(value):
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False

    
#Target_file_loc  = r"E:\VeReMi Dataset\VeReMi and BSMList\Main_data_shuffled_CIC_Features.csv"
Target_file_loc = r"E:\VeReMi Dataset\Some VeReMi files\jsonoutput-9-7-A0-1-0.csv"
 

# **************************************  

## Reading a CSV file with low_memory set to False
#Data_target_df = pd.read_csv(Target_file_loc)

#Data_target_df = pd.read_csv(Target_file_loc, low_memory=False)
 
#Data_target_df = pd.read_csv(Target_file_loc, error_bad_lines=False)   # Source - https://stackoverflow.com/a

Data_target_df = pd.read_csv(Target_file_loc , on_bad_lines='skip')

Data_target_df.info()  


'''
 0   type        413 non-null    int64  
 1   rcvTime     413 non-null    float64
 2   vehicleId   413 non-null    int64  
 3   pos         413 non-null    object 
 4   pos_noise   413 non-null    object 
 5   spd         413 non-null    object 
 6   spd_noise   413 non-null    object 
 7   acl         413 non-null    object 
 8   acl_noise   413 non-null    object 
 9   hed         413 non-null    object 
 10  hed_noise   413 non-null    object 
 11  sender_GPS  413 non-null    object
 '''

print(" >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>. .. .. .. .")

#get file size  # Conversion to kilobytes, megabytes  .. .

file_size_bytes = os.path.getsize(Target_file_loc)
file_size_kb = file_size_bytes / 1024
file_size_mb = file_size_kb / 1024


print("Sample Size is :", file_size_mb, "MB")

#The problem is with csv files in CICDDos 2019, there are leading whistespaces in column names, because of which the key error is coming .. .

Data_target_df.columns = Data_target_df.columns.str.strip()


#print("analyze class distribution ", Data_target_df.groupby("Label").size())

#print("analyze class distribution ", Data_target_df.groupby("Label").size())


#print("analyze class distribution ", Data_target_df.groupby("label").size())

#print("analyze class distribution ", Data_target_df.groupby("binary_label_encoded").size())



#Data Cleaning and Feature Engineering

#There are some columns that are not really useful and hence we will proceed to drop them.
#Also, there are some missing values so let’s drop all those rows with empty values:

print(Data_target_df.info())


print(" **************************************")


print("DataFrame  after modified  >>> ")

Data_target_df.head()

# The noisy data rectifying step:
#Removing duplicate records can help reduce noise and redundancy in our dataset.

# Remove duplicate rows:  
Data_target_df = Data_target_df.drop_duplicates()

# Remove anomaly rows:  
# Delete the rows containing float values ​​in the 'vehicleId' column .. .
mask = Data_target_df['vehicleId'].apply(lambda x: not is_float(x) or float(x).is_integer())
Data_target_df = Data_target_df[mask]


# The null value data rectifying step:
#Removing rows or columns with a significant amount of missing data. 
#Remove rows with missing values:  
Data_target_df = Data_target_df.dropna()


# The infinity data values rectifying step:
Data_target_df.replace([np.inf, -np.inf], np.nan)
Data_target_df.dropna(inplace=True)


# label encoding
from sklearn.preprocessing import LabelEncoder
for col in Data_target_df.columns:
    le = LabelEncoder()
    Data_target_df[col] = le.fit_transform(Data_target_df[col])
Data_target_df.info()


# save target dataframe to new location
Data_target_df.to_csv(
         r"E:\VeReMi Dataset\Some VeReMi files\jsonoutput-9-7-A0-1-0_cleaned.csv",
        index=False)

 



