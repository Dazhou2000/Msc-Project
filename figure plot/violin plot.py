import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns

# Specify the directory containing the CSV files
directory_path = r'C:\Users\ASUS\Desktop\MSc dataset'

# Use glob to get a list of all CSV files in the directory
csv_files = glob.glob(os.path.join(directory_path, '*.csv'))

# Sort the files to ensure correct order
csv_files.sort()
for file in csv_files:
    print(file)

# Initialize lists to store each patient's data
patients_heart_rate = []
patients_seizure = []

# Loop through the files to read each patient's data
# Assuming each patient has exactly two files: one for heart rate and one for seizure
for i in range(0, len(csv_files), 2):
    if i + 1 < len(csv_files):  # Ensure there's a pair of files
        heart_rate_file = csv_files[i]
        seizure_file = csv_files[i + 1]

        # Read the heart rate CSV file
        heart_rate_df = pd.read_csv(heart_rate_file)

        # Read the seizure CSV file
        seizure_df = pd.read_csv(seizure_file)

        # Append the data to the respective lists
        patients_heart_rate.append(heart_rate_df)
        patients_seizure.append(seizure_df)

        # Optionally, print the first few rows to check
        patient_num = i // 2 + 1
        print(f"Patient {patient_num} Heart Rate Data:")
        print(heart_rate_df.head())
        print(f"Patient {patient_num} Seizure Data:")
        print(seizure_df.head())

# Accessing data of patient 1
p1_heart_rate = patients_heart_rate[3]
p1_seizure = patients_seizure[3]

recording_time = p1_heart_rate.iloc[:,1]
heart_rate =p1_heart_rate.iloc[:,2]

seizure_time = p1_seizure.iloc[:,1]

'''
fig = px.violin(p1_heart_rate.iloc[:,2], y="value")
fig.show()
'''
# Specify the range: e.g. an hour before seizure, half an hour after seizure
before_index = 3600/300
after_index = 1800/300

# point of interest, assume the ith seizure point
list_during=[]
list_before=[]
list_after=[]
for k in range(len(seizure_time)):
 point_of_interest = seizure_time[k]
 interest_index = round(point_of_interest/300)
 start_index = max(0, interest_index-before_index)
 end_index = min(len(recording_time)-1, interest_index+after_index)


 # Segment the dataset based on the range
 before_range = p1_heart_rate.iloc[int(start_index):interest_index+1]
 after_range = p1_heart_rate.iloc[interest_index:int(end_index)+1]
 during_range= p1_heart_rate.iloc[int(start_index): int(end_index)+1]
 list_during.extend(during_range.iloc[:,2])
 list_before.extend(before_range.iloc[:, 2])
 list_after.extend(after_range.iloc[:, 2])

# find the missing points from dataset
index = [i for i, value in enumerate(p1_heart_rate.iloc[:,2]) if value != value]
index_during =[i for i, value in enumerate(list_during) if value != value]
index_before =[i for i, value in enumerate(list_before) if value != value]
index_after =[i for i, value in enumerate(list_after) if value != value]
# create a new list without missing points
clean = [value for i, value in enumerate(p1_heart_rate.iloc[:,2]) if i not in index]
clean_during = [value for i, value in enumerate(list_during) if i not in index]
clean_before = [value for i, value in enumerate(list_before) if i not in index]
clean_after = [value for i, value in enumerate(list_after) if i not in index]

df1 = pd.DataFrame({'Patient3': 'Whole', 'heart rate': clean})
df2 = pd.DataFrame({'Patient3': 'During seizure', 'heart rate': clean_during})
df3 = pd.DataFrame({'Patient3': 'Before seizure', 'heart rate': clean_before})
df4 = pd.DataFrame({'Patient3': 'After seizure', 'heart rate': clean_after})

# Concatenate the DataFrames
data_combined = pd.concat([df1, df2, df3, df4])

# Create the violin plot
plt.figure(1)
sns.violinplot(x='Patient3', y='heart rate', data=data_combined)

plt.show()