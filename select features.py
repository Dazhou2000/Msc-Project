from sklearn.ensemble import RandomForestClassifier
import time
from extract_features import window_size
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tsfel
from sklearn.inspection import permutation_importance


print(window_size)
# Step 1: Specify the directory containing the CSV files
directory_path = r'C:\Users\ASUS\Desktop\MSc dataset'

# Step 2: Use glob to get a list of all CSV files in the directory

csv_files = glob.glob(os.path.join(directory_path, '*.csv'))

# Step 3: Sort the files to ensure correct order
csv_files.sort()
for file in csv_files:
    print(f"Found file: {file}")

# Initialize dictionaries to store each patient's data
patients_heart_rate = {}
patients_seizure = {}

# Step 4: Loop through the files and categorize them
for file in csv_files:
    filename = os.path.basename(file)
    patient_id = filename.split('_')[0]

    # Read the CSV file
    df = pd.read_csv(file)

    # Add an ID column with the first two letters of the dataset name

    if 'heart' in filename:
        patients_heart_rate[patient_id] = (filename, df)
    elif 'seizure' in filename:
        patients_seizure[patient_id] = (filename, df)

# Step 5: Sort patient IDs and get the dataset for the 8th patient
sorted_patient_ids = sorted(patients_heart_rate.keys())

# # Initialize lists to store each patient's data
choose_patient_id= sorted_patient_ids[0]
heart_rate_name, heart_rate_df = patients_heart_rate[choose_patient_id]
seizure_name, seizure_df = patients_seizure[choose_patient_id]
print(f"choose: {heart_rate_name[:3]} and {seizure_name[:3]}")
print(heart_rate_df.head())
# Accessing data of one patient

p1_heart_rate = heart_rate_df
p1_seizure = seizure_df

# find timestamp of windows
p1_heart_rate['time']= p1_heart_rate['seconds_since_hr_recording'].iloc[(window_size-1):].reset_index(drop=True)

# get windows of extracted features
features = pd.read_csv('extracted_features_P1_ with window_288.csv')
features= features.drop(columns='Unnamed: 0')

#Normalising Features
features = (features-features.mean())/features.std()

# reduce redundancy (drop highly correlated features)
corr_features = tsfel.correlated_features(features)
features.drop(corr_features, axis=1, inplace=True)

# add window timestamp
features['timestamp']=p1_heart_rate['time'].copy()

# find seizure event index and convert to window index
features['seizure_event'] = 0
def find_nearest_heart_rate_time(seizure_time):
    time_diffs = abs(features['timestamp'] - seizure_time)
    return time_diffs.idxmin()

index = seizure_df['seconds_since_hr_recording'].apply(find_nearest_heart_rate_time)
features.loc[index, 'seizure_event'] = 1

# set seizure labels (for windows that contains seizure event, it's corresponding index labels are set to 1)
seizure_indices = features.index[features['seizure_event'] == 1].tolist()
features['label']=0
for index in seizure_indices:
    features.loc[index:index+window_size-1, 'label'] = 1
print(features)
features.to_csv('features and seizure labels.csv')
# length =len(features)
# train_length = int(length * 0.8)
# test_length = int(length * 0.1)
# gap_length = int(length * 0.1)
# print(length,train_length,test_length,gap_length)
# if train_length + test_length + gap_length == length:
#     print('successful')
# else:
#     print('wrong segment')
X = features.drop(columns=['label','timestamp','seizure_event'])
y = features['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# X_train = X.iloc[:train_length]
# X_test = X.iloc[train_length + gap_length:]
# y_train = y.iloc[:train_length]
# y_test = y.iloc[train_length + gap_length:]

model = RandomForestClassifier(n_estimators=100, random_state=0,n_jobs=-1)
model.fit(X_train, y_train)
feature_names = X.columns

# start_time = time.time()
# importances_MDI = model.feature_importances_
# std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
# elapsed_time = time.time() - start_time
# print(elapsed_time)
# sorted_idx = importances_MDI.argsort()
#
# importance_df_MDI = pd.DataFrame({'Importance': importances_MDI, 'Std': std},
#                                  index=feature_names[sorted_idx]).sort_values('Importance',ascending=True)
# importance_df_MDI.to_csv(f'{heart_rate_name[:3]}MDI_importance with window_{window_size}.csv')
#
# # plot
# plt.figure(figsize=(10, 8))
# plt.barh(importance_df_MDI.index, importance_df_MDI['Importance'], xerr=importance_df_MDI['Std'], align='center', alpha=0.6, ecolor='black', capsize=10)
# plt.xlabel('Importance')
# plt.ylabel('Features')
# plt.title('Feature MDI Importance with Standard Deviation')
# plt.show()


# permutation importance
start_time = time.time()
result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=0,n_jobs=-1)
perm_importances = result.importances_mean
perm_std = result.importances_std
elapsed_time = time.time() - start_time
print(elapsed_time)
sorted_idx1 = perm_importances.argsort()

importance_df_perm=pd.DataFrame({'Importance': perm_importances, 'Std': perm_std},
                           index=feature_names[sorted_idx1]).sort_values('Importance',ascending=True)
importance_df_perm.to_csv(f'{heart_rate_name[:3]}perm_importance with window_{window_size}.csv')

# plot
plt.figure(2)
plt.barh(importance_df_perm.index, importance_df_perm['Importance'], xerr=importance_df_perm['Std'], align='center', alpha=0.6, ecolor='black', capsize=10)
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Permutation Importance with Standard Deviation')
plt.show()


