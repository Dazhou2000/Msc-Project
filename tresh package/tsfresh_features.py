from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.feature_extraction import extract_features
from tsfresh.utilities.dataframe_functions import roll_time_series, make_forecasting_frame
import glob
import os
import pandas as pd

# Step 1: Specify the directory containing the CSV files
directory_path = r'C:\Users\ASUS\Desktop\MSc dataset'

# Step 2: Use glob to get a list of all CSV files in the directory
csv_files = glob.glob(os.path.join(directory_path, '*.csv'))

# Step 3: Sort the files to ensure correct order
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
print(f"choose: {heart_rate_name} and {seizure_name}")
print(heart_rate_df.head())
# Accessing data of one patient

p1_heart_rate = heart_rate_df
p1_seizure = seizure_df
reference_date = pd.Timestamp('2019-11-01')
p1_heart_rate['time'] = reference_date + pd.to_timedelta(p1_heart_rate['seconds_since_hr_recording'], unit='s')
p1_heart_rate['value'] = (p1_heart_rate['value'] - p1_heart_rate['value'].mean())/p1_heart_rate['value'].std() # z standardization
p1_heart_rate['value'] = p1_heart_rate['value'].interpolate(method='linear') # linear interpolation

df =pd.DataFrame(p1_heart_rate)
df['id']=1
df = df[:100]
print(df)
df_rolled = roll_time_series(df, column_id="id", column_sort="Unnamed: 0",max_timeshift=23,min_timeshift=23,n_jobs=1)
print(df_rolled)


fc = {
    'mean': None,
    'abs_energy': None,
    'autocorrelation': [{'lag':1}]
}
features = extract_features(df_rolled,default_fc_parameters=fc,column_id="id", column_sort="Unnamed: 0", column_value="value",n_jobs=1)
print(features)
print(features.shape)
