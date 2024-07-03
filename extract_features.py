import glob
import os
import pandas as pd
import tsfel
import settings
import get_heart_rate_cycles
import custom_features


window_size = 288*2
def main():
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
    print(f"choose: {heart_rate_name} and {seizure_name}")
    print(heart_rate_df.head())
    # Accessing data of one patient

    p1_heart_rate = heart_rate_df
    p1_seizure = seizure_df

    # assume start at a timestamp
    reference_date = pd.Timestamp('2019-11-01')
    p1_heart_rate['time'] = reference_date + pd.to_timedelta(p1_heart_rate['seconds_since_hr_recording'], unit='s')
    p1_heart_rate['value'] = (p1_heart_rate['value'] - p1_heart_rate['value'].mean())/p1_heart_rate['value'].std() # z standardization
    p1_heart_rate['value'] = p1_heart_rate['value'].interpolate(method='linear') # linear interpolation
    #p1_heart_rate['value'][pd.isnull(p1_heart_rate['value'])] = p1_heart_rate['value'].mean() # mean interpolation

    y = p1_heart_rate.resample('5min', label = 'right', on = 'time').mean().reset_index().value.to_numpy()
    sampling_freq = 1/300

    # get multi-day cycles: per scales hours, i.e. scales = 6,12,24
    scales=get_heart_rate_cycles.heart_rate_cycles(y,sampling_freq) # get multi-day cycles: per scales hours

    signal_window = tsfel.utils.signal_processing.signal_window_splitter(y, window_size, overlap=1-1/window_size)

    cfg_file = settings.get_features_by_domain(domain=None,json_path="features.json") # tsfel feature path
    tsfel_features = tsfel.time_series_features_extractor(cfg_file, signal_window,sampling_freq,n_jobs=-1) # extract features from tsfel package

    custom_feature=[]
    for i in range(len(signal_window)):
     abs_mean= custom_features.wavelet_power(signal_window[i],scales)
     custom_feature.append(abs_mean)

    df_custom= pd.DataFrame(custom_feature, columns=[f'Scale_{scale/24}' for scale in scales])
    df_tsfel = pd.DataFrame(tsfel_features)
    df_combine = pd.concat([df_tsfel,df_custom],axis=1) # combine custom and tsfel features into one dataframe
    print(df_tsfel)
    print(df_custom)
    print(df_combine)
    csv_file_name = f'extracted_features_{heart_rate_name[:3]} with window_{window_size}.csv'
    df_combine.to_csv(csv_file_name)

if __name__ == "__main__":
    main()
