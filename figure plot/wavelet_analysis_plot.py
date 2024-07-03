import glob
import os
import pywt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
from datetime import timedelta, datetime
import plotly.express as px
import seaborn as sns
from numpy.fft import rfft, rfftfreq
import matplotlib.gridspec as gridspec

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
        '''
        print(f"Patient {patient_num} Heart Rate Data:")
        print(heart_rate_df.head())
        print(f"Patient {patient_num} Seizure Data:")
        print(seizure_df.head())
        '''

# Accessing data of patient 1
p1_heart_rate = patients_heart_rate[0]
p1_seizure = patients_seizure[0]

# assume start at a timestamp
reference_date = pd.Timestamp('2019-11-01')
p1_heart_rate['time'] = reference_date + pd.to_timedelta(p1_heart_rate['seconds_since_hr_recording'], unit='s')
p1_heart_rate['value'] = (p1_heart_rate['value'] - p1_heart_rate['value'].mean())/p1_heart_rate['value'].std() # z standardization
p1_heart_rate['value'] = p1_heart_rate['value'].interpolate(method='linear')

y = p1_heart_rate.resample('5min', label = 'right', on = 'time').mean().reset_index().value.to_numpy()
y = y[:24]

# continuous wavelet transform
widths = [43200/3600, 86400/3600, 1209600/3600, 5140800/3600, 10584000/3600] # scale
wavelet = "cmor1.5-1.0" #morlet wavelet
cwtmatr, freqs = pywt.cwt(y, widths, wavelet,sampling_period = 1/12)


global_wavelet_spectrum = np.mean(np.abs(cwtmatr)**2, axis=1)

# FFT
yf = rfft(y)
xf = rfftfreq(len(y), 300)

fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])

# plot figure
ax1 = fig.add_subplot(gs[0, 0])
pcm=ax1.pcolormesh(p1_heart_rate['time'][:24], widths, np.abs(cwtmatr))
ax1.set_xlabel('Time(s)')
ax1.set_ylabel('Frequency(HZ)')
ax1.set_title("Continuous Wavelet Transform (Scaleogram)")
ax1.set_yscale('log')
fig.colorbar(pcm, ax=ax1)

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(global_wavelet_spectrum,widths)
ax2.set_xlabel('Power')
ax2.set_title('global wavelet spectrum')
ax2.set_yscale('log')

ax3 = fig.add_subplot(gs[1, :])
ax3.plot(xf, np.abs(yf), 'g-')
ax3.set_xlabel('Frequency')
ax3.set_ylabel('Power')
ax3.set_title('Fourier Transform')
ax3.set_xscale('log')
plt.tight_layout()
plt.show()
'''
graph_size = (4,1.8)
hist_size = (1.6,1.6)
font_size_tick = 8
marker_sz = 3.5
hr_buffer = 0.1
ax_lw = 0.25
plot_lw = 0.5

for dataset, sfilter in zip([p1_heart_rate,slow, monthly], ['original','2-day', '7-day']):
    plt.style.use("seaborn-v0_8-white")
    fig = plt.figure(figsize=graph_size)
    plt.plot(dataset['time'], dataset['value'], 'black', linewidth=plot_lw,
             label='{} smoothed heart rate'.format(sfilter))
    plt.ylabel('Heart Rate (BPM)', fontsize=font_size_tick)
    plt.yticks(fontsize=font_size_tick)
    plt.xticks(rotation=25, fontsize=font_size_tick)

    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.setp(ax.spines.values(), linewidth=ax_lw)
    monthyearFmt = mdates.DateFormatter('%b')
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(monthyearFmt)

    start, end = ax.get_ylim()
    plt.show()
'''