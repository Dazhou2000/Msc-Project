import xlrd
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


heart_rate_data = pd.read_excel('C:/Users/ASUS/Desktop/P1_heart_rate_data.xlsx')
seizure_data = pd.read_excel('C:/Users/ASUS/Desktop/P1_seizure_diary.xlsx')

print("heart_rate_column title:", heart_rate_data.columns.values)
print("seizure_column title:", seizure_data.columns.values)


x = heart_rate_data.iloc[:,1]
# Time scale: original data scale-seconds
x_scale = x/3600/24
# Heart rate
y = heart_rate_data.iloc[:,2]

# Down-sample: original dataset samples at every 5 minutes, days = 24 hours = 288 * 5 minutes
n = 288
x_days = x_scale[::n]
y_days = y[::n]

# Seizure data
x1 = seizure_data.iloc[0:100,1] # down-sample limit, the last few data points may be discarded
x1_scale = x1/3600/24

# Linear interpolate, get the value of heart rate of seizures
interpolation_function = interp1d(x_days, y_days, kind='linear')
x_value_to_interpolate = x1_scale
y_interpolated = interpolation_function(x_value_to_interpolate)

# Plot the heart line chart with seizure time
plt.figure(1)
plt.plot(x_days, y_days, label = 'heart rate')
plt.scatter(x_value_to_interpolate,y_interpolated, c = 'red', label = 'seizures')
plt.legend()
plt.title('Heart rate and seizures')
plt.xlabel('time-days')  # Customize x-axis label
plt.ylabel('heart rate value(bmp')  # Customize y-axis label
plt.grid(False)  # Add grid
plt.show()

# Segment the data, find the range of heart before and after seizures

# Specify the range: e.g. an hour before seizure, half an hour after seizure
before_index = 3600/300
after_index = 1800/300

# point of interest, assume the ith seizure point
sum = [0]*19
for k in range(len(x1)):
 point_of_interest = x1[k]
 interest_index = round(point_of_interest/300)
 start_index = max(0, interest_index-before_index)
 end_index = min(len(x)-1, interest_index+after_index)
 whole_range = [None]*len(x1)

 # Segment the dataset based on the range
 before_range = heart_rate_data.iloc[int(start_index):interest_index+1]
 after_range = heart_rate_data.iloc[interest_index:int(end_index)+1]
 whole_range[k] = heart_rate_data.iloc[int(start_index): int(end_index)+1]
 sum = [a+b for a,b in zip(sum,whole_range[k].iloc[:,2])]
 Time_duration = x[0:19]/60



 interpolation_function1 = interp1d(whole_range[k].iloc[:,1], whole_range[k].iloc[:,2], kind='linear')
 x_value_to_interpolate1 = point_of_interest
 y_interpolated1 = interpolation_function1(x_value_to_interpolate1)

 # Plot multi segments and their average
 plt.figure(3)
 a,=plt.plot(Time_duration,whole_range[k].iloc[:,2],color='gray')
 #plt.plot(before_range.iloc[:,1], before_range.iloc[:,2], color = 'gray')
 #plt.plot(after_range.iloc[:, 1], after_range.iloc[:, 2], color = 'gray' )
 # plt.legend()
 plt.xlim(left=0)
 plt.title('Time segment around seizures')
 plt.xlabel('time-minutes')  # Customize x-axis label
 plt.ylabel('heart rate value(bmp)')  # Customize y-axis label

average=[value/len(x1) for value in sum]
b,=plt.plot(Time_duration,average,color='red',)
plt.legend([a,b],['seizure segments','average seizure segment'],loc='upper left')
plt.show()
'''
# Plot heart rate of one segment part
plt.figure(2)
plt.plot(before_range.iloc[:,1], before_range.iloc[:,2], color = 'blue', label = 'before seizure')
plt.plot(after_range.iloc[:, 1], after_range.iloc[:, 2], color = 'green', label = 'after seizure')
plt.scatter(before_range.iloc[int(before_index),1],before_range.iloc[int(before_index),2], color = 'black', label = 'rough seizure'),
plt.scatter(x_value_to_interpolate1, y_interpolated1, c ='red', label = 'exact seizure')
plt.legend()
plt.title('Segment:Heart rate and seizures')
plt.xlabel('time-seconds')  # Customize x-axis label
plt.ylabel('heart rate value(bmp')  # Customize y-axis label
plt.show()
'''