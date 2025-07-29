#!/usr/bin/env python
# coding: utf-8

# In[1]:


import netCDF4


# In[2]:


from netCDF4 import Dataset


# In[3]:


import pandas as pd
import pygrib
import re
import numpy as np
from datetime import timedelta
import os


# In[4]:


cols= ['PREC']


# In[5]:


#Create a folder

# Base path
base_path = r"D:\D\Ruvision\GFS\Realtime GFS"

# Folder name from the list

folder_name = cols[0]

# Full path to create
folder_path = os.path.join(base_path, folder_name)

# Create the folder
os.makedirs(folder_path, exist_ok=True)

print(f"Folder created: {folder_path}")


# In[6]:


forecast_hr = np.arange(15, 85,3)


# In[7]:


forecast_hr


# In[8]:


from datetime import timedelta


# In[9]:


def find_key(dictionary, element):
    for key, value in dictionary.items():
        if element in value:
            return key
    return 'None'


# In[10]:


latbounds = [22.5 - 0.25, 23.5]
lonbounds = [72 , 73 + 0.25]


# In[11]:


time_from_ref = np.arange(15,85,3)


# In[12]:


variable_name = cols[0]


# In[13]:


from datetime import datetime

# Get the current UTC date and time
current_utc_datetime = datetime.utcnow().date()
current_utc_datetime


# In[14]:


import os
import requests
from datetime import datetime, timedelta

base_url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"

# Create a directory to store the downloaded files
output_directory = f"D:\D\Ruvision\GFS\Realtime GFS\{variable_name}"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Define the date for the forecast (in UTC)
current_date = datetime.utcnow()
#current_date=current_date - timedelta(days=1)

# List of initialization times (00, 06, 12, and 18 UTC)
init_times = [6]

for time in init_times:
    # Loop through forecast hours from 3 to 72 in 3-hour intervals
    for forecast_hour in range(15, 85, 3):
        # Construct the URL for the current forecast hour and initialization time
        url = f"{base_url}?dir=%2Fgfs.{current_date.strftime('%Y%m%d')}%2F{time:02d}%2Fatmos&file=gfs.t{time:02d}z.pgrb2.0p25.f{forecast_hour:03d}&var_PRATE=on&lev_surface=on&subregion=&toplat=47&leftlon=55&rightlon=105&bottomlat=0"

        # Extract the filename from the URL
        filename = f"gfs.{current_date.strftime('%Y%m%d')}.t{time:02d}z.pgrb2.0p25.f{forecast_hour:03d}"
        filepath = os.path.join(output_directory, filename)

        # Send a GET request to the URL
        response = requests.get(url)

        # Check if the request was successful (HTTP status code 200)
        if response.status_code == 200:
            # Save the downloaded file
            with open(filepath, "wb") as file:
                file.write(response.content)
            print(f"Downloaded: {filename}")
        else:
            print(f"Failed to download: {url}, HTTP status code: {response.status_code}")


# In[15]:


columns_prec = []

for i in cols:
    for time_steps in forecast_hr:
        for j in np.arange(23.5,22.25,-0.25):
            for k in np.arange(72.0,73.25,0.25):
                columns_prec.append(f'{i}_{j}_{k}_{time_steps:03d}')


# In[16]:


import pandas as pd

# Get today's date at 06:00 UTC
ts_06utc = pd.Timestamp.utcnow().normalize() + pd.Timedelta(hours=6)
#ts_06utc = (pd.Timestamp.utcnow() - pd.Timedelta(days=1)).normalize() + pd.Timedelta(hours=6)

# Create DataFrame with just one timestamp (06:00 UTC)
data_prec = pd.DataFrame(index=[ts_06utc], columns=columns_prec)

# Remove timezone info (if any)
data_prec.index = data_prec.index.tz_localize(None)


# In[17]:


data_prec


# In[18]:


root_directory = f"D:\D\Ruvision\GFS\Realtime GFS\{variable_name}"
# Regular expression pattern to match the filenames in the format "gfs.t00z.pgrb2.0p25.f021"
filename_pattern = r'gfs\.\d{8}\.t\d{2}z\.pgrb2\.0p25\.f\d{3}$'

# Dictionary to store the filenames grouped by directory
directory_names = {}

# Function to check if the filename matches the expected pattern
def is_grib2_file(filename):
    return re.match(filename_pattern, filename)

# Walk through the root directory and its subdirectories
for dirpath, dirnames, filenames in os.walk(root_directory):
    x = str(dirpath).split(os.sep)[-1]
    directory_names[x] = []

    # Filter the filenames based on the expected pattern and store them in the dictionary
    for filename in filenames:
        if is_grib2_file(filename):
            directory_names[x].append(filename)

# Print the filenames of the GRIB2 files in each directory
for directory, filenames in directory_names.items():
    print(f"Directory: {directory}")
    for filename in filenames:
        print(filename)
    print()  # Add an empty line to separate directories


# In[19]:


# Open the GRIB file
filename = f"D:\D\Ruvision\GFS\Realtime GFS\{variable_name}\gfs.20250722.t06z.pgrb2.0p25.f015"
grbs = pygrib.open(filename)

# Print information about each GRIB message (parameter)
for grb in grbs:
    print(f"Parameter Name: {grb.name}")
    print(f"Level: {grb.level}")
    print(f"Units: {grb.units}")
    print(f"Values: {grb.values}")
    print(f"Grid Shape: {grb.values.shape}")
    print("----------")

# Close the GRIB file
grbs.close()


# In[20]:


counter=0
for time_step in data_prec.index:

    year = time_step.year
    month = time_step.month
    day = time_step.day

    ref_time = time_step.hour

    date_temp = pd.date_range(start = time_step + timedelta(hours = 15), end = time_step + timedelta(hours = 84) , freq = '3h')
    col_temp = np.arange(0,25)

    data_temp = pd.DataFrame(index = date_temp, columns=col_temp)

    for time_lag in time_from_ref:

        filename = f'gfs.{year}{month:02d}{day:02d}.t06z.pgrb2.0p25.f{time_lag:03d}'

        grib =f'{root_directory}/{filename}'
        grbs=pygrib.open(grib)
        variable = 'Precipitation rate'

        for grb in grbs.select(name=variable):
            temp=grb.values
            #reshaped_data= data.reshape(1, 9, 9)
            #data=reshaped_data
            lats, lons= grb.latlons()
            lats_reshaped = lats[:,0]  # Reshape latitudes to (189,)
            reversed_arr = lats_reshaped[::-1]
            lons_reshaped = lons[0,:]  # Reshape longitudes to (,201)
            lats=reversed_arr
            lons=lons_reshaped
            parameter_name = grb.name
            level_type = grb.typeOfLevel
            parameter_units = grb.parameterUnits
            level = grb.level
            forecast_time = grb.forecastTime
            valid_date = grb.validDate

        # latitude lower and upper index
        latli = np.argmin( np.abs( reversed_arr - latbounds[1] ) )
        latui = np.argmin( np.abs( reversed_arr - latbounds[0] ) ) 

        # longitude lower and upper index
        lonli = np.argmin( np.abs( lons_reshaped- lonbounds[0] ) )
        lonui = np.argmin( np.abs( lons_reshaped - lonbounds[1] ) )  

        time = pd.to_datetime(f'{year}-{month}-{day}') + timedelta(hours = int(int(ref_time) + int(time_lag)))
        data= temp[latli:latui, lonli:lonui][::-1]
        time_prev = time - timedelta(hours = 3)

        if ((int(time_lag)%6) == 0):
            if len(np.ravel(data)) == 25:
                data_temp.loc[time][0:25] =  (np.ravel(data)*21600) - np.ravel(data_temp.loc[time_prev][0:25])

        elif (((int(time_lag) % 6) != 0) & ((int(time_lag) % 3) == 0)):
            if len(np.ravel(data)) == 25:
                data_temp.loc[time][0:25] = (np.ravel(data)*10800)
        else:
            print(filename)

    data_prec.loc[time_step][0:600] = np.ravel(data_temp)

    counter += 1
    if (counter)%100 == 0:
        print(f'Loop {counter} Done!')


# In[21]:


data_prec.isnull().sum().max()


# In[22]:


data_prec


# In[23]:


data_final_interpolated1=data_prec


# In[24]:


data_final_interpolated1 = data_final_interpolated1.shift(freq=pd.Timedelta(hours=5, minutes=30))


# In[25]:


data_final_interpolated1 = data_final_interpolated1.rename_axis('DateTime')


# In[26]:


extracted_rows = data_final_interpolated1[data_final_interpolated1.index.time == pd.Timestamp("11:30").time()]


# In[27]:


data_prec_lead_day_1 = extracted_rows.loc[:, ~extracted_rows.columns.str.contains('|'.join(['039', '042', '045', '048', '051', '054', '057', '060', '063', '066', '069', '072', '075', '078', '081', '084']))]
data_prec_lead_day_2 = extracted_rows.loc[:, ~extracted_rows.columns.str.contains('|'.join(['015', '018', '021', '024', '027', '030', '033', '036', '063', '066', '069', '072', '075', '078', '081', '084']))]
data_prec_lead_day_3 = extracted_rows.loc[:, ~extracted_rows.columns.str.contains('|'.join(['015', '018', '021', '024', '027', '030', '033', '036', '039', '042', '045', '048', '051', '054', '057', '060']))]


# In[28]:


columns_prec = []
for i in cols:
    for j in np.arange(23.5,22.25,-0.25):
        for k in np.arange(72.0,73.25,0.25):
            columns_prec.append(f'{i}_{j}_{k}')


# In[29]:


start_date = extracted_rows.index[0]
end_date = extracted_rows.index[-1]


# In[30]:


# Update the start and end dates
updated_start_date_1 = pd.to_datetime(start_date) + pd.Timedelta(hours=15)
updated_end_date_1 = pd.to_datetime(end_date) + pd.Timedelta(hours=36)
# Update the start and end dates
updated_start_date_2 = pd.to_datetime(start_date) + pd.Timedelta(hours=39)
updated_end_date_2 = pd.to_datetime(end_date) + pd.Timedelta(hours=60)
# Update the start and end dates
updated_start_date_3 = pd.to_datetime(start_date) + pd.Timedelta(hours=63)
updated_end_date_3 = pd.to_datetime(end_date) + pd.Timedelta(hours=84)

data_prec_1 = pd.DataFrame(index = pd.date_range(start=updated_start_date_1, end=updated_end_date_1, freq = '3h'), columns = columns_prec)
data_prec_2 = pd.DataFrame(index = pd.date_range(start=updated_start_date_2, end=updated_end_date_2, freq = '3h'), columns = columns_prec)
data_prec_3 = pd.DataFrame(index = pd.date_range(start=updated_start_date_3, end=updated_end_date_3, freq = '3h'), columns = columns_prec)


# In[31]:


selected_rows_1 = data_prec_1.loc[data_prec_1.index.time == pd.Timestamp('02:30:00').time()]
selected_rows_2 = data_prec_1.loc[data_prec_1.index.time == pd.Timestamp('05:30:00').time()]
selected_rows_3 = data_prec_1.loc[data_prec_1.index.time == pd.Timestamp('08:30:00').time()]
selected_rows_4 = data_prec_1.loc[data_prec_1.index.time == pd.Timestamp('11:30:00').time()]
selected_rows_5 = data_prec_1.loc[data_prec_1.index.time == pd.Timestamp('14:30:00').time()]
selected_rows_6 = data_prec_1.loc[data_prec_1.index.time == pd.Timestamp('17:30:00').time()]
selected_rows_7 = data_prec_1.loc[data_prec_1.index.time == pd.Timestamp('20:30:00').time()]
selected_rows_8 = data_prec_1.loc[data_prec_1.index.time == pd.Timestamp('23:30:00').time()]

selected_rows_9 = data_prec_2.loc[data_prec_2.index.time == pd.Timestamp('02:30:00').time()]
selected_rows_10 = data_prec_2.loc[data_prec_2.index.time == pd.Timestamp('05:30:00').time()]
selected_rows_11 = data_prec_2.loc[data_prec_2.index.time == pd.Timestamp('08:30:00').time()]
selected_rows_12 = data_prec_2.loc[data_prec_2.index.time == pd.Timestamp('11:30:00').time()]
selected_rows_13 = data_prec_2.loc[data_prec_2.index.time == pd.Timestamp('14:30:00').time()]
selected_rows_14 = data_prec_2.loc[data_prec_2.index.time == pd.Timestamp('17:30:00').time()]
selected_rows_15 = data_prec_2.loc[data_prec_2.index.time == pd.Timestamp('20:30:00').time()]
selected_rows_16 = data_prec_2.loc[data_prec_2.index.time == pd.Timestamp('23:30:00').time()]

selected_rows_17 = data_prec_3.loc[data_prec_3.index.time == pd.Timestamp('02:30:00').time()]
selected_rows_18 = data_prec_3.loc[data_prec_3.index.time == pd.Timestamp('05:30:00').time()]
selected_rows_19 = data_prec_3.loc[data_prec_3.index.time == pd.Timestamp('08:30:00').time()]
selected_rows_20 = data_prec_3.loc[data_prec_3.index.time == pd.Timestamp('11:30:00').time()]
selected_rows_21 = data_prec_3.loc[data_prec_3.index.time == pd.Timestamp('14:30:00').time()]
selected_rows_22= data_prec_3.loc[data_prec_3.index.time == pd.Timestamp('17:30:00').time()]
selected_rows_23 = data_prec_3.loc[data_prec_3.index.time == pd.Timestamp('20:30:00').time()]
selected_rows_24 = data_prec_3.loc[data_prec_3.index.time == pd.Timestamp('23:30:00').time()]


# In[32]:


x1=data_prec_lead_day_1.iloc[:, :25]
x2=data_prec_lead_day_1.iloc[:, 25:50]
x3=data_prec_lead_day_1.iloc[:, 50:75]
x4=data_prec_lead_day_1.iloc[:, 75:100]
x5=data_prec_lead_day_1.iloc[:, 100:125]
x6=data_prec_lead_day_1.iloc[:, 125:150]
x7=data_prec_lead_day_1.iloc[:, 150:175]
x8=data_prec_lead_day_1.iloc[:, 175:]

x9=data_prec_lead_day_2.iloc[:, :25]
x10=data_prec_lead_day_2.iloc[:, 25:50]
x11=data_prec_lead_day_2.iloc[:, 50:75]
x12=data_prec_lead_day_2.iloc[:, 75:100]
x13=data_prec_lead_day_2.iloc[:, 100:125]
x14=data_prec_lead_day_2.iloc[:, 125:150]
x15=data_prec_lead_day_2.iloc[:, 150:175]
x16=data_prec_lead_day_2.iloc[:, 175:]

x17=data_prec_lead_day_3.iloc[:, :25]
x18=data_prec_lead_day_3.iloc[:, 25:50]
x19=data_prec_lead_day_3.iloc[:, 50:75]
x20=data_prec_lead_day_3.iloc[:, 75:100]
x21=data_prec_lead_day_3.iloc[:, 100:125]
x22=data_prec_lead_day_3.iloc[:, 125:150]
x23=data_prec_lead_day_3.iloc[:, 150:175]
x24=data_prec_lead_day_3.iloc[:, 175:]


# In[33]:


selected_rows_1.loc[:, :] = x1.values
selected_rows_2.loc[:, :] = x2.values
selected_rows_3.loc[:, :] = x3.values
selected_rows_4.loc[:, :] = x4.values
selected_rows_5.loc[:, :] = x5.values
selected_rows_6.loc[:, :] = x6.values
selected_rows_7.loc[:, :] = x7.values
selected_rows_8.loc[:, :] = x8.values
selected_rows_9.loc[:, :] = x9.values
selected_rows_10.loc[:, :] = x10.values
selected_rows_11.loc[:, :] = x11.values
selected_rows_12.loc[:, :] = x12.values
selected_rows_13.loc[:, :] = x13.values
selected_rows_14.loc[:, :] = x14.values
selected_rows_15.loc[:, :] = x15.values
selected_rows_16.loc[:, :] = x16.values
selected_rows_17.loc[:, :] = x17.values
selected_rows_18.loc[:, :] = x18.values
selected_rows_19.loc[:, :] = x19.values
selected_rows_20.loc[:, :] = x20.values
selected_rows_21.loc[:, :] = x21.values
selected_rows_22.loc[:, :] = x22.values
selected_rows_23.loc[:, :] = x23.values
selected_rows_24.loc[:, :] = x24.values


# In[34]:


merged_df_1 = pd.concat([selected_rows_1, selected_rows_2, selected_rows_3, selected_rows_4, 
                       selected_rows_5, selected_rows_6, selected_rows_7, selected_rows_8], axis=0)

merged_df_2 = pd.concat([selected_rows_9, selected_rows_10, selected_rows_11, selected_rows_12, 
                       selected_rows_13, selected_rows_14, selected_rows_15, selected_rows_16], axis=0)

merged_df_3 = pd.concat([selected_rows_17, selected_rows_18, selected_rows_19, selected_rows_20, 
                       selected_rows_21, selected_rows_22, selected_rows_23, selected_rows_24], axis=0)

merged_df_1 = merged_df_1.rename_axis('DateTime')
merged_df_1.reset_index('DateTime', inplace=True)
sorted_df_1 = merged_df_1.sort_values(by='DateTime', ascending=True)
sorted_df_1.set_index('DateTime', inplace=True)
data_X_Lead_Day_1=sorted_df_1


merged_df_2 = merged_df_2.rename_axis('DateTime')
merged_df_2.reset_index('DateTime', inplace=True)
sorted_df_2 = merged_df_2.sort_values(by='DateTime', ascending=True)
sorted_df_2.set_index('DateTime', inplace=True)
data_X_Lead_Day_2=sorted_df_2

merged_df_3 = merged_df_3.rename_axis('DateTime')
merged_df_3.reset_index('DateTime', inplace=True)
sorted_df_3 = merged_df_3.sort_values(by='DateTime', ascending=True)
sorted_df_3.set_index('DateTime', inplace=True)
data_X_Lead_Day_3=sorted_df_3

group_idx_1 = (data_X_Lead_Day_1.index.to_series().reset_index(drop=True).index // 8)
group_idx_2 = (data_X_Lead_Day_2.index.to_series().reset_index(drop=True).index // 8)
group_idx_3 = (data_X_Lead_Day_3.index.to_series().reset_index(drop=True).index // 8)

summed_data_1 = data_X_Lead_Day_1.groupby(group_idx_1).sum()
summed_data_2 = data_X_Lead_Day_2.groupby(group_idx_2).sum()
summed_data_3 = data_X_Lead_Day_3.groupby(group_idx_3).sum()

# Get today's date normalized to midnight
today = pd.Timestamp.today().normalize()
#today = pd.Timestamp.today().normalize()- pd.Timedelta(days=1)

# Generate dates for tomorrow, day after, and two days after — all at 23:30
date1 = today + pd.Timedelta(days=1, hours=23, minutes=30)
date2 = today + pd.Timedelta(days=2, hours=23, minutes=30)
date3 = today + pd.Timedelta(days=3, hours=23, minutes=30)

# Create DataFrames
data_prec_1 = pd.DataFrame(index=pd.date_range(start=date1, end=date1, freq='24h'), columns=summed_data_1.columns)
data_prec_2 = pd.DataFrame(index=pd.date_range(start=date2, end=date2, freq='24h'), columns=summed_data_2.columns)
data_prec_3 = pd.DataFrame(index=pd.date_range(start=date3, end=date3, freq='24h'), columns=summed_data_3.columns)

summed_data_1['DateTime']= data_prec_1.index
summed_data_2['DateTime']= data_prec_2.index
summed_data_3['DateTime']= data_prec_3.index

summed_data_1.set_index('DateTime', inplace=True)
summed_data_2.set_index('DateTime', inplace=True)
summed_data_3.set_index('DateTime', inplace=True)


# In[35]:


summed_data_2


# In[36]:


X1 = pd.read_excel(f"D:/D/Ruvision/GFS/Realtime GFS/{variable_name}_Ahmedabad_Lead Day 1_daily basis.xlsx")
X2 = pd.read_excel(f"D:/D/Ruvision/GFS/Realtime GFS/{variable_name}_Ahmedabad_Lead Day 2_daily basis.xlsx")
X3 = pd.read_excel(f"D:/D/Ruvision/GFS/Realtime GFS/{variable_name}_Ahmedabad_Lead Day 3_daily basis.xlsx")


# In[37]:


X1.set_index('DateTime', inplace=True)
X2.set_index('DateTime', inplace=True)
X3.set_index('DateTime', inplace=True)


# In[38]:


Data_X1= pd.concat([X1, summed_data_1], axis=0)
Data_X2= pd.concat([X2, summed_data_2], axis=0)
Data_X3= pd.concat([X3, summed_data_3], axis=0)


# In[39]:


Data_X2


# In[40]:


filename_1 = f"D:/D/Ruvision/GFS/Realtime GFS/{variable_name}_Ahmedabad_Lead Day 1_daily basis.xlsx"
filename_2 = f"D:/D/Ruvision/GFS/Realtime GFS/{variable_name}_Ahmedabad_Lead Day 2_daily basis.xlsx"
filename_3 = f"D:/D/Ruvision/GFS/Realtime GFS/{variable_name}_Ahmedabad_Lead Day 3_daily basis.xlsx"

# Save the DataFrame
Data_X1.to_excel(filename_1)
Data_X2.to_excel(filename_2)
Data_X3.to_excel(filename_3)


# In[ ]:





# In[ ]:





# In[ ]:




