# Ahmedabad_Realtime_Forecast_Model_RUVISION

There are two sets of refactored codes:
a) Real time Data download and Processing of Variables-V Component of wind at 1000 mbar pressure level (V1000), U Component of wind at 1000 mbar pressure level (U1000) and Precipitation Rate (PREC)
b)Final Modeling to forecast for 3 lead days and generate plot

For first set of code, refer Refactored Code_Data Processing_U1000_V1000_PREC.py file

Initialisation hours is 6 UTC or IST 11:30 AM for the current day, say today at IST 11:30 hrs. We are going to generate forecast for tomorrow, day after tomorrow and the subsequent date. 
Variables: U1000, V1000, PREC
Forecast hours: 15 to 84 lead hours subsequently be converted into daily basis forecasts (15-36 hrs: 1 lead day, 39-60 hrs: 2nd lead day and 63-84 hrs: third lead day)
Initialize Latbounds, Lonbounds for Ahmedabad, 5X5 grid size
Create Folder with Variable name
Provide URL Structure
Download data for each variable and store into folders with variable names created in basepath
There are two sections of Preprocessing: preprocess_wind function is for preprocessing data for variables like V1000 and U1000; and preprocess_precipitation function is for preprocessing Precipitation data. The method for preprocessing wind variables are same but for precipitation is different. Precipitation is achieved by converting Precipitation Rate into Acc Prec. If the acc is of 6 hours, then precipitation rate*3600*6 (Convert seconds into hr) and if the acc is of 3 hours, then precipitation rate*3600*3
Split and reshape Function is to classify data into 3 lead days
Aggregate and Save function is to convert hourly data into daily basis. If Variable== PREC, then we sum hourly data and if wind, then Average. Finally we are storing data for 3 lead days into seperate files for each variable. 
