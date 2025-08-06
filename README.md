# Ahmedabad_Realtime_Forecast_Model_RUVISION

👉 The.py modules primarily executes two operations:

a) Data download and Processing of Variables
b) Final Modeling to forecast for 3 lead days and generate plot

👉Variables of Interest: GFS Data U1000 (U Component of wind at 1000 mbar pressure level), V1000 (V Component of wind at 1000 mbar pressure level), and PREC (Precipitation Rate)
Observed Data: IMD gridded Daily Rain Data at 0.25 degree resolution👇
Pai et al. (2014). Pai D.S., Latha Sridhar, Rajeevan M., Sreejith O.P., Satbhai N.S. and Mukhopadhyay B., 2014: Development of a new high spatial resolution (0.25° X 0.25°)Long period (1901-2024) daily gridded rainfall data set over India and its comparison with existing data sets over the region; MAUSAM, 65, 1(January 2014), pp1-18.
Training Data: 2015-23
Testing: 2024-Present

👉 Initialization:

a) Initialisation hours: 6 UTC or IST 11:30 AM for the current day.
b) Forecast hours: 15 to 84 lead hours subsequently be converted into daily basis forecasts (15-36 hrs: 1 lead day, 39-60 hrs: 2nd lead day and 63-84 hrs: third lead day)
c) Initialize Latbounds, Lonbounds for Ahmedabad, 5X5 grid size
d) Two stage Censored Quantile Regression threshold values

👉 Forecast Goal: We are going to generate Ahmedabad Rain forecast for tomorrow, day after tomorrow and the subsequent date. 

👉 Steps to set up Ahmedabad Rain Forecast Model

a) Literature study for variable selection
b) Download GFS Data from historical archive of NCEP-NCAR (0.25 degree resolution) for different variables at various levels  India
c) Preprocessing of Variables with lat and lon bounds restricted within Ahmedabad City
d) Historical Data saved in Excel for various Lead Days
d) Model set up- Two stage Censored Quantile Regression (CQR) based on GLM, RF, XGB and LGR
d) More than 100 experiments conducted with different variable combination
e) Computed Scores like Recall, Precision, Accuracy, F1 value, HSS, CQVSS
f) Based on scores, selected best variable combination
g) Next build up code for real-time operations and webhosting.

👉Realtime Operations

a) Create basepath
b) Create Folder with Variable name in basepath
c) Provide URL Structure
d) Download data for each variable and store into folders with variable names created in basepath. Pygrib Data Download from NOMADS/NCEP-NCAR for 06 hrs UTC
e) There are two sections of Preprocessing: preprocess_wind function is for preprocessing data for variables like windspeeds (V1000 and U1000); and preprocess_precipitation function is for preprocessing Precipitation data. The method for preprocessing wind variables are same but for precipitation is different. Precipitation is achieved by converting Precipitation Rate into Acc Prec. If the acc is of 6 hours, then precipitation rate*3600*6 (Convert seconds into hr) and if the acc is of 3 hours, then precipitation rate*3600*3
f) Split and reshape Function is to classify data into 3 lead days
g) Aggregate and Save function is to convert hourly data into daily basis. If Variable== PREC, then we sum hourly data and if wind, then Average. Finally we are concatenating the current data to the past data saved already in excel for each of lead days.  

h) #Modeling part
The code picks up excel files saved for all lead days, and for each lead day, U1000, V1000 abd Prec variables are combined, normalized and PCA done to reduce number of features. The PCA data stored into pickle files. GLM based Two Step Censored Quantile Regression performed in order to forecast for next 3 days from the date of initialization. 

👉Automation and Refactoring of Code
👉GitHub Code Review
👉Taken to HPC and the code will be scheduled to operate everyday at 5 PM and forecasts will be generated for next 3 days
👉 Web hosting and statistical forecasts shown on clipre.ai



