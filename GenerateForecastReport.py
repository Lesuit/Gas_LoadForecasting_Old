#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 16:08:32 2021

@author: kuipan
"""

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import sklearn
from sklearn import metrics
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from google.cloud import storage
import get_weather
from get_weather import get_hist_avg_weather
from get_weather import get_past_actual_weather
from get_weather import get_forecast_weather

# Upload files to storage bucket
def upload_cloud_storage(filepath):
    destination_blob_name = filepath.split('/')[-1]
    storage_client = storage.Client()
    bucket = storage_client.get_bucket('short_term_gas_forecast')
    blog = bucket.blob(destination_blob_name)
    blog.upload_from_filename(filepath)

# Download files from storage bucket
def download_cloud_storage(filepath):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket('short_term_gas_forecast')
    blob1 = bucket.blob(filepath)
    blob1.download_to_filename('/tmp/' + filepath)
    

def filter_df_by_date(df, dates, col_name):
    new_df = df[df[col_name].isin(dates)]
    return new_df

def LoadTrainedModels_helper(zone, profile, Models_Dict):
    for i in range(3):
        saved_model_name = 'trained_models/' + str(zone) + profile + '_' + str(i+1) + '.h5'
        model = keras.models.load_model(saved_model_name)
        Models_Dict[saved_model_name] = model

def LoadTrainedModels(Models_Dict=dict()):
    zone = 101
    for profile in ['APT', 'COMM', 'INDU', 'LCOM', 'MAPT', 'MCOM', 'RES']:
        LoadTrainedModels_helper(zone, profile, Models_Dict)

    zone = 2601
    for profile in ['COM_LO', 'RES', 'RUR']:
        LoadTrainedModels_helper(zone, profile, Models_Dict)

    zone = 103
    for profile in ['APT', 'COMM', 'LCOM', 'MAPT', 'MCOM', 'MIND','MRES','RES']:
        LoadTrainedModels_helper(zone, profile, Models_Dict)
    return Models_Dict


def ProcessWeatherForMLInputs(past_and_forecast_weather_df):
    final_df = past_and_forecast_weather_df

    new_column_names = ['date', 
                        'Cgy_max_temp', 'Cgy_min_temp', 'Cgy_avg_temperature', 
                        'Cgy_hourly_avg_temperature', 'Cgy_avg_wind_speed', 
                        'Edm_max_temp', 'Edm_min_temp', 'Edm_avg_temperature',
                        'Edm_hourly_avg_temperature','Edm_avg_wind_speed']

    final_df.columns = new_column_names

    final_df = final_df.assign(**{'Edm HDD New': 
                                  [max(0, 18-x) for x in final_df['Edm_avg_temperature'].values],
                                  'Cgy HDD New':
                                  [max(0, 18-x) for x in final_df['Cgy_avg_temperature'].values]})

    final_df = final_df.assign(**{'Dayofweek': [x.dayofweek for x in final_df['date']], 
                                  'Monthofyear': [x.month for x in final_df['date']]})

    final_df = final_df.assign(**{'2D_Dayofweek': 
                                  [np.sin(2*np.pi*(x)/7) for x in final_df['Dayofweek']],
                                  '2D_Monthofyear': 
                                  [np.sin(2*np.pi*x/12) for x in final_df['Monthofyear']],
                                  'Weekend': 
                                  ((final_df['Dayofweek'] == 5) | (final_df['Dayofweek'] == 6)).astype(float),
                                  'Edm_temp*wind':
                                  final_df['Edm_avg_temperature']*final_df['Edm_avg_wind_speed'], 
                                  'Cgy_temp*wind':
                                  final_df['Cgy_avg_temperature']*(final_df['Cgy_avg_wind_speed'])})

    final_df = final_df.assign(**{
        'Edm_temp_squared': final_df['Edm_avg_temperature']**2,
        'Edm_temp_cubic': final_df['Edm_avg_temperature']**3,
        'Cgy_temp_squared': final_df['Cgy_avg_temperature']**2,
        'Cgy_temp_cubic': final_df['Cgy_avg_temperature']**3
    })

    final_df = final_df.assign(**{
        'Edm_wind_squared': final_df['Edm_avg_wind_speed']**2,
        'Edm_wind_cubic': final_df['Edm_avg_wind_speed']**3,
        'Cgy_wind_squared': final_df['Cgy_avg_wind_speed']**2,
        'Cgy_wind_cubic': final_df['Cgy_avg_wind_speed']**3
    })

    final_df = final_df.assign(**{
        'Edm_temp_diff': final_df['Edm_avg_temperature'].diff(),
        'Cgy_temp_diff': final_df['Cgy_avg_temperature'].diff()
    })

    final_df.dropna(inplace=True)
    return final_df

# Add pandas columns values as a new column.
def add_column_values(input_df):
    final_sum = 0
    for col_name in input_df.columns:
        final_sum += input_df[col_name]
    return list(final_sum)

# Calculate site counts weighted average temperature
def CalculateSiteCountsWeightedAvgTemp(cgy_cnt, edm_cnt, cgy_temp, edm_temp):
    weighted_avg = []
    for a, b, c, d in zip(cgy_cnt, edm_cnt, cgy_temp, edm_temp):
        weighted_avg.append((a*c+b*d)/(a+b))
    return weighted_avg


def model_prediction(Models_Dict, city, zone, profile, df, output_df, is_PreUsages=False):
    if is_PreUsages:
        df = df.assign(**{
            'previousday': df['average_usage'].shift(7)
        })
    
    temp = ['_avg_temperature', '_min_temp', '_max_temp', '_hourly_avg_temperature',
            '_temp_squared', '_temp_cubic', '_temp_diff',
            '_avg_wind_speed', '_wind_squared', '_wind_cubic', '_temp*wind', ' HDD New']
    temp = [city + tem for tem in temp]
    
    if is_PreUsages:
        columns = temp + ['Dayofweek','Monthofyear', '2D_Dayofweek', '2D_Monthofyear', 'Weekend', 
                          'previousday'] #'previous2day', 'previous3day', 'previous7day'
    else:
        columns = temp + ['Dayofweek','Monthofyear', '2D_Dayofweek', '2D_Monthofyear', 'Weekend']


    x_raw = df[columns].values
    
    saved_scaler_name = 'trained_models/' + str(zone) + profile + 'scaler.pkl'
    with open(saved_scaler_name, 'rb') as input:
        scaler = pickle.load(input)
    x = scaler.transform(x_raw)

    saved_ymax_name = 'trained_models/' + str(zone) + profile + 'ymax.pkl'
    with open(saved_ymax_name, 'rb') as input:
        y_max = pickle.load(input)
      
    y_preds = []
    for i in range(3):
        saved_model_name = 'trained_models/' + str(zone) + profile + '_' + str(i+1) + '.h5'
#         model = keras.models.load_model(saved_model_name)
        model = Models_Dict[saved_model_name]
        y_pred = list(model.predict(x)*y_max)
        y_preds.append(y_pred)
    mean_ypred = [np.mean([y_preds[0][i], y_preds[1][i], y_preds[2][i]]) for i in range(len(y_preds[0]))]
    output_df[str(zone) + profile] = mean_ypred


def GetLoadForecastingResult(Models_Dict, filename):  # 'uploads/NG ST Forecast Report.xlsm'
    past_df = get_past_actual_weather(22, -1)  # use 22 instead of 21 is because we need to calculate temp diff
    forecast_df = get_forecast_weather()
    past_df_new = past_df[forecast_df.columns]  # This is to make sure that the column names match!!
    past_and_forecast_weather_df = pd.concat([past_df_new, forecast_df], axis=0)
    final_df = ProcessWeatherForMLInputs(past_and_forecast_weather_df)
    
    output_df = pd.DataFrame()
    output_df['date'] = final_df['date']

    city = 'Edm'
    zone = 101
    for profile in ['APT', 'COMM', 'INDU', 'LCOM', 'MAPT', 'MCOM', 'RES']:
        model_prediction(Models_Dict, city, zone, profile, final_df, output_df, is_PreUsages=False)

    city = 'Edm'
    zone = 2601
    for profile in ['COM_LO', 'RES', 'RUR']:
        model_prediction(Models_Dict, city, zone, profile, final_df, output_df, is_PreUsages=False)

    city = 'Cgy'
    zone = 103
    for profile in ['APT', 'COMM', 'LCOM', 'MAPT', 'MCOM', 'MIND','MRES','RES']:
        model_prediction(Models_Dict, city, zone, profile, final_df, output_df, is_PreUsages=False)
    
    download_cloud_storage(filename)
    print ('successfully downloaed site counts file from blogstorage!!!', '\n')
    site_counts = pd.read_excel('/tmp/' + filename)
    site_counts['Date'] = [pd.Timestamp(year=x.year, month=x.month, day=x.day) for x in site_counts['Date']]
    
    
    common_dates = []
    for d in site_counts['Date']:
        if d in output_df['date'].values:
            common_dates.append(d)
    print (common_dates)
    
    site_counts = filter_df_by_date(site_counts, common_dates, 'Date')
    output_df = filter_df_by_date(output_df, common_dates, 'date')
    final_df = filter_df_by_date(final_df, common_dates, 'date')
    ## Generate total consumptions for each zone profile
    consumptions = []
    col_names = []
    for x in site_counts.columns[1:]:
        col_name = x.split('.')[0]
        try:
            counts = site_counts[x].values
            avg_consum = output_df[col_name].values
            total_consum = np.multiply(counts, avg_consum)
            consumptions.append(total_consum)
            col_names.append(x)
        except:
            continue   
    df = pd.DataFrame(np.array(consumptions).transpose())
    df.columns = col_names

    ## Find the zone profile col names
    fixed_cols = []
    float_cols = []
    fixed_cgy = []
    float_cgy = []
    fixed_edm = []
    float_edm = []
    for col in df.columns:
        if col.endswith('1'):
            float_cols.append(col)
            if col.startswith('103'):
                float_cgy.append(col)
            else:
                float_edm.append(col)
        else:
            fixed_cols.append(col)
            if col.startswith('103'):
                fixed_cgy.append(col)
            else:
                fixed_edm.append(col)        

    # Generate final output columns
    total_forecast_GJ = add_column_values(df)
    fixed_forecast_GJ = add_column_values(df[fixed_cols])
    float_forecast_GJ = add_column_values(df[float_cols])

    total_site_count_Cgy = add_column_values(site_counts[fixed_cgy + float_cgy])
    fixed_site_count_Cgy = add_column_values(site_counts[fixed_cgy])
    float_site_count_Cgy = add_column_values(site_counts[float_cgy])

    total_site_count_Edm = add_column_values(site_counts[fixed_edm + float_edm])
    fixed_site_count_Edm = add_column_values(site_counts[fixed_edm])
    float_site_count_Edm = add_column_values(site_counts[float_edm])

    #Determine how many days for backcast and forecast
    time_now = datetime.now()
    UTC_timestamp = time_now.timestamp()
    cur_timestamp = pd.Timestamp(UTC_timestamp, unit='s', tz='US/Mountain')
    cur_timestamp = pd.Timestamp(year=cur_timestamp.year, month=cur_timestamp.month, day=cur_timestamp.day)
    
    NumOfBackcastDays = len([x for x in common_dates if x<cur_timestamp])
    NumOfForecastDays = len([x for x in common_dates if x>=cur_timestamp])
    Final_Output_File = pd.DataFrame()
    Final_Output_File['date'] = output_df['date']

    Final_Output_File['Total Actual GJ'] = ['NA']*(NumOfBackcastDays + NumOfForecastDays)
    Final_Output_File['Total Forecast GJ'] = ['NA']*NumOfBackcastDays + total_forecast_GJ[NumOfBackcastDays:]
    Final_Output_File['Fixed Forecast GJ'] = ['NA']*NumOfBackcastDays + fixed_forecast_GJ[NumOfBackcastDays:]
    Final_Output_File['Floating Forecast GJ'] = ['NA']*NumOfBackcastDays + float_forecast_GJ[NumOfBackcastDays:]

    Final_Output_File['Total Backcast GJ'] = total_forecast_GJ[:NumOfBackcastDays] + ['NA']*NumOfForecastDays
    Final_Output_File['Fixed Backcast GJ'] = fixed_forecast_GJ[:NumOfBackcastDays] + ['NA']*NumOfForecastDays
    Final_Output_File['Floating Backcast GJ'] = float_forecast_GJ[:NumOfBackcastDays] + ['NA']*NumOfForecastDays

    Final_Output_File['Total Site Count CGY'] = total_site_count_Cgy
    Final_Output_File['Fixed Site Count CGY'] = fixed_site_count_Cgy
    Final_Output_File['Floating Site Count CGY'] = float_site_count_Cgy

    Final_Output_File['Total Site Count EDM'] = total_site_count_Edm
    Final_Output_File['Fixed Site Count EDM'] = fixed_site_count_Edm
    Final_Output_File['Floating Site Count EDM'] = float_site_count_Edm

    # Handling weather output columns
    temp_df = final_df.iloc[:,:11]
    temp_df['Site Count Weighted Average Temp (Celsius)'] = CalculateSiteCountsWeightedAvgTemp(total_site_count_Cgy, 
                                                                                               total_site_count_Edm, 
                                                                                               temp_df['Cgy_avg_temperature'].tolist(),
                                                                                               temp_df['Edm_avg_temperature'].tolist())

    temp_df.columns = ['date', 'CGY MAX Temp (Celsius)', 'CGY MIN Temp (Celsius)', 'CGY AVG Temp (Celsius)',
                       'CGY Hourly AVG Temp (Celsius)', 'CGY AVG Windspeed (mph)',
                       'EDM MAX Temp (Celsius)', 'EDM MIN Temp (Celsius)', 'EDM AVG Temp (Celsius)',
                       'EDM Hourly AVG Temp (Celsius)', 'EDM AVG Windspeed (mph)',
                       'Site Count Weighted Average Temp (Celsius)']

    # Get historial average weather on the same day as today.
    HIST_AVG = get_hist_avg_weather()
    Final_Output_File = pd.merge(Final_Output_File, temp_df, on='date')
    Final_Output_File = pd.merge(Final_Output_File, HIST_AVG, on='date')

    cur_time = cur_timestamp.strftime('%m-%d-%Y')
    report_filename = cur_time + '_NG_ST_Forecast&Backcast.xlsx'
    ReportFilePath = '/tmp/' + report_filename
    Final_Output_File.to_excel(ReportFilePath)
    
    upload_cloud_storage(ReportFilePath)
    
    return report_filename