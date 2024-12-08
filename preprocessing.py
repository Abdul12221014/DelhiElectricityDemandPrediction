import os
import pandas as pd
import csv
from sklearn.preprocessing import MinMaxScaler

# Paths
electricity_folder_path = '/Users/abdulkadir/Desktop/DEDRS/delhi_dispatch'
weather_file_path = '/Users/abdulkadir/Desktop/DEDRS/weatherdata.csv'

# Preprocess Electricity Data
def preprocess_electricity_data(folder_path):
    electricity_data = []

    # Load electricity data from all files in the folder
    for month_folder in os.listdir(folder_path):
        month_path = os.path.join(folder_path, month_folder)
        if os.path.isdir(month_path):  # Only consider directories
            for file in os.listdir(month_path):
                if fi