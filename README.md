# Delhi Electricity Demand Prediction

This repository contains a machine learning model that predicts electricity demand for Delhi based on historical electricity usage and weather data. The model is designed to assist energy providers in forecasting demand, optimizing resource allocation, and improving grid management.

## Project Overview

The project focuses on predicting electricity consumption for various regions in Delhi, including areas served by Delhi Electric Supply Undertaking (DESU) and other distribution companies. The model uses weather data such as temperature, humidity, and other meteorological parameters, alongside historical electricity consumption data, to generate demand forecasts.

## Dataset

### 1. **Electricity Data:**
   - The dataset includes electricity usage records for different regions in Delhi.
   - Key columns: `TIMESLOT`, `DELHI`, `BRPL`, `BYPL`, `NDPL`, `NDMC`, `MES`.

### 2. **Weather Data:**
   - Meteorological parameters such as temperature, humidity, precipitation, and wind speed.
   - The data helps capture environmental conditions influencing electricity demand.

## Files

- `model.py`: Contains the implementation of the machine learning model used for predicting electricity demand.
- `preprocessing.py`: Responsible for cleaning, transforming, and preparing the raw data for training.
- `electricity_demand_model.h5`: The trained model saved in HDF5 format.
- `scaler.pkl`: The scaler used to normalize the data before feeding it to the model.
- `processed_electricity_data.csv`: The preprocessed electricity consumption data.
- `processed_weather_data.csv`: The preprocessed weather data.
- `weatherdata.csv`: Raw weather data for the year 2022.
- `delhi_dispatch/`: Directory containing additional dispatch data (if applicable).

## Requirements

- Python 3.x
- Keras
- TensorFlow
- Pandas
- NumPy
- Scikit-learn

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Abdul12221014/DelhiElectricityDemandPrediction.git
