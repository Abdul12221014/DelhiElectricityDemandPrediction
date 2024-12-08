from sklearn.metrics import mean_squared_error, r2_score
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the datasets
electricity_df = pd.read_csv('/Users/abdulkadir/Desktop/DEDRS/processed_electricity_data.csv')
weather_df = pd.read_csv('/Users/abdulkadir/Desktop/DEDRS/processed_weather_data.csv')

# Data Preprocessing: Prepare your features and labels
# Assuming 'DELHI' column is the target variable for electricity usage
electricity_features = electricity_df.drop(['TIMESLOT', 'DELHI'], axis=1)
electricity_usage = electricity_df['DELHI'].values.reshape(-1, 1)

# Scaling the features and target
scaler = MinMaxScaler(feature_range=(0, 1))
electricity_scaled = scaler.fit_transform(electricity_features)
electricity_usage_scaled = scaler.fit_transform(electricity_usage)

# Prepare the data for the LSTM model
X = []
y = []
look_back = 5  # Number of previous time steps to use for predicting the next time step

for i in range(look_back, len(electricity_scaled)):
    X.append(electricity_scaled[i-look_back:i, :])
    y.append(electricity_usage_scaled[i, 0])

X = np.array(X)
y = np.array(y)

# Split the data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1)

# Evaluate the model and calculate R² score
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print("R² Score: ", r2)

# Save the trained model
model.save('/Users/abdulkadir/Desktop/DEDRS/electricity_demand_model.h5')
print("Model saved to /Users/abdulkadir/Desktop/DEDRS/electricity_demand_model.h5")

# Optional: Save the scaler for future predictions
import joblib
joblib.dump(scaler, '/Users/abdulkadir/Desktop/DEDRS/scaler.pkl')
print("Scaler saved to /Users/abdulkadir/Desktop/DEDRS/scaler.pkl")

# Optionally, plot the predictions vs actual values
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.show()
