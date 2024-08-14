import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Generate dummy time series data
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data)-time_step-1):
        a = data[i:(i+time_step), 0]
        X.append(a)
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Prepare the dataset
np.random.seed(7)
data = np.sin(np.linspace(0, 100, 1000))  # Sine wave
data = data.reshape(-1, 1)

time_step = 10
X, y = create_dataset(data, time_step)

# Reshape for LSTM [samples, time steps, features]
X = X.reshape(X.shape[0], X.shape[1], 1)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=10, batch_size=1, verbose=1)

# Make predictions
predictions = model.predict(X)

# Plot the results
plt.plot(y, label='True Data')
plt.plot(predictions, label='Predicted Data')
plt.legend()
plt.show()
