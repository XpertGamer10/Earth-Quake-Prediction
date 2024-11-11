# Code By-Shivansh Vasu

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor  # Use RandomForestRegressor instead

# Load data
data = pd.read_csv("Data.csv")

# Convert data to numpy array if necessary
data = np.array(data)

# Separate features and target
X = data[:, 0:-1].astype(float)  # Ensure features are floats
y = data[:, -1].astype(float)  # Ensure target is float for regression

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialize and train the regressor
rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)

# Optional: evaluate model performance
y_pred = rfr.predict(X_test)
print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred))

# Evaluate model performance
from sklearn import metrics

# Predictions
y_pred = rfr.predict(X_test)

# Mean Squared Error
mse = metrics.mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)



# R² Score
r2_score = metrics.r2_score(y_test, y_pred)
print("R² Score:", r2_score)


# Save the trained model
pickle.dump(rfr, open('model.pkl', 'wb'))

# Code By-Shivansh Vasu
