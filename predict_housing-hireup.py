# Code by Mackenzie Nisbet for hireup submission
# Import kaggle, pandas for dataframe handling, numpy for math, libraries to handle unzipping the data, and scikitlearn for the machine learning model
import kaggle
import zipfile
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Authenticate to use kaggle's API (kaggle's API key is already placed in %HOMEPATH%\.kaggle and competition rules are accepted)
kaggle.api.authenticate()

# Download the competition zip using kaggle's API
kaggle.api.competition_download_files('house-prices-advanced-regression-techniques', path='.')

# Find and extract the zip file to the current directory
zip_file = 'house-prices-advanced-regression-techniques.zip'

if os.path.exists(zip_file):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall('.') 
    print("Files extracted successfully.")
else:
    print("Zip file not found.")

# Load the datasets
train_data = pd.read_csv('train.csv')

# X = house attributes, y = target variable
X = train_data.drop(columns=['SalePrice', 'Id'])
y = train_data['SalePrice']
average_price = y.mean()

# Get the numeric values, replace missing values with median
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Get the categorical features, replace missing values with median then one-hot encode
categorical_features = X.select_dtypes(include=['object']).columns

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers into a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create a pipeline that includes the preprocessor and the model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# Split into test and training data (not using the provided train.csv since there's no price to check against)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model on the training data
pipeline.fit(X_train, y_train)

# Predict on the test data
y_pred = pipeline.predict(X_test)

# Evaluate the model based on RSME 
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Print the RMSE value, average house price, and error ratio
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
print(f'Average price: {average_price:.2f}')
error_ratio = rmse / average_price
print(f'Error ratio: {error_ratio:.2f}')

# Average house price: 180921.20
# Result RMSE for this simple model: 29445.77
# Error ratio: .16 (16%)
# This is a rather poor error ratio for a machine learning model, but a good start before any other adjustments or tuning. 