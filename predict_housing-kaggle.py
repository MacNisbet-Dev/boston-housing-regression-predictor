# Code by Mackenzie Nisbet for kaggle competition
# Import kaggle, pandas for dataframe handling, libraries to handle unzipping the data, and scikitlearn for the machine learning model
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
test_data = pd.read_csv('test.csv')

# X = house attributes, y = target variable
train_X = train_data.drop(columns=['SalePrice', 'Id'])
train_y = train_data['SalePrice']
test_X = test_data.drop(columns=['Id'])

# Get the numeric values, replace missing values with median
numeric_features = train_X.select_dtypes(include=['int64', 'float64']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Get the categorical features, replace with mode then one-hot encode
categorical_features = train_X.select_dtypes(include=['object']).columns

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

# Fit the model on the training data
pipeline.fit(train_X, train_y)

# Predict on the test data
test_predictions = pipeline.predict(test_X)

# Creates prediction and results columns for kaggle submission
submission = pd.DataFrame({
    'Id': test_data['Id'],  
    'SalePrice': test_predictions 
})

# Save the submission dataframe to a CSV file
submission.to_csv('submission.csv', index=False)
