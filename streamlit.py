import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
data = pd.read_csv('cityu10c_train_dataset.csv')

# Drop the second column
data.drop(data.columns[1], axis=1, inplace=True)

# Step 1: Remove rows with more than 20% missing data
threshold = len(data.columns) * 0.2
data = data.dropna(thresh=threshold)

# Step 2: Replace numeric Null values with the median of their respective columns
numeric_columns = data.select_dtypes(include=[np.number]).columns
for column in numeric_columns:
    median_value = data[column].median()
    data[column].fillna(median_value, inplace=True)

# Step 3: Replace categorical Null values with 'Unknown'
categorical_columns = data.select_dtypes(include=[object]).columns
for column in categorical_columns:
    data[column].fillna('Unknown', inplace=True)

# Step 4: Encode categorical data into binary (0 and 1)
data = pd.get_dummies(data, drop_first=True)

# Assuming the last column is the target variable and the rest are features
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save the model and scaler
joblib.dump(clf, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Load the model and scaler
clf = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit app
st.title('Classification Model Prediction')

# Create input fields for each feature
input_data = {}
for feature in X.columns:
    input_data[feature] = st.number_input(f'Enter {feature}', value=0.0)

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

# Standardize the input data
input_scaled = scaler.transform(input_df)

# Make prediction
if st.button('Predict'):
    prediction = clf.predict(input_scaled)
    st.write(f'Prediction: {prediction[0]}')
