# -*- coding: utf-8 -*-
"""
Calories Burnt Prediction.ipynb

Objective:
Predict calories burnt during exercise using biometric and workout data.

This script includes:
1. Data loading and preprocessing
2. Exploratory data analysis (EDA)
3. Correlation analysis and visualization
4. Model training using XGBoost Regressor
5. Evaluation of model performance
"""

# === Importing Required Libraries ===

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

# === Data Collection and Processing ===

# Load calorie data from CSV into a DataFrame
calories = pd.read_csv('calories.csv')

# Load exercise session data
exercise_data = pd.read_csv('exercise.csv')


"""Combining the two dataframes"""

calories_data = pd.concat([exercise_data, calories['Calories']], axis = 1)

# check the size of the dataset
calories_data.shape

# getting some info regarding the data
calories_data.info()

#checking for any missing values
calories_data.isnull().sum()

# Confirmed: no missing data in the dataset

""" Exploratory Data Analysis (EDA) """

# get some statistical measures about the data
calories_data.describe()


# Set seaborn theme for cleaner plots
sns.set_theme(style="whitegrid")

# plotting the gender column in count plot
sns.countplot(calories_data['Gender'])

# finding the distribution of "Age" column
sns.histplot(data = calories_data, x = 'Age', kde = True)

# finding the distribution of "Height" column
sns.histplot(data = calories_data, x = 'Height', kde = True)

"""Finding the Correlation in the dataset"""

# finding the distribution of "Weight" column
sns.histplot(data = calories_data, x = 'Weight', kde = True)


""" Correlation Analysis """

# Compute correlation matrix (excluding Gender)
correlation = calories_data.drop('Gender', axis=1).corr()

# Distribution of Calories
plt.figure(figsize=(8, 5))
sns.histplot(calories_data['Calories'], bins=30, kde=True, color='blue')
plt.title('Distribution of Calories Burnt')
plt.xlabel('Calories')
plt.ylabel('Frequency')
plt.show()

# constructing a heatmap to understand the correlation

plt.figure(figsize=(10, 6))
numerical_cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']
sns.heatmap(calories_data[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

""" Scatter Plot: Duration vs Calories Burnt """

# Convert gender from numeric to string for visualization
calories_data['Gender'] = calories_data['Gender'].replace({0: 'male', 1: 'female'})

plt.figure(figsize=(8, 5))
sns.scatterplot(x='Duration', y='Calories', hue='Gender', data=calories_data)
plt.title('Duration vs Calories Burnt')
plt.xlabel('Duration (minutes)')
plt.ylabel('Calories')
plt.show()
# Convert 'Gender' back to numerical values for model training
calories_data.replace({"Gender": {'male': 0, 'female': 1}}, inplace = True)

# Convert gender back to numeric for model training
calories_data.replace({"Gender": {'male': 0, 'female': 1}}, inplace = True)

""" Feature Selection """

X = calories_data.drop(columns = ['User_ID', 'Calories'], axis = 1)
Y = calories_data['Calories']

# Preview features and target
print(X)
print(Y)

""" Train-Test Data Split """

# Split dataset into training and testing sets (80/20)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 2)

print(X.shape, X_train.shape, X_test.shape)

""" Model Training using XGBoost Regressor """

# Initialize XGBoost Regressor model
model = XGBRegressor()

# Train the model on the training data
model.fit(X_train, Y_train)

""" Model Evaluation """

# Evaluating model based off Test Data
test_data_prediction = model.predict(X_test)

print("Predicted Calories:", test_data_prediction)

# Calculate Mean Absolute Error (MAE)
mae = metrics.mean_absolute_error(Y_test, test_data_prediction)
print("Mean Absolute Error = ", mae)