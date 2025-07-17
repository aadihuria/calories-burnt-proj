# -*- coding: utf-8 -*-
"""Calories Burnt Prediction.ipynb

Importing the Dependencies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

"""Data Collection & Processing"""

#loading data from csv file to pandas DataFrame
calories = pd.read_csv('calories.csv')

# print the first 5 rows of the dataframe
calories.head()

exercise_data = pd.read_csv('exercise.csv')
# higher intensity workout = higher heart rate

exercise_data.head()

"""Combining the two dataframes"""

calories_data = pd.concat([exercise_data, calories['Calories']], axis = 1)

calories_data.head()

# check the size of the dataset
calories_data.shape

# getting some info regarding the data
calories_data.info()

#checking for any missing values
calories_data.isnull().sum()

"""Dataset is complete and there are no missing values

### **Data Analysis**
"""

# get some statistical measures about the data
calories_data.describe()

"""Data Collection & Processing"""

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


"""1. Positive Correlation
2. Negative Correlation
"""

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

# Scatter plot of Duration vs Calories
plt.figure(figsize=(8, 5))
# Revert 'Gender' column for plotting
calories_data['Gender'] = calories_data['Gender'].replace({0: 'male', 1: 'female'})
sns.scatterplot(x='Duration', y='Calories', hue='Gender', data=calories_data)
plt.title('Duration vs Calories Burnt')
plt.xlabel('Duration (minutes)')
plt.ylabel('Calories')
plt.show()
# Convert 'Gender' back to numerical values for model training
calories_data.replace({"Gender": {'male': 0, 'female': 1}}, inplace = True)

"""Converting the text data to numeric values"""

calories_data.replace({"Gender": {'male': 0, 'female': 1}}, inplace = True)

"""Separating features and target"""

X = calories_data.drop(columns = ['User_ID', 'Calories'], axis = 1)
Y = calories_data['Calories']

print(X)

print(Y)

"""Splitting data into training data and test data"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 2)

print(X.shape, X_train.shape, X_test.shape)

"""Model Training

XGBoost Regressor
"""

# loading the model
model = XGBRegressor()

# training the model with X_train
model.fit(X_train, Y_train)

"""Evaluation

Prediction on Test Data
"""

# Evaluating model based off Test Data
test_data_prediction = model.predict(X_test)

print(test_data_prediction)

"""Mean Absolute Error"""

mae = metrics.mean_absolute_error(Y_test, test_data_prediction)

print("Mean Absolute Error = ", mae)