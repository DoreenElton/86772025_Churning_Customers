# -*- coding: utf-8 -*-
"""86772025_Churning_Customers.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ss69XqZBMHw-Tdi7HJ6aNL399G8lPALf
"""

from google.colab import drive
drive.mount('/content/drive')

#import Statements
import pandas as pd
import numpy as np
import tensorflow as tf
#Encoder
from sklearn.preprocessing import LabelEncoder

# Imputer
from sklearn.impute import KNNImputer

#Visualization
import seaborn as sns
import matplotlib.pyplot as plt


#Models
from tensorflow.keras.models import save_model
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.metrics import accuracy_score, roc_auc_score

#Loading the dataset
churn_df = pd.read_csv("/content/drive/My Drive/Colab Notebooks/CustomerChurn_dataset.csv")

#Showing the Dataset
churn_df
#Creating a dataset as a copy of the original dataset
new_churn_df = churn_df.copy()

#Displaying the information of the dataset
churn_df.info()

new_churn_df

"""##Question 1
###Using the given dataset to extract the relevant features that can define a customer churn.
"""

#Droping irrelevant column
churn_df.drop('customerID', axis=1, inplace=True)
churn_df.info()

churn_df['TotalCharges'] = pd.to_numeric(churn_df['TotalCharges'], errors='coerce', downcast = 'float')
churn_df.info()
# What we expect to see is the column TotalCharges to change to type float64

#Checking to see if there are any missing values
churn_df.isnull().sum()

#Using KNNImputer to attempt to impute the missing values
imputer = KNNImputer(n_neighbors = 6)

churn_df[['TotalCharges']] = imputer.fit_transform(churn_df[['TotalCharges']])
churn_df.isnull().sum()

#Using label encoding to encode all columns in the churn_df dataframe as integers to allow for easier processing by algorithms
new_churn_df = churn_df.copy()
label_encoder = LabelEncoder()

for i in churn_df.select_dtypes(['object']):
    churn_df[i] = label_encoder.fit_transform(churn_df[i])

churn_df.info()

# What we expect to see are most of the columns data type switching from 'Object' to 'int64'

churn_df

#Calculating the correlation matrix for the features in the dataframe and extracting the relevant features based on
#their correlation strength to the target variable 'Churn'
correlation_matrix = churn_df.corr()
target_feature = correlation_matrix['Churn'].abs().sort_values(ascending=False)
relevant_features = target_feature[~target_feature.index.str.contains('Churn')]
relevant_features

#selecting a specified number of top relevant features based on their absolute correlation with the target variable 'Churn'
#and excluding 'Churn' itself from the top features
num_features_to_select = 9
top_relevant_features = target_feature.abs().sort_values(ascending=False).index[:num_features_to_select].tolist()
top_relevant_features.remove('Churn')
top_relevant_features

"""#Question 2
###Using Exploratory Data Analysis (EDA) skills to find out which customer profiles relate to churning a lot.


"""

#Creating a new column 'Customer profile' by combing other existing colums
new_churn_df['Customer_profile'] = new_churn_df['gender'].astype(str) + '_' + \
new_churn_df['SeniorCitizen'].map({0: 'non-senior', 1:'senior'}).astype(str)+ '_' + \
new_churn_df['Partner'].astype(str) + '_' + new_churn_df['Dependents'].astype(str)

#Creating a dataframe to show the count of each unique 'Customer_profile'
pd.DataFrame(new_churn_df['Customer_profile'].value_counts())

#using the seaborn library to create a countplot that visualizes the comparison of churn across different customer profiles.
plt.figure(figsize=(12, 8))
sns.countplot(x='Customer_profile', hue='Churn', data=new_churn_df, palette='Set2')
plt.xticks(rotation=45, ha='right')
plt.title('Churn Comparison by Customer Profile')
plt.show()

"""#Question 3
###Using the features i defined in question (1) define and train a Multi-Layer Perceptron model using the Functional API
"""

#creating a feature subset of the top correlated features with the churn
top_features= churn_df[top_relevant_features]
top_features

# Choosing the dependent and independent variables
X = top_features
y = churn_df['Churn']

scaler = StandardScaler()
X = scaler.fit_transform(X.copy())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

'''
# Define the preprocessor
scaler = StandardScaler()
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), list(X.columns))
    ],
    remainder='passthrough'
)'''

# Defining the architecture of the MLP model using the Keras Functional API
inputs = Input(shape=(X_train.shape[1],))
x = Dense(64, activation='relu')(inputs)
x = Dense(32, activation='relu')(x)
x = Dense(8, activation='relu')(x)
outputs = Dense(1, activation='sigmoid')(x)

# Creating the model using the Functional API
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Evaluating the model on the test set
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)

# Calculating and printing the accuracy score
accuracy = accuracy_score(y_test, y_pred_binary)
print(f'Accuracy: {accuracy}')

# Define the MLP model
mlp_model = MLPClassifier(max_iter=100)

# Define the hyperparameter grid
param_grid = {
    'hidden_layer_sizes': [(64, 32), (128, 64), (32,)],
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['adam', 'sgd'],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
}

# Perform cross-validated grid search
grid_search = GridSearchCV(mlp_model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Define the MLP model
#mlp_model = MLPClassifier(max_iter=100)

# Create a pipeline with preprocessing and MLP model
#pipeline = Pipeline([
    #('preprocessor', preprocessor),
    #('mlpclassifier', mlp_model)
#])

# Define the hyperparameter grid
#param_grid = {
    #'mlpclassifier__hidden_layer_sizes': [(64, 32), (128, 64), (32,)],
    #'mlpclassifier__activation': ['relu', 'tanh', 'logistic'],
    #'mlpclassifier__solver': ['adam', 'sgd'],
    #'mlpclassifier__learning_rate': ['constant', 'invscaling', 'adaptive'],
#}

# Perform cross-validated grid search
#grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, scoring='accuracy')
#grid_search.fit(X_train, y_train)

# Get the best model
#best_model = grid_search.best_estimator_

# Evaluate the model on the test set
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy}')

# Get the best hyperparameters
best_params = grid_search.best_params_
print(f'Best Hyperparameters: {best_params}')

'''
# Preprocessing using StandardScaler
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), top_features.columns.tolist())
    ],
    remainder='passthrough'
)

# Create MLP model with the best hyperparameters
mlp_model = MLPClassifier(
    activation='relu',
    alpha=0.0001,
    hidden_layer_sizes=(32,),
    learning_rate='adaptive',
    solver='adam'
)

# Create a pipeline with preprocessing and MLP model
pipeline = make_pipeline(preprocessor, mlp_model)




# Train the model (assuming X_train and y_train are your training data)
pipeline.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = pipeline.predict(X_test)
'''
best_params = {'activation': 'tanh', 'hidden_layer_sizes': (64, 32), 'learning_rate': 'constant', 'solver': 'adam'}

tuned_mlp_model = MLPClassifier(
    activation=best_params['activation'],
    hidden_layer_sizes=best_params['hidden_layer_sizes'],
    learning_rate=best_params['learning_rate'],
    solver=best_params['solver'])

tuned_mlp_model.fit(X_train, y_train)

y_pred = tuned_mlp_model.predict(X_test)


# Calculate and print the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

"""#Question 4
###Evaluating the model’s accuracy and calculating the AUC score
"""

# Evaluate the model on the test set
y_pred = tuned_mlp_model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)

# Calculate and print the accuracy score
accuracy = accuracy_score(y_test, y_pred_binary)
print(f'Accuracy: {accuracy}')

# Calculate and print the AUC score
auc_score = roc_auc_score(y_test, y_pred)
print(f'AUC Score: {auc_score}')

import joblib
import pickle


# Save the scikit-learn pipeline (including MLPClassifier) using joblib
#joblib.dump(best_model, 'churning_model.pkl')

joblib.dump(tuned_mlp_model, 'tuned_mlp_model.h5')

with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

with open('label.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)

!pip freeze > requirements.txt