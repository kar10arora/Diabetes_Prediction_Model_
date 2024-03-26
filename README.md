Predict whether Someone has Diabetes using machine learning¶
Using the Diabetes dataset from kaggle and using data science and machine learning model tools to create a model that predicts that 
whether a patient has diabetes or not...

Using this 5 important approach:
1.Problem definition
2.Data collection
3.Evaluation
4.Data modelling
5.Experimentation

# Problem Defintion
Based on the clinical Parameters,,whether a patient has diabetes or not??

# Data Collection
The data on the basis of which the model is trained is taken from kaggle community dataset link :- https://www.kaggle.com/datasets/iammustafatz/
diabetes-prediction-dataset

# Evaluation
If we can reach 95% accuracy at predicting whether or not a patient has diabetes during the proof of concept, we'll pursue the project.

# Features
The Diabetes prediction dataset is a collection of medical and demographic data from patients, along with their diabetes status (positive or negative).
The data includes features such as age, gender, body mass index (BMI), hypertension, heart disease, smoking history, HbA1c level, and 
blood glucose level. This dataset can be used to build machine learning models to predict diabetes in patients based on their medical
history and demographic information. This can be useful for healthcare professionals in identifying patients who may be at risk of developing
diabetes and in developing personalized treatment plans. Additionally, the dataset can be used by researchers to explore the relationships 
between various medical and demographic factors and the likelihood of developing diabetes.

# Necessary Tools
# import the necessary tools to visualize the data and create the model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
%matplotlib inline

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report,recall_score,precision_score,f1_score,confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import train_test_split , cross_val_score
