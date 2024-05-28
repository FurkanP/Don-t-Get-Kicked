# Import Libraries and Check Data
#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from sklearn.model_selection import *
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.metrics import classification_report
import plotly.express as px
from sklearn.model_selection import cross_val_predict, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_row', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%3f' % x)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Read Data
#%%
train_df = pd.read_csv("/kaggle/input/DontGetKicked/training.csv")
test_df = pd.read_csv("/kaggle/input/DontGetKicked/test.csv")
submissions_df = test_df[['RefId']]
print(train_df.shape)

# Function to Check Dataframe
#%%
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1], numeric_only=True).T)
check_df(train_df)

# Visualization
#%%
# Target Variable Visualization
target_counts = train_df.IsBadBuy.value_counts()
plt.figure(figsize=(5, 4))
sns.barplot(x=['No', 'Yes'], y=target_counts, palette=['g', 'r'])
plt.ylabel('Target_Counts')
plt.title("Is Bad Buy")
plt.show()

# Note: The data is imbalanced
#%%
# VehicleAge Variable Visualization
vehicle_age = pd.DataFrame(train_df["VehicleAge"].value_counts()).sort_index()
vehicle_age.columns = ["Vehicle_age"]
vehicle_age.index.name = ""
plt.figure(figsize=(6, 4))
sns.barplot(x=vehicle_age.index, y='Vehicle_age', data=vehicle_age, palette='Set2')
plt.ylabel('Vehicle Counts')
plt.xlabel('Age of Vehicle')
plt.title("Vehicle Age")
plt.show()

# Nationality with IsBadBuy
#%%
nationality_target = px.histogram(train_df, x="Nationality", color="IsBadBuy")
nationality_target.show()

# VehicleAge with IsBadBuy
#%%
plt.title("Vehicle Age vs Is Bad Buy")
sns.countplot(x="VehicleAge", data=train_df, hue="IsBadBuy", palette=['y', 'c'])
plt.legend(['0:GoodBuy', '1:BadBuy'])
plt.show()

# Make with IsBadBuy
#%%
make_target = px.histogram(train_df, x="Make", color="IsBadBuy", height=500, width=800)
make_target.show()

# Numeric Variable Visualization
#%%
numerical_features = train_df.select_dtypes(include=['float64', 'int64']).columns.drop('RefId')
train_df[numerical_features].hist(figsize=(20, 15), color="#3498db", bins=30, xlabelsize=8, ylabelsize=8)

# Correlation Matrix
#%%
plt.figure(figsize=(25, 10))
sns.heatmap(train_df[numerical_features].corr(), cmap=sns.diverging_palette(220, 10, as_cmap=True), annot=True, linewidths=.5, fmt='.3f')
plt.show()

# Feature Engineering
#%%
# Transform Date Columns
def transform_data(df):
    df['PurchDate'] = pd.to_datetime(df['PurchDate'])
    return df

train_df = transform_data(train_df)
test_df = transform_data(test_df)

# Create Date Features
def date_feature(df):
    df['PurchYear'] = df['PurchDate'].dt.year
    df['PurchMonth'] = df['PurchDate'].dt.month
    return df

train_df = date_feature(train_df)
test_df = date_feature(test_df)

# Define Seasons
def seasons(df):
    if df['PurchMonth'] >= 3 and df['PurchMonth'] <= 5:
        val = 'Spring'
    elif df['PurchMonth'] >= 6 and df['PurchMonth'] <= 8:
        val = 'Summer'
    elif df['PurchMonth'] >= 9 and df['PurchMonth'] <= 11:
        val = 'Autumn'
    else:
        val = 'Winter'
    return val

train_df['Seasons'] = train_df.apply(seasons, axis=1)
test_df['Seasons'] = test_df.apply(seasons, axis=1)

# Encode SubModel Feature
def submodel(df):
    df.loc[:, 'SubModel_SEDAN'] = df.loc[:, 'SubModel'].str.contains('SEDAN', case=False, na=False).astype(int)
    df.loc[:, 'SubModel_CAB'] = df.loc[:, 'SubModel'].str.contains('CAB', case=False, na=False).astype(int)
    df.loc[:, 'SubModel_CUV'] = df.loc[:, 'SubModel'].str.contains('CUV', case=False, na=False).astype(int)
    df.loc[:, 'SubModel_MINIVAN'] = df.loc[:, 'SubModel'].str.contains('MINIVAN', case=False, na=False).astype(int)
    df.loc[:, 'SubModel_UTILITY'] = df.loc[:, 'SubModel'].str.contains('UTILITY', case=False, na=False).astype(int)
    df.loc[:, 'SubModel_SPORT'] = df.loc[:, 'SubModel'].str.contains('SPORT', case=False, na=False).astype(int)
    df.loc[:, 'SubModel_PASSENGER'] = df.loc[:, 'SubModel'].str.contains('PASSENGER', case=False, na=False).astype(int)
    df.loc[:, 'SubModel_SUV'] = df.loc[:, 'SubModel'].str.contains('SUV', case=False, na=False).astype(int)
    df.loc[:, 'SubModel_WAGON'] = df.loc[:, 'SubModel'].str.contains('WAGON', case=False, na=False).astype(int)
    df.loc[:, 'SubModel_CONVERTIBLE'] = df.loc[:, 'SubModel'].str.contains('CONVERTIBLE', case=False, na=False).astype(int)
    df.loc[:, 'SubModel_HATCHBACK'] = df.loc[:, 'SubModel'].str.contains('HATCHBACK', case=False, na=False).astype(int)
    return df

train_df = submodel(train_df)
test_df = submodel(test_df)

# Create Cost and Mileage Features
def cost_miles_cols(df):
    df.loc[:, 'cost_per_mile'] = df.loc[:, 'VehBCost'] / df.loc[:, 'VehOdo']
    df.loc[:, 'warranty_per_cost'] = df.loc[:, 'WarrantyCost'] / df.loc[:, 'VehBCost']
    df.loc[:, 'warranty_per_mile'] = df.loc[:, 'WarrantyCost'] / df.loc[:, 'VehOdo']

    df.loc[:, 'Age_prep'] = df.loc[:, 'VehicleAge'] + 1
    df.loc[:, 'cost_per_year'] = df.loc[:, 'VehBCost'] / df.loc[:, 'Age_prep']
    df.loc[:, 'miles_per_year'] = df.loc[:, 'VehOdo'] / df.loc[:, 'Age_prep']

    return df

train_df = cost_miles_cols(train_df)
test_df = cost_miles_cols(test_df)

# Display Processed Data
#%%
print(train_df.head())

# Drop Unnecessary Columns
#%%
unique_id = ['RefId', 'BYRNO']
with_many_categories = ['VNZIP1', 'PurchDate', 'PurchMonth', 'Make', 'Model', 'SubModel', 'Trim', 'VNST', 'Color']
redundant = ['WheelTypeID']
high_correlation = ['MMRCurrentAuctionCleanPrice',    # 99% corr with MMRCurrentAuctionAveragePrice
                    'MMRCurrentRetailCleanPrice',      # 99% corr with MMRCurrentRetailAveragePrice
                    'MMRAcquisitionAuctionCleanPrice',  # 99% corr with MMRAcquisitionAuctionAveragePrice
                    'MMRAcquisitonRetailCleanPrice',    # 99% corr with MMRQcquisitionRetailAverageprice
                    'VehYear'                          # 96% corr with VehicleAge
                   ]
columns_to_drop = unique_id + with_many_categories + redundant + high_correlation
train_df.drop(columns_to_drop, axis=1, inplace=True)
test_df.drop(columns_to_drop, axis=1, inplace=True)

# Separate Target and Features
#%%
targets = train_df['IsBadBuy']
train_df.drop('IsBadBuy', axis=1, inplace
