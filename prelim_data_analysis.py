import pandas as pd
import seaborn as sb
import numpy as np
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn
from statsmodels.stats.outliers_influence import variance_inflation_factor


# Read in the data
df = pd.read_csv("housing.csv")

# Convert the date
# df['date'] = pd.to_datetime(df['date'])
# df['month'] = df['date'].apply(lambda date:date.month)
# df['year'] = df['date'].apply(lambda date:date.year)

# Analying the Data
print(df.head(10))
print(df.columns.values)
print(df.isnull().sum())
print(df.info())

# Dropping instances with missing bedrooms
df = df.dropna()
print(df.isnull().sum())

# Changing ocean_prox to dummy var
print(df['ocean_proximity'].value_counts())
df['ocean_proximity'].replace(['<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND'], [1,0,2,3,4], inplace=True)
print(df.head(10))

# More data analysis
print(df.describe().transpose())

# View Correlations
# sb.heatmap(df.corr(), annot=True, cmap="YlGnBu")
# plt.title("Correlation Matrix")
# plt.show()

# Look at distributions of variables
# df.hist(bins=100)
# plt.title("Distributions of Varaibles before Pre-Processing")
# plt.show()

# VIF Data
vif_data = pd.DataFrame()
vif_data["feature"] = df.columns
vif_data['VIF'] = [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]
print(vif_data)