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

# Remove outliers (since they are outliers)
df = df[df['housing_median_age'] <= 50]
df = df[df['median_house_value'] <= 450000]
df = df[df['ocean_proximity'] != 4]
# df.hist(bins=100)
# plt.title("Distributions of Varaibles before Pre-Processing")
# plt.show()

# Plotting latitudes and longitudes
# plt.figure(figsize=(10, 6))
# plt.scatter(df['longitude'], df['latitude'], s=5, color='blue')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.title('Latitude vs Longitude of California Data')
# plt.grid(True)
# plt.show()

# Creating ZIP Codes
from arcgis.geocoding import reverse_geocode
from arcgis.geometry import Geometry
from arcgis.gis import GIS
import pandas as pd

gis = GIS("http://www.arcgis.com", "nolio1", "PD_hU2L.CLzM2sD")

def get_zip(df, lon_field, lat_field):
    location = reverse_geocode((Geometry({"x":float(df[lon_field]), "y":float(df[lat_field]), "spatialReference":{"wkid": 4326}})))
    return location['address']['Postal']

df2 = pd.DataFrame({
    'Lat': [29.39291, 29.39923, 29.40147, 29.38752, 29.39291, 29.39537, 29.39343, 29.39291, 29.39556],
    'Lon': [-98.50925, -98.51256, -98.51123, -98.52372, -98.50925, -98.50402, -98.49707, -98.50925, -98.53148]
})

zipcodes = df2.apply(get_zip, axis=1, lat_field='Lat', lon_field='Lon')

print(zipcodes.head(10))

# Moving the data to numpy
# Features
X = df.drop('median_house_value',axis=1)

# Label
y = df['median_house_value']

# Split
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y,test_size=0.3,random_state=42)

df = df.to_numpy()

scaler = sklearn.preprocessing.MinMaxScaler()

# fit and transfrom
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# everything has been scaled between 1 and 0
print('Max: ',X_train.max())
print('Min: ', X_train.min())

# Building the model
# model = tf.keras.Sequential()

# # input layer
# model.add(tf.keras.layers.Dense(9,activation='relu'))

# # hidden layers
# model.add(tf.keras.layers.Dense(9,activation='relu'))
# model.add(tf.keras.layers.Dense(5,activation='relu'))
# model.add(tf.keras.layers.Dense(2,activation='relu'))

# # output layer
# model.add(tf.keras.layers.Dense(1))

# model.compile(optimizer='adam',loss='mse')

# model.fit(x=X_train,y=y_train.values,
#           validation_data=(X_test,y_test.values),
#           batch_size=128,epochs=400)

# losses = pd.DataFrame(model.history.history)

# # Plotting losses
# plt.figure(figsize=(15,5))
# sb.lineplot(data=losses,lw=3)
# plt.xlabel('Epochs')
# plt.ylabel('')
# plt.title('Training Loss per Epoch')
# sb.despine()

# # predictions on the test set
# predictions = model.predict(X_test)

# # Predictions
# print('MAE: ',sklearn.metrics.mean_absolute_error(y_test,predictions))
# print('MSE: ', sklearn.metrics.mean_squared_error(y_test,predictions))
# print('RMSE: ',np.sqrt(sklearn.metrics.mean_squared_error(y_test,predictions)))
# print('Variance Regression Score: ',sklearn.metrics.explained_variance_score(y_test,predictions))

# print('\n\nDescriptive Statistics:\n',df['price'].describe())



