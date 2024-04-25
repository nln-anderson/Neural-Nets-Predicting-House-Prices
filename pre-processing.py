from prelim_data_analysis import df
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


# Changing ocean_prox to dummy var
print(df['ocean_proximity'].value_counts())
df['ocean_proximity'].replace(['<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND'], [1,0,2,3,4], inplace=True)
print(df.head(10))

# View Correlations (-> correlation_matrix.png in directory)
# sb.heatmap(df.corr(), annot=True, cmap="YlGnBu")
# plt.title("Correlation Matrix")
# plt.show()

# VIF Data
# vif_data = pd.DataFrame()
# vif_data["feature"] = df.columns
# vif_data['VIF'] = [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]
# print(vif_data)
# # vif_data.to_csv("vif.csv")
# import dataframe_image as dfi
# dfi.export(vif_data,"vif.png")

# Dropping total_bedrooms
df = df.drop(['total_bedrooms'], axis=1)
print(df.isnull().sum())

# Creating new variable rooms per person and dropping total_rooms
df['rooms_per_person'] = df['total_rooms'] / df['population']
df = df.drop(['total_rooms'], axis=1)

# Creating house per person and dropping households
df['houses_per_person'] = df["households"] / df['population']
df = df.drop(['households'], axis=1)

# New Correlation Matrix
sb.heatmap(df.corr(), annot=True, cmap="YlGnBu")
plt.title("Correlation Matrix after Dropping Variables")
plt.show()

# Remove outliers (since they are outliers)
df = df[df['housing_median_age'] <= 50]
df = df[df['median_house_value'] <= 475000]
df = df[df['ocean_proximity'] != 4]

# Create new distribution chart
df.hist(bins=100)
plt.title("Distributions of Varaibles after Pre-Processing")
plt.show()

# Export the dataset
df.to_csv("preprocessed_data.csv")