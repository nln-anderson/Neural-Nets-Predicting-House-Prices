from prelim_data_analysis import df
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


# Dropping instances with missing bedrooms
df = df.dropna()
print(df.isnull().sum())

# Changing ocean_prox to dummy var
print(df['ocean_proximity'].value_counts())
df['ocean_proximity'].replace(['<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND'], [1,0,2,3,4], inplace=True)
print(df.head(10))

# View Correlations (-> correlation_matrix.png in directory)
# sb.heatmap(df.corr(), annot=True, cmap="YlGnBu")
# plt.title("Correlation Matrix")
# plt.show()

# VIF Data
vif_data = pd.DataFrame()
vif_data["feature"] = df.columns
vif_data['VIF'] = [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]
print(vif_data)
# vif_data.to_csv("vif.csv")
import dataframe_image as dfi
dfi.export(vif_data,"vif.png")