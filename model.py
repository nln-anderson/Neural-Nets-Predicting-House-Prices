# The machine learning model

import sklearn
import numpy as np
import pandas as pd
import sklearn.metrics
import sklearn.model_selection
import sklearn.neural_network
import sklearn.preprocessing
import matplotlib.pyplot as plt
import seaborn as sb

# Import the data
df = pd.read_csv("preprocessed_data.csv")
df = df.drop(['Unnamed: 0'], axis=1)

# Convert to numpy
data = df.to_numpy()
print(data.shape)

# Create our training and testing sets
X = df.drop(['median_house_value'], axis=1)
X = X.to_numpy()

y = df['median_house_value']
y = y.to_numpy()

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y,test_size= 0.3,random_state= 42)

# Scale the X sets
scaler = sklearn.preprocessing.MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# Create network
network = sklearn.neural_network.MLPRegressor(solver= 'adam', hidden_layer_sizes=(5), verbose=True, learning_rate_init=0.1, max_iter=300)

# Train the network
network.fit(X_train, y_train)

# Test the network
predictions = network.predict(X_test)

print(f"RMSE: {np.sqrt(sklearn.metrics.mean_squared_error(y_test, predictions))}")

# Print Loss Function
pd.DataFrame(network.loss_curve_).plot()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# Showing true vs predicted
plt.figure(figsize=(10,10))
plt.scatter(y_test, predictions, c='crimson')
plt.yscale('log')
plt.xscale('log')

p1 = max(max(predictions), max(y_test))
p2 = min(min(predictions), min(y_test))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()

