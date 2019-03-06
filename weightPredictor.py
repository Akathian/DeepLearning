import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

# read data
# dataframe = pd.read_fwf('brain_body.txt') 
dataframe = pd.read_csv('challenge_dataset.txt', sep=',', 
	header=None, names=['Brain', 'Body'])

# brain measurments
x_values = dataframe[['Brain']]
# body measurements
y_values = dataframe[['Body']]

# train model on data using linear regression
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)

# visualize results
plt.scatter(x_values, y_values)
plt.plot(x_values, body_reg.predict(x_values))
plt.show()
