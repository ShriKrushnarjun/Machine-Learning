import pandas as pd
import quandl, math, datetime
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style

# Quering data frm server
df = quandl.get("WIKI/GOOGL")

# Cutting out features we want and creating some new
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume', ]]
df["HL_PCR"] = (df["Adj. High"] - df["Adj. Close"]) / df["Adj. Close"] * 100
df["PCR_Change"] = (df["Adj. Close"] - df["Adj. Open"]) / df["Adj. Open"] * 100

# Creating future entries in forecast column
forecast_col = "Adj. Close"
df.fillna(-99999, inplace=True)

# Creating future entries in forecast column with respect to Adj. Close and removing NAN
forecast_out = int(math.ceil(0.01 * len(df)))
df["Forecast"] = df[forecast_col].shift(-forecast_out)

# Collecting and removing NAN data rows
new_df = df.tail(forecast_out)
del new_df["Forecast"]
df.dropna(inplace=True)

# Collection data for classifier
X = np.array(df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume', 'HL_PCR', 'PCR_Change']])
y = np.array(df["Forecast"])

# Dividing data for training and testing
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# Classification Algorithem and n_jobs shows thredding instances at a time
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

# Predictions
forecast_set = clf.predict(new_df)
print(forecast_set, accuracy, forecast_out)

# Plotting Graph
style.use("ggplot")

# Collecting data for graph and setting graph elements
new_df.insert(len(new_df.columns), "Forecast", forecast_set)
df = df.append(new_df)

df["Forecast"].plot()
plt.legend(loc=4)
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()
