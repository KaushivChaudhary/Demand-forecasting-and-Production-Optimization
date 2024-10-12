# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 11:26:22 2024

@author: Kaushiv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import lag_plot
from scipy.stats import skew, kurtosis
from feature_engine.outliers import Winsorizer
import scipy.stats as stats
import pylab
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pickle, joblib

data=pd.read_csv(r"C:\Users\kaushiv chaudhary\Downloads\Monthlu_data EDA.xlsx")
data.dtypes
duplicates = data.duplicated()
duplicate_rows = data[data.duplicated()]
print(duplicate_rows)

data=data.drop_duplicates()
#outliers 
sns.boxplot(data['Economic Index'])
sns.boxplot(data['Industry Growth Rate (%)'])


winsor_iqr = Winsorizer(capping_method='iqr', 
                        tail='both', 
                        fold=1.5, 
                        variables=['Economic Index','Industry Growth Rate (%)'])

data[['Economic Index', 'Industry Growth Rate (%)']] = winsor_iqr.fit_transform(data[['Economic Index', 'Industry Growth Rate (%)']])

joblib.dump(winsor_iqr, 'winzor')

sns.boxplot(data['Economic Index'])
sns.boxplot(data['Industry Growth Rate (%)'])


#missing values
data['Economic Index'].isnull().sum()
data['Industry Growth Rate (%)'].isnull().sum()
data['Variant'].isna().sum()

stats.probplot(data['Economic Index'], dist="norm", plot=pylab)
data['Economic Index'].skew()
data['Economic Index']=np.log(data['Economic Index'])
stats.probplot(data['Economic Index'], dist="norm", plot=pylab)
data['Economic Index'].skew()

stats.probplot(data['Industry Growth Rate (%)'],dist='norm',plot=pylab)
data['Industry Growth Rate (%)'].skew()
data['Industry Growth Rate (%)'] = np.sqrt(data['Industry Growth Rate (%)'])
data['Industry Growth Rate (%)'].isna().sum()
stats.probplot(data['Industry Growth Rate (%)'], dist="norm", plot=plt)
data['Industry Growth Rate (%)'].skew()

mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
data["Industry Growth Rate (%)"] = pd.DataFrame(mean_imputer.fit_transform(data[['Industry Growth Rate (%)']]))
data['Industry Growth Rate (%)'].isnull().sum()
joblib.dump(mean_imputer,'impute')

scale = StandardScaler()
data[['Economic Index', 'Industry Growth Rate (%)']]= scale.fit_transform(data[['Economic Index','Industry Growth Rate (%)']])
joblib.dump(scale,'scale')

data['Economic Index'].plot() # time series plot 
data['Industry Growth Rate (%)'].plot()

data.to_csv('data_final.csv', index=False)
from IPython.display import FileLink

# Provide a download link for the CSV file
FileLink('data_final.csv')


plt.figure(figsize=(14, 6))

# Plot for Economic Index
plt.subplot(2, 1, 1)
plt.plot(data['Economic Index'], label='Economic Index', color='blue')
plt.title('Economic Index Over Time')
plt.ylabel('Economic Index')
plt.legend()

# Plot for Industry Growth Rate
plt.subplot(2, 1, 2)
plt.plot(data['Industry Growth Rate (%)'], label='Industry Growth Rate (%)', color='green')
plt.title('Industry Growth Rate (%) Over Time')
plt.ylabel('Growth Rate (%)')
plt.legend()

plt.tight_layout()
plt.show()

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(data['Economic Index'], model='additive',period=12)
plt.figure(figsize=(12, 8))
plt.subplot(4, 1, 1)
plt.plot(data['Economic Index'])
plt.title('Original Time Series')
plt.subplot(4, 1, 2)
plt.plot(decomposition.trend.dropna())
plt.title('Trend Component')
plt.subplot(4, 1, 3)
plt.plot(decomposition.seasonal.dropna())
plt.title('Seasonal Component')
plt.subplot(4, 1, 4)
plt.plot(decomposition.resid.dropna())
plt.title('Residual Component')
plt.tight_layout()
plt.show()


decomposition = seasonal_decompose(data['Industry Growth Rate (%)'], model='additive',period=12)
plt.figure(figsize=(12, 8))
plt.subplot(4, 1, 1)
plt.plot(data['Industry Growth Rate (%)'])
plt.title('Original Time Series')
plt.subplot(4, 1, 2)
plt.plot(decomposition.trend.dropna())
plt.title('Trend Component')
plt.subplot(4, 1, 3)
plt.plot(decomposition.seasonal.dropna())
plt.title('Seasonal Component')
plt.subplot(4, 1, 4)
plt.plot(decomposition.resid.dropna())
plt.title('Residual Component')
plt.tight_layout()
plt.show()


# Plot the random walk
plt.figure(figsize=(10, 6))
plt.plot(data['Month'], random_walk, label='Random Walk (Economic Index)')
plt.title('Random Walk for Economic Index over Time')
plt.xlabel('Month')
plt.ylabel('Random Walk Value')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Display the plot
plt.show()

# 80/20 PARETO CHART
import matplotlib.pyplot as plt

# Assuming 'Variant' column exists, and we want to analyze counts
variant_counts = monthly_data['Variant'].value_counts()

# Cumulative sum for Pareto chart
cum_percentage = variant_counts.cumsum() / variant_counts.sum() * 100

# Pareto chart
fig, ax1 = plt.subplots()

# Bar plot for absolute frequencies
ax1.bar(variant_counts.index, variant_counts, color='blue')
ax1.set_ylabel('Frequency', color='blue')
plt.xticks(rotation=45)

# Line plot for cumulative percentage
ax2 = ax1.twinx()
ax2.plot(variant_counts.index, cum_percentage, color='red', marker='o')
ax2.set_ylabel('Cumulative Percentage', color='red')
plt.xticks(rotation=45)
plt.title('80/20 Pareto Chart for Variants')
plt.show()

# KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_data)

# Plot the clusters
plt.figure(figsize=(8, 6))
plt.scatter(data['Economic Index'], data['Industry Growth Rate (%)'], c=data['Cluster'], cmap='viridis', alpha=0.5)
plt.title('Clustering of Economic Index and Industry Growth Rate')
plt.xlabel('Economic Index')
plt.ylabel('Industry Growth Rate (%)')
plt.colorbar(label='Cluster')
plt.show()


# STEPS TO CHECK FOR STATIONARITY

# Plot the time series data to visually check for stationarity
plt.figure(figsize=(10, 5))
plt.plot(data['Month'], data['Economic Index'], label='Economic Index')
plt.plot(data['Month'], data['Industry Growth Rate (%)'], label='Industry Growth Rate (%)')
plt.title('Time Series Data')
plt.xlabel('Billing Date')
plt.ylabel('Values')
plt.legend()
plt.xticks(rotation=45)
plt.show()

# Apply ADF test on both columns
adf_test(data['Economic Index'], 'Economic Index')
adf_test(data['Industry Growth Rate (%)'], 'Industry Growth Rate (%)')

plt.figure(figsize=(10, 6))
plt.plot(data.index, data.iloc[:, 0])  # Assuming you want to plot the first column
plt.title('Time Series')
plt.xlabel('Date')
plt.ylabel('Values')
plt.show()

# Select only the numeric column for rolling calculation
numeric_data = data.select_dtypes(include=['float64', 'int64'])

# Calculate rolling statistics
rolling_mean = numeric_data.rolling(window=12).mean()
rolling_std = numeric_data.rolling(window=12).std()

# Plot rolling statistics
plt.figure(figsize=(10, 6))
plt.plot(data['Economic Index'], label='Original')
plt.plot(data['Industry Growth Rate (%)'])
plt.plot(rolling_mean, color='red', label='Rolling Mean')
plt.plot(rolling_std, color='black', label='Rolling Std')
plt.legend()
plt.show()

# Perform Augmented Dickey-Fuller test
adf_result = adfuller(data.iloc[:, 0])  # Assuming the data has a single column
print('ADF Statistic:', adf_result[0])
print('p-value:', adf_result[1])


import matplotlib.pyplot as plt
plt.hist(data['Economic Index'], bins=10, edgecolor='black')
plt.hist(data['Industry Growth Rate (%)'], bins=10, edgecolor='black')

sns.boxplot(data['Economic Index'])
sns.boxplot(data['Industry Growth Rate (%)'])
import seaborn as sns
sns.boxplot(data=data)


 plt.plot(data['Month'], data['Economic Index'])
 plt.xticks(rotation=45)
 plt.plot(data['Month'], data['Industry Growth Rate (%)'])

 lag_plot(data['Economic Index'])
 lag_plot(data['Industry Growth Rate (%)'])
import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(data['Economic Index'], lags = 12)
tsa_plots.plot_pacf(data['Economic Index'], lags = 12)

tsa_plots.plot_acf(data['Industry Growth Rate (%)'], lags = 12)
tsa_plots.plot_pacf(data['Industry Growth Rate (%)'], lags = 12)

plt.scatter(data['Economic Index'],data['Industry Growth Rate (%)'])
