# Extract relevant columns
plot_7_variables = ['arr_flights', 'arr_delay', 'weather_ct', 'nas_ct', 'late_aircraft_ct', 'arr_cancelled' ]
data_2020 = top_20_airport_2020.groupby('month_code')[plot_7_variables].sum()
data_2022 = top_20_airport_2022.groupby('month_code')[plot_7_variables].sum()

# Calculate correlation matrices
correlation_matrix_2020 = data_2020.corr()
correlation_matrix_2022 = data_2022.corr()

# Plotting
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

# Heatmap for 2020
sns.heatmap(correlation_matrix_2020, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=axes[0])
axes[0].set_title('Correlation Heatmap - 2020')

# Heatmap for 2022
sns.heatmap(correlation_matrix_2022, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=axes[1])
axes[1].set_title('Correlation Heatmap - 2022')

plt.tight_layout()
plt.show()# Extract relevant columns
plot_7_variables = ['arr_flights', 'arr_delay', 'weather_ct', 'nas_ct', 'late_aircraft_ct', 'arr_cancelled' ]
data_2020 = top_20_airport_2020.groupby('month_code')[plot_7_variables].sum()
data_2022 = top_20_airport_2022.groupby('month_code')[plot_7_variables].sum()

# Calculate correlation matrices
correlation_matrix_2020 = data_2020.corr()
correlation_matrix_2022 = data_2022.corr()

# Plotting
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

# Heatmap for 2020
sns.heatmap(correlation_matrix_2020, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=axes[0])
axes[0].set_title('Correlation Heatmap - 2020')

# Heatmap for 2022
sns.heatmap(correlation_matrix_2022, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=axes[1])
axes[1].set_title('Correlation Heatmap - 2022')

plt.tight_layout()
plt.show()# Extract relevant columns
plot_7_variables = ['arr_flights', 'arr_delay', 'weather_ct', 'nas_ct', 'late_aircraft_ct', 'arr_cancelled' ]
data_2020 = top_20_airport_2020.groupby('month_code')[plot_7_variables].sum()
data_2022 = top_20_airport_2022.groupby('month_code')[plot_7_variables].sum()

# Calculate correlation matrices
correlation_matrix_2020 = data_2020.corr()
correlation_matrix_2022 = data_2022.corr()

# Plotting
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

# Heatmap for 2020
sns.heatmap(correlation_matrix_2020, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=axes[0])
axes[0].set_title('Correlation Heatmap - 2020')

# Heatmap for 2022
sns.heatmap(correlation_matrix_2022, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=axes[1])
axes[1].set_title('Correlation Heatmap - 2022')

plt.tight_layout()
plt.show()# Extract relevant columns
plot_7_variables = ['arr_flights', 'arr_delay', 'weather_ct', 'nas_ct', 'late_aircraft_ct', 'arr_cancelled' ]
data_2020 = top_20_airport_2020.groupby('month_code')[plot_7_variables].sum()
data_2022 = top_20_airport_2022.groupby('month_code')[plot_7_variables].sum()

# Calculate correlation matrices
correlation_matrix_2020 = data_2020.corr()
correlation_matrix_2022 = data_2022.corr()

# Plotting
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

# Heatmap for 2020
sns.heatmap(correlation_matrix_2020, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=axes[0])
axes[0].set_title('Correlation Heatmap - 2020')

# Heatmap for 2022
sns.heatmap(correlation_matrix_2022, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=axes[1])
axes[1].set_title('Correlation Heatmap - 2022')

plt.tight_layout()
plt.show()import pandas as pd 
import os 
import opendatasets as od
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

# Read and display data
flight_delay_URL= "https://www.kaggle.com/datasets/sriharshaeedala/airline-delay/"
od.download(flight_delay_URL, force=True)

# Display data
flight_delay= pd.read_csv("resources\Airline_Delay_Cause.csv")
flight_delay

Data Cleaning

# Drop last five columns(data not needed for analysis), drop "carrier" and ,"airport_name" column
last_five_columns = ['carrier','airport_name','carrier_delay', 'weather_delay','nas_delay','security_delay', 'late_aircraft_delay']
flight_delay.drop(columns= last_five_columns, inplace=True)
flight_delay

# Use the `count` function to view count of non-null values for each column
flight_delay.count()


# Check datatypes 
flight_delay.dtypes

# Count nulls 
flight_delay.isnull().sum()

# Drop Nan from data set
flight_delay= flight_delay.dropna()
flight_delay

# Set 'Year' as index 
flight_delay.set_index(flight_delay['year'], inplace=True)
# Drop extra date column
flight_delay.drop(columns=['year'],inplace=True)
# Display data
flight_delay.head()

# Drop year 2013-2019, 2021 & 2023
first_years= [2013,2014,2015,2016,2017, 2018,2019,2021,2023]
flight_delay.drop(index=first_years, inplace=True)
flight_delay

# reset index column
yearly_flight_delay= flight_delay.reset_index()
yearly_flight_delay

#Count frequency of "carrier_name" column
flight_delay['carrier_name'].value_counts()

#Collect top 20 airports from dataframe

top_20_total_arrival_list = ["ATL","BOS","CLT","DCA","DEN","DFW","DTW","EWR","IAH","JFK","LAS","LAX","LGA","MCO","MSP","ORD","PHX","SEA","SFO","SLC","TPA"]
#creating top 20 DataFrame from the airport data
top_20_airport=yearly_flight_delay[yearly_flight_delay["airport"].isin(["ATL","BOS","CLT","DCA","DEN","DFW","DTW","EWR","IAH","JFK","LAS","LAX","LGA","MCO","MSP","ORD","PHX","SEA","SFO","SLC","TPA"])]
top_20_airport.tail()

#recount frequency of "airport" column
top_20_airport['airport'].value_counts()

#recount frequency of "carrier_name" column
top_20_airport['carrier_name'].value_counts()

# rename 'month' column to 'month_code', 'carrier_name' to 'carrier'
top_20_airport = top_20_airport.rename(columns={
    "month": "month_code",
    "carrier_name": "carrier"
})
top_20_airport.head()

# Descriptive analysis on dataset
top_20_airport.describe()

#Performance analysis

#Plot the total number of arriving and delayed flight for each month to identify any sesonal trends or variations.

from matplotlib.ticker import FuncFormatter
# Filter data for the years 2020 and 2023
top_20_airport_2020 = top_20_airport[top_20_airport['year'] == 2020]
top_20_airport_2022 = top_20_airport[top_20_airport['year'] == 2022]

# Define variables for plot
Plot_1_variables = ['arr_flights', 'arr_delay']

# Create subplots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

# Plot for 2020
top_20_airport_2020.groupby('month_code')[Plot_1_variables].sum().plot(ax=axes[0], kind='bar', title='Monthly Analysis - 2020')
axes[0].set_ylabel('Scale')
# Plot for 2022
top_20_airport_2022.groupby('month_code')[Plot_1_variables].sum().plot(ax=axes[1], kind='bar', title='Monthly Analysis - 2022')
axes[1].set_ylabel('Scale')

for ax in axes:
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:,.0f}'.format(x)))


plt.tight_layout()
plt.show()



#Plot graph comparing the number of delayed, cancelled, and diverted flight for each month.

# Define variables for plot 
plot_2_variables = ['arr_del15','arr_cancelled', 'arr_diverted']

# Create subplots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

# Plot for 2020
top_20_airport_2020.groupby('month_code')[plot_2_variables].sum().plot(ax=axes[0], kind='bar', title='Monthly Analysis - 2020')
axes[0].set_ylabel('Scale')
# Plot for 2022
top_20_airport_2022.groupby('month_code')[plot_2_variables].sum().plot(ax=axes[1], kind='bar', title='Monthly Analysis - 2022')
axes[1].set_ylabel('Scale')

for ax in axes:
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:,.0f}'.format(x)))


plt.tight_layout()
plt.show()



# Define variables for plot
plot_3_variables = ['arr_flights', 'arr_delay']

# Group by month and sum for both years
data_2020 = top_20_airport_2020.groupby('month_code')[plot_3_variables].sum()
data_2022 = top_20_airport_2022.groupby('month_code')[plot_3_variables].sum()

# Calculate the ratio of delayed flights to arriving flights
data_2020['delayed_ratio'] = data_2020['arr_delay'] / data_2020['arr_flights']
data_2022['delayed_ratio'] = data_2022['arr_delay'] / data_2022['arr_flights']

# Plotting individual line graphs for delayed ratio
plt.figure(figsize=(12, 6))

plt.plot(data_2020.index, data_2020['delayed_ratio'], label='2020', marker='o', color='red')
plt.plot(data_2022.index, data_2022['delayed_ratio'], label='2022', marker='o', color='magenta')

plt.title('Delayed Flight Ratio (Monthly) - 2020 vs 2022')
plt.xlabel('Month')
plt.ylabel('Delayed Flight Ratio')
plt.legend()
plt.grid(True)
plt.show()



# Define variables for plot 
plot_4_variables = ['arr_flights', 'arr_cancelled']

# Group by month and sum for both years
data_2020 = top_20_airport_2020.groupby('month_code')[plot_4_variables].sum()
data_2022 = top_20_airport_2022.groupby('month_code')[plot_4_variables].sum()

# Calculate the ratio of delayed flights to arriving flights
data_2020['delayed_ratio'] = data_2020['arr_cancelled'] / data_2020['arr_flights']
data_2022['delayed_ratio'] = data_2022['arr_cancelled'] / data_2022['arr_flights']

# Plotting individual line graphs for delayed ratio
plt.figure(figsize=(12, 6))

plt.plot(data_2020.index, data_2020['delayed_ratio'], label='2020', marker='o', color='red')
plt.plot(data_2022.index, data_2022['delayed_ratio'], label='2022', marker='o', color='magenta')

plt.title('Cancelled Flight Ratio (Monthly) - 2020 vs 2022')
plt.xlabel('Month')
plt.ylabel('Cancelled Flight Ratio')
plt.legend()
plt.grid(True)
plt.show()



# Define variables for plot
plot_5_variables = ['arr_flights', 'arr_delay']

# Group by month and sum for both years
data_2020 = top_20_airport_2020.groupby('month_code')[plot_5_variables].sum()
data_2022 = top_20_airport_2022.groupby('month_code')[plot_5_variables].sum()

# Plotting
plt.figure(figsize=(9, 4))

plt.scatter(data_2020['arr_flights'], data_2020['arr_delay'], label='2020', color='orange')
plt.scatter(data_2022['arr_flights'], data_2022['arr_delay'], label='2022', color='green')

for ax in axes:
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:,.0f}'.format(x)))

plt.title('Relation Between 2020 and 2022 Analysis')
plt.xlabel('Total Arrival Flights')
plt.ylabel('Total Arrival Delays')
plt.legend()
plt.grid(True)
plt.show()



#Root Cause Analysis

# Define variables for plot 
plot_6_variables = ['carrier_ct','weather_ct', 'nas_ct', 'security_ct', 'late_aircraft_ct']

# Create subplots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

# Plot for 2020
top_20_airport_2020.groupby('month_code')[plot_6_variables].sum().plot(ax=axes[0], kind='bar', title='Root_Cause Analysis - 2020')
axes[0].set_ylabel('Scale')
# Plot for 2022
top_20_airport_2022.groupby('month_code')[plot_6_variables].sum().plot(ax=axes[1], kind='bar', title='Root_Cause Analysis - 2022')
axes[1].set_ylabel('Scale')

for ax in axes:
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:,.0f}'.format(x)))


plt.tight_layout()
plt.show()


#Correlation Analaysis¶


# Extract relevant columns
plot_7_variables = ['arr_flights', 'arr_delay', 'weather_ct', 'nas_ct', 'late_aircraft_ct', 'arr_cancelled' ]
data_2020 = top_20_airport_2020.groupby('month_code')[plot_7_variables].sum()
data_2022 = top_20_airport_2022.groupby('month_code')[plot_7_variables].sum()

# Calculate correlation matrices
correlation_matrix_2020 = data_2020.corr()
correlation_matrix_2022 = data_2022.corr()

# Plotting
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

# Heatmap for 2020
sns.heatmap(correlation_matrix_2020, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=axes[0])
axes[0].set_title('Correlation Heatmap - 2020')

# Heatmap for 2022
sns.heatmap(correlation_matrix_2022, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=axes[1])
axes[1].set_title('Correlation Heatmap - 2022')

plt.tight_layout()
plt.show()# Extract relevant columns
plot_7_variables = ['arr_flights', 'arr_delay', 'weather_ct', 'nas_ct', 'late_aircraft_ct', 'arr_cancelled' ]
data_2020 = top_20_airport_2020.groupby('month_code')[plot_7_variables].sum()
data_2022 = top_20_airport_2022.groupby('month_code')[plot_7_variables].sum()

# Calculate correlation matrices
correlation_matrix_2020 = data_2020.corr()
correlation_matrix_2022 = data_2022.corr()

# Plotting
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

# Heatmap for 2020
sns.heatmap(correlation_matrix_2020, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=axes[0])
axes[0].set_title('Correlation Heatmap - 2020')

# Heatmap for 2022
sns.heatmap(correlation_matrix_2022, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=axes[1])
axes[1].set_title('Correlation Heatmap - 2022')

plt.tight_layout()
plt.show()# Extract relevant columns
plot_7_variables = ['arr_flights', 'arr_delay', 'weather_ct', 'nas_ct', 'late_aircraft_ct', 'arr_cancelled' ]
data_2020 = top_20_airport_2020.groupby('month_code')[plot_7_variables].sum()
data_2022 = top_20_airport_2022.groupby('month_code')[plot_7_variables].sum()

# Calculate correlation matrices
correlation_matrix_2020 = data_2020.corr()
correlation_matrix_2022 = data_2022.corr()

# Plotting
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

# Heatmap for 2020
sns.heatmap(correlation_matrix_2020, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=axes[0])
axes[0].set_title('Correlation Heatmap - 2020')

# Heatmap for 2022
sns.heatmap(correlation_matrix_2022, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=axes[1])
axes[1].set_title('Correlation Heatmap - 2022')

plt.tight_layout()
plt.show()# Extract relevant columns
plot_7_variables = ['arr_flights', 'arr_delay', 'weather_ct', 'nas_ct', 'late_aircraft_ct', 'arr_cancelled' ]
data_2020 = top_20_airport_2020.groupby('month_code')[plot_7_variables].sum()
data_2022 = top_20_airport_2022.groupby('month_code')[plot_7_variables].sum()

# Calculate correlation matrices
correlation_matrix_2020 = data_2020.corr()
correlation_matrix_2022 = data_2022.corr()

# Plotting
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

# Heatmap for 2020
sns.heatmap(correlation_matrix_2020, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=axes[0])
axes[0].set_title('Correlation Heatmap - 2020')

# Heatmap for 2022
sns.heatmap(correlation_matrix_2022, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=axes[1])
axes[1].set_title('Correlation Heatmap - 2022')

plt.tight_layout()
plt.show()# Extract relevant columns
plot_7_variables = ['arr_flights', 'arr_delay', 'weather_ct', 'nas_ct', 'late_aircraft_ct', 'arr_cancelled' ]
data_2020 = top_20_airport_2020.groupby('month_code')[plot_7_variables].sum()
data_2022 = top_20_airport_2022.groupby('month_code')[plot_7_variables].sum()

# Calculate correlation matrices
correlation_matrix_2020 = data_2020.corr()
correlation_matrix_2022 = data_2022.corr()

# Plotting
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

# Heatmap for 2020
sns.heatmap(correlation_matrix_2020, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=axes[0])
axes[0].set_title('Correlation Heatmap - 2020')

# Heatmap for 2022
sns.heatmap(correlation_matrix_2022, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=axes[1])
axes[1].set_title('Correlation Heatmap - 2022')

plt.tight_layout()
plt.show()# Extract relevant columns
plot_7_variables = ['arr_flights', 'arr_delay', 'weather_ct', 'nas_ct', 'late_aircraft_ct', 'arr_cancelled' ]
data_2020 = top_20_airport_2020.groupby('month_code')[plot_7_variables].sum()
data_2022 = top_20_airport_2022.groupby('month_code')[plot_7_variables].sum()

# Calculate correlation matrices
correlation_matrix_2020 = data_2020.corr()
correlation_matrix_2022 = data_2022.corr()

# Plotting
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

# Heatmap for 2020
sns.heatmap(correlation_matrix_2020, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=axes[0])
axes[0].set_title('Correlation Heatmap - 2020')

# Heatmap for 2022
sns.heatmap(correlation_matrix_2022, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=axes[1])
axes[1].set_title('Correlation Heatmap - 2022')

plt.tight_layout()
plt.show()# Extract relevant columns
plot_7_variables = ['arr_flights', 'arr_delay', 'weather_ct', 'nas_ct', 'late_aircraft_ct', 'arr_cancelled' ]
data_2020 = top_20_airport_2020.groupby('month_code')[plot_7_variables].sum()
data_2022 = top_20_airport_2022.groupby('month_code')[plot_7_variables].sum()

# Calculate correlation matrices
correlation_matrix_2020 = data_2020.corr()
correlation_matrix_2022 = data_2022.corr()

# Plotting
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

# Heatmap for 2020
sns.heatmap(correlation_matrix_2020, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=axes[0])
axes[0].set_title('Correlation Heatmap - 2020')

# Heatmap for 2022
sns.heatmap(correlation_matrix_2022, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=axes[1])
axes[1].set_title('Correlation Heatmap - 2022')

plt.tight_layout()
plt.show()
