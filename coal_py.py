#Business problem
'''As of now coal prices are obtained as need be rather than forecasting on what might be the coal prices.
This sometimes leads to spending more cost in purchasing the coal,which is the raw material for steel manufacturing.'''


#Business objective:
'''Maximize cost savings and profitability by optimizing procurement and sourcing strategies through accurate price forecasting.'''

#Business comstraint:
'''Minimize the impact of price volatility on production cost.'''

#Business success criteria:
'''Achieving 10% increase in profit margins through optimized procurement and pricing strategy.'''

#Machine learning success criteria:
'''Achieve an accuray of at least 95%'''

#Economic success criteria:
'''Generating a 20% increase in revenue from coal and iron ore sales within the first year of implememntation.'''

#import necessary libraries.
import pandas as pd
import numpy as np
from sqlalchemy import create_engine,text
import sweetviz
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure
from feature_engine.outliers import Winsorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from matplotlib import pyplot
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import statsmodels.graphics.tsaplots as tsa_plots
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_percentage_error

#import data
data = pd.read_csv(r"C:\Users\ADMIN\OneDrive\Documents\Data science\Project\Data_set.csv")
data

#show firt  5 rows
data.head()

#show the column names
data.columns

#Information about the data 
data.info()
data.describe()
#connect with mysql
user = "root"
pw = "Nibedita12345"
db = "coal_forecast"
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

data.to_sql("coal",con = engine,if_exists = "replace",index = False)

sql = "select * from coal;"
coal = pd.read_sql_query(text(sql),engine.connect())
coal.head(10)


#perform Exploratory data analysis
#first moment business decission(mean, median, mode)


# List of column names to iterate over
columns = [
    'Coal_RB_4800_FOB_London_Close_USD',
    'Coal_RB_5500_FOB_London_Close_USD',
    'Coal_RB_5700_FOB_London_Close_USD',
    'Coal_RB_6000_FOB_CurrentWeek_Avg_USD',
    'Coal_India_5500_CFR_London_Close_USD',
    'Price_WTI',
    'Price_Brent_Oil',
    'Price_Dubai_Brent_Oil',
    'Price_ExxonMobil',
    'Price_Shenhua',
    'Price_All_Share',
    'Price_Mining',
    'Price_LNG_Japan_Korea_Marker_PLATTS',
    'Price_ZAR_USD',
    'Price_Natural_Gas',
    'Price_ICE',
    'Price_Dutch_TTF',
    'Price_Indian_en_exg_rate'
]

# Loop through the columns and calculate mean, median, and mode
for column in columns:
    mean_value = coal[column].mean()
    median_value = coal[column].median()
    mode_value = coal[column].mode()[0]  # mode() returns a Series; [0] gets the first mode

    print(f"{column}:")
    print(f"  Mean: {mean_value}")
    print(f"  Median: {median_value}")
    print(f"  Mode: {mode_value}\n")


#2nd moment business decission(standard deviation , variance, range)
# Loop through the columns and calculate variance, standard deviation, and range
for column in columns:
    variance = coal[column].var()
    std_dev = coal[column].std()
    value_range = coal[column].max() - coal[column].min()

    print(f"{column}:")
    print(f"  Variance: {variance}")
    print(f"  Standard Deviation: {std_dev}")
    print(f"  Range: {value_range}\n")

#3rd moment business decission

# Loop through the columns and calculate skewness 
for column in columns:
    skewness = coal[column].skew()
   

    print(f"{column}:")
    print(f"  Skewness: {skewness}")
    
    
#4th moment business decission

# Loop through the columns and calculate skewness and kurtosis
for column in columns:
    kurtosis = coal[column].kurt()

    print(f"{column}:")
    print(f"  Kurtosis: {kurtosis}\n")  
    


'''Here it is showing that there is some missing values'''

# plot some graphs (univariate,multivariate,bivariate)

'''Bar graph'''
# Loop through the columns and create bar charts 
for column in columns:
    # Bar chart
    x = np.arange(1, len(coal[column]) + 1)
    plt.bar(x, coal[column])
    plt.xlabel('Index')
    plt.ylabel(column)
    plt.title(f'{column} - Bar Chart')
    plt.show()
    
    
'''# Histogram'''

for column in columns:
    plt.hist(coal[column], bins=30)  # Adjust the number of bins as needed
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.title(f'{column} - Histogram')
    plt.show()


'''density plot'''

# Loop through the columns and create density plots
for column in columns:
    plt.figure(figsize=(8, 6))
    sns.kdeplot(data=coal[column], shade=True)
    plt.xlabel(column)
    plt.ylabel('Density')
    plt.title(f'Density Plot of {column}')
    plt.show()

# Create scatter plots for pairs of columns
for i in range(len(columns)):
    for j in range(i + 1, len(columns)):
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=coal[columns[i]], y=coal[columns[j]])
        plt.xlabel(columns[i])
        plt.ylabel(columns[j])
        plt.title(f'Scatter Plot of {columns[i]} vs {columns[j]}')
        plt.show()


'''Pair plot'''

sns.pairplot(coal)
plt.show()


#after plotting all the graphs it is shown that some data are right skewed and some are left skewed.

'''#########################################################   Data preprocessing  #################################################'''

#short the values according to dates.

coal['Date'] = pd.to_datetime(coal['Date'] ,format="%d-%m-%Y", dayfirst=True) 
coal.sort_values(by=['Date'], inplace=True, ascending=True)

coal.head()


#set index for the date column

#coal.set_index('Date')

# Ensure the index is a DatetimeIndex
#coal.index = pd.to_datetime(coal.index)
#coal.head()

# Check the index type to confirm
#print(type(coal.index))


#find missing values
coal.isnull().sum()

#treat the missing values with mean of the column values

#using method forward and backward fill
#plot graph
# List of columns to plot
columns = [
    'Coal_RB_4800_FOB_London_Close_USD',
    'Coal_RB_5500_FOB_London_Close_USD',
    'Coal_RB_5700_FOB_London_Close_USD',
    'Coal_RB_6000_FOB_CurrentWeek_Avg_USD',
    'Coal_India_5500_CFR_London_Close_USD',
    'Price_WTI',
    'Price_Brent_Oil',
    'Price_Dubai_Brent_Oil',
    'Price_ExxonMobil',
    'Price_Shenhua',
    'Price_All_Share',
    'Price_Mining',
    'Price_LNG_Japan_Korea_Marker_PLATTS',
    'Price_ZAR_USD',
    'Price_Natural_Gas',
    'Price_ICE',
    'Price_Dutch_TTF',
    'Price_Indian_en_exg_rate'
]

# Loop through each column and create a plot
for column in columns:
    plt.figure(figsize=(20, 10), dpi=80)
    plt.plot(coal['Date'], coal[column], linewidth=2)
    plt.xlabel('Date')
    plt.ylabel(column)
    plt.title(f'Time Series Plot of {column}')
    plt.grid(True)
    plt.show()

#applying  (forward fill)
# Loop through each column and apply forward fill

coal = coal.ffill()

#check is there any missing values
coal.isnull().sum()

#apply backward interpolation for the rest columns where the missing values can not be removed by forward fill

coal.Price_Indian_en_exg_rate = coal.Price_Indian_en_exg_rate.fillna(method = 'bfill')

coal.isnull().sum()

#check the outliers by boxplot

coal[columns].plot(kind = "box",subplots = True,sharey = False,figsize =(70,40))
plt.subplots_adjust(wspace = 1.5) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()

#applying winsorization techniques using iqr method and fill the outliers with NAN

winsor = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=columns)

# Fit the winsorizer to the data
outlier = winsor.fit(coal[columns])

# Save the winsorizer model
joblib.dump(outlier, 'winsor.pkl')

# Apply the transformation
coal[columns] = outlier.transform(coal[columns]) 
                                            
#again checking the outliers

coal.plot(kind = "box",subplots = True,sharey = False,figsize =(70,40))
plt.subplots_adjust(wspace = 1.5) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()
                                                                                    


# replace capped values with NaN
for col in [columns]:  # iterate over numeric columns
    coal[col] = np.where((coal[col] == coal[col].max()) | (coal[col] == coal[col].min()), np.nan, coal[col])

print(coal)

#again find the null values
coal.isnull().sum()


for column in columns:
    coal[column] = coal[column].interpolate(method='ffill')


#again check null values
coal.isnull().sum()

coal.Price_WTI = coal.Price_WTI.fillna(method = 'bfill')
coal.Price_Brent_Oil = coal.Price_Brent_Oil.fillna(method = 'bfill')
coal.Price_Dubai_Brent_Oil = coal.Price_Dubai_Brent_Oil.fillna(method = 'bfill')
coal.Price_All_Share = coal.Price_All_Share.fillna(method = 'bfill')
coal.Price_Mining = coal.Price_Mining.fillna(method = 'bfill')
coal.Price_Indian_en_exg_rate = coal.Price_Indian_en_exg_rate.fillna(method = 'bfill')
coal.Price_Natural_Gas = coal.Price_Natural_Gas.fillna(method = 'bfill')
coal.Price_ICE = coal.Price_ICE.fillna(method = 'bfill')


#again check there is any  null values
coal.isnull().sum()

################################***********###########**********************************##########################
################stationary test

'''Stationarity means that the statistical properties of a time series i.e. mean, variance and covariance do not change over time'''

'''##############Augmented Dickey Fuller (“ADF”) test'''


#By applying ADF techniques to prove know that the data is stationary or not
#determine null and alternate hypothesis


'''# null hypothesis Ho = Time series is non-stationary in nature
#    alternate hypothesis Ha = Time series is stationary in nature'''

#if ADF statistic < Critical value ,then reject the null hypothesis
## if ADF statistic > critical value, then failed to reject the null hypothesis

#checking data is stationary or not by using graph

for column in columns:
    plt.plot(coal[column])
    plt.title(f"Plot of {column}")
    plt.show()
    
    x = coal[column].values
    result = adfuller(x)
    print(f"ADF Statistic for {column}: {result[0]}")
    print(f"p-value: {result[1]}")
    print("Critical values:")
    for key, value in result[4].items():
        print(f"\t{key}: {value:.3f}")
    
    if result[0] < result[4]["5%"]:
        print(f"Reject Ho - {column} is Stationary")
    else:
        print(f"Failed to Reject Ho - {column} is Non-stationary")
    
    print("\n")

#Here the insight is only Price_WTI column is stationary and others are non-stationary.

#calculating correlation
# Drop the date column if it's not needed for correlation analysis
coal_numeric = coal.drop(columns=['Date'])
coal_numeric

# Calculate the correlation matrix
corr_matrix = coal_numeric.corr()

#show in heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Save the DataFrame to a CSV file
coal.to_csv('pre_processed_data.csv', index=False)
import os
os.getcwd()
#perform Auto_EDA

#sweetviz
my_report = sweetviz.analyze([coal,"coal"])
my_report.show_html("report.html")

#D-tale
import dtale
d = dtale.show(coal)
d.open_browser()

#autoviz
from autoviz.AutoViz_Class import AutoViz_Class
av = AutoViz_Class()
a = av.AutoViz(r"C:\Users\ADMIN\OneDrive\Documents\Data science\Project\Data_set.csv",chart_format='html')
import os
os.getcwd()



########################         MODEL BUILDING             #########################


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Convert the 'Date' column to datetime if not already done
coal['Date'] = pd.to_datetime(coal['Date'], dayfirst=True, errors='coerce')#handle errors during conversion.

# Define the date range for training and testing
train_start = pd.Timestamp('2020-02-04')
train_end = pd.Timestamp('2023-12-29')
test_start = pd.Timestamp('2024-01-01')
test_end = pd.Timestamp('2024-06-28')

# Define the columns for forecasting
target_columns = ['Price_WTI', 'Price_Brent_Oil', 'Price_Dubai_Brent_Oil', 'Price_ExxonMobil', 'Price_Shenhua', 
                  'Price_All_Share', 'Price_Mining', 'Price_LNG_Japan_Korea_Marker_PLATTS', 'Price_ZAR_USD', 
                  'Price_Natural_Gas', 'Price_ICE', 'Price_Dutch_TTF', 'Price_Indian_en_exg_rate']

# Function to create sequences for LSTM
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)

# Parameters
n_steps = 12  # Number of past datapoints or time steps include i.e how many data should take for prediction.
epochs = 50  #no of times the data is pass through the model
batch_size = 32

#Batch size is the number of training samples that are fed to the  network at once. 


# Loop through each target column
for target_column in target_columns:
    print(f"Forecasting for {target_column}")

    if target_column not in coal.columns:
        print(f"Column {target_column} not found in the dataset.")
        continue

    # Extract data
    target_data = coal[['Date', target_column]].dropna()
    
    # Split data into training and testing sets using boolean indexing
    #(Boolean indexing is used to filter/reduce data by selecting subsets of the data from a given Pandas DataFrame.
    #The subsets are chosen based on the actual values of the data in the DataFrame and not their row/column labels.)
    train_data = target_data[(target_data['Date'] >= train_start) & (target_data['Date'] <= train_end)][target_column]
    test_data = target_data[(target_data['Date'] >= test_start) & (target_data['Date'] <= test_end)][target_column]

    if train_data.empty or test_data.empty:
        print(f"No data available for {target_column} in the specified date range.")
        continue

    # Scale data
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data.values.reshape(-1, 1))
    test_scaled = scaler.fit_transform(test_data.values.reshape(-1, 1))

    # Create sequences
    X_train, y_train = create_sequences(train_scaled, n_steps)
    X_test, y_test = create_sequences(test_scaled, n_steps)

    # Reshape for LSTM input( in order to make it more manageable and useful. Reshaping data involves transforming the data from one
    #format to another, such as from wide to long or vice versa. This can help to make the data more accessible, easier to analyze, and more informative.)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Define LSTM model
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(n_steps, 1)),#This is an LSTM layer with 50 units (neurons).#The ReLU (Rectified Linear Unit) activation function is applied to the output of the LSTM units, which introduces non-linearity into the model and helps it to learn complex patterns.
        Dense(1)                                               #n_steps: This is the number of time steps the model will look back in the sequence.
#1: This indicates that each time step has a single feature
    ])#Dense(1): This is a fully connected (dense) layer with a single output. It takes the output from the LSTM layer and maps it to a single value, which is useful for predicting the next value in a time series.
    model.compile(optimizer='adam', loss='mse')
#The Adam optimizer is used for updating the model weights during training. 


    # Fit model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)#verbose=0 means no output will be printed during the training process.
#verbose=1 would print a progress bar for each epoch.
#verbose=2 would print one line per epoch with the loss.

    # Make predictions on training data
    y_train_pred = model.predict(X_train, verbose=0)
    y_train_pred = scaler.inverse_transform(y_train_pred)

    # Make predictions on testing data
    y_test_pred = model.predict(X_test, verbose=0)
    y_test_pred = scaler.inverse_transform(y_test_pred)

    # Calculate MAPE for training data
    y_train_inverse = scaler.inverse_transform(y_train.reshape(-1, 1))
    train_mape = mean_absolute_percentage_error(y_train_inverse, y_train_pred)
    print(f'MAPE for {target_column} (Train): {train_mape:.2f}')

    # Calculate MAPE for testing data
    y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))
    test_mape = mean_absolute_percentage_error(y_test_inverse, y_test_pred)
    print(f'MAPE for {target_column} (Test): {test_mape:.2f}')

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(target_data['Date'], target_data[target_column], label='Actual')
    plt.plot(target_data['Date'][(target_data['Date'] >= train_start) & (target_data['Date'] <= train_end)][n_steps:], 
             y_train_pred, label='Train Forecast', color='orange')
    plt.plot(target_data['Date'][(target_data['Date'] >= test_start) & (target_data['Date'] <= test_end)][n_steps:], 
             y_test_pred, label='Test Forecast', color='green')
    plt.legend()
    plt.title(f'Forecast vs Actuals for {target_column}')
    plt.show()

###################              regression model application               ##################



import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.ensemble import StackingRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import joblib
import tensorflow as tf

# Custom MAPE loss function for LSTM
def mape_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs((y_true - y_pred) / tf.clip_by_value(tf.abs(y_true), 1e-8, tf.reduce_max(tf.abs(y_true))))) * 100

# Specify column names
variable_factors = ['Price_WTI', 'Price_Brent_Oil', 'Price_Dubai_Brent_Oil',
                    'Price_ExxonMobil', 'Price_Shenhua', 'Price_All_Share', 
                    'Price_Mining', 'Price_LNG_Japan_Korea_Marker_PLATTS',
                    'Price_ZAR_USD', 'Price_Natural_Gas', 'Price_ICE',
                    'Price_Dutch_TTF','Price_Indian_en_exg_rate']

coal_prices_cols = ['Coal_RB_4800_FOB_London_Close_USD', 'Coal_RB_5500_FOB_London_Close_USD',
                    'Coal_RB_5700_FOB_London_Close_USD', 'Coal_RB_6000_FOB_CurrentWeek_Avg_USD', 
                    'Coal_India_5500_CFR_London_Close_USD']

# Load and preprocess your data
# Assume coal DataFrame is already defined and sorted by 'Date'
# coal = pd.read_csv('your_coal_data.csv')  # Example: Load your data
# coal['Date'] = pd.to_datetime(coal['Date'])

# Sort the data by date to ensure chronological order
coal = coal.sort_values(by='Date')  # Ensure chronological order

# Split the data into training and test sets based on date
train_data = coal[(coal['Date'] >= '2020-04-02') & (coal['Date'] <= '2023-12-29')]
test_data = coal[(coal['Date'] >= '2024-01-01') & (coal['Date'] <= '2024-06-28')]

# Extract input and output data
X_train = train_data[variable_factors].values
Y_train = train_data[coal_prices_cols].values
X_test = test_data[variable_factors].values
Y_test = test_data[coal_prices_cols].values

# Scale the inputs
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, 'minmax_scaler.joblib')

# Reshape data for LSTM input [samples, time steps, features]
X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Define LSTM model with custom MAPE loss function
def build_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', input_shape=input_shape))
    model.add(Dense(len(coal_prices_cols)))  # Output layer size matches coal price columns
    model.compile(optimizer='adam', loss=mape_loss)  # Use custom MAPE loss
    return model

# Train LSTM model
lstm_model = build_lstm((X_train_lstm.shape[1], X_train_lstm.shape[2]))
lstm_model.fit(X_train_lstm, Y_train, epochs=100, batch_size=32, verbose=2)

# Save LSTM model
lstm_model.save('lstm_model.h5')

# Forecast the variable factors
forecast_train = lstm_model.predict(X_train_lstm)
forecast_test = lstm_model.predict(X_test_lstm)

# Function to train and evaluate models using MAPE for each coal price column
def train_and_evaluate_each_target(models, forecast_train, forecast_test, Y_train, Y_test, target_names):
    results = {}
    overall_best_model_name = None
    overall_best_model = None
    overall_best_mape = float('inf')

    # Iterate over each target column
    for idx, target_name in enumerate(target_names):
        print(f'\nEvaluating models for target: {target_name}')
        results[target_name] = {}

        # Get the training and testing data for this specific target
        y_train_target = Y_train[:, idx]
        y_test_target = Y_test[:, idx]

        # Train and evaluate each model
        for name, model in models:
            model.fit(forecast_train, y_train_target)
            train_predictions = model.predict(forecast_train)
            test_predictions = model.predict(forecast_test)
            
            # Calculate MAPE for both train and test data
            train_mape = mean_absolute_percentage_error(y_train_target, train_predictions)
            test_mape = mean_absolute_percentage_error(y_test_target, test_predictions)
            
            results[target_name][name] = {'train_mape': train_mape, 'test_mape': test_mape}
            print(f'{name} Train MAPE for {target_name}: {train_mape:.2f}')
            print(f'{name} Test MAPE for {target_name}: {test_mape:.2f}')

            # Check for the best model across all targets based on average test MAPE
            if test_mape < overall_best_mape:
                overall_best_mape = test_mape
                overall_best_model_name = name
                overall_best_model = model

    return results, (overall_best_model_name, overall_best_model, overall_best_mape)

# Define models
models = [
    ('XGBoost', XGBRegressor()),
    ('GradientBoosting', GradientBoostingRegressor()),
    ('RandomForest', RandomForestRegressor()),
    ('ExtraTrees', ExtraTreesRegressor())
]

# Evaluate models
results, (best_model_name, best_model, best_mape) = train_and_evaluate_each_target(models, forecast_train, forecast_test, Y_train, Y_test, coal_prices_cols)

# Train and evaluate stacking model
stacking_model = StackingRegressor(
    estimators=models,
    final_estimator=GradientBoostingRegressor()
)

# Calculate and print MAPE for the stacking model for each coal price target
for idx, target_name in enumerate(coal_prices_cols):
    stacking_model.fit(forecast_train, Y_train[:, idx])
    stacking_train_predictions = stacking_model.predict(forecast_train)
    stacking_test_predictions = stacking_model.predict(forecast_test)
    
    stacking_train_mape = mean_absolute_percentage_error(Y_train[:, idx], stacking_train_predictions)
    stacking_test_mape = mean_absolute_percentage_error(Y_test[:, idx], stacking_test_predictions)
    
    print(f'Stacking Model Train MAPE for {target_name}: {stacking_train_mape:.2f}')
    print(f'Stacking Model Test MAPE for {target_name}: {stacking_test_mape:.2f}')
    
    # Check if stacking model is the best overall
    if stacking_test_mape < best_mape:
        best_mape = stacking_test_mape
        best_model_name = 'Stacking Model'
        best_model = stacking_model

# Save the best overall model
joblib.dump(best_model, f'best_model_{best_model_name}.joblib')
print(f'\nBest overall model is {best_model_name} with Test MAPE: {best_mape:.2f}')


# Combine train and test data for future use
combined_data = pd.concat([train_data, test_data], axis=0)
combined_data.to_csv('combined_coal_data.csv', index=False)  # Save combined data to a CSV file
print("\nCombined train and test data saved to 'combined_coal_data.csv'.")

import os
os.getcwd()



