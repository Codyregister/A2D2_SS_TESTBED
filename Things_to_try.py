import pandas as pd
from h3 import h3

def transform_data(df, time_window):
    # Convert 'datetime' to datetime object
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Generate a temporal bin for each entry
    df['time_bin'] = df['datetime'].dt.floor(time_window)
    
    # Generate a spatial bin for each entry
    df['spatial_bin'] = df.apply(lambda row: h3.geo_to_h3(row['lat'], row['long'], 9), axis=1) # I'm using a resolution of 9, which gives a hexagon edge length of ~4.85km. Adjust this value as necessary for your use case
    
    # Aggregate data
    df_aggregated = df.groupby(['time_bin', 'spatial_bin']).size().reset_index(name='count')
    
    return df_aggregated

# Usage
# df_transformed = transform_data(df, '1H') # Here, '1H' indicates a 1 hour time window
#This function will transform the input DataFrame by creating temporal and spatial bins, and then aggregating the count of unique device IDs that appear within each bin.
#The geo_to_h3 function converts a set of latitude and longitude coordinates into an H3 index (a spatial bin) with a specified resolution. Higher resolution values result in smaller bins. The resolution can be tweaked as per your need.
#Please adjust the temporal window and h3 resolution to better suit your data and problem requirements. This example uses a temporal window of 1 hour and a h3 resolution of 9.

def rolling_count(df, window_size):
    # Convert 'time_bin' back to datetime format if it's not
    if df['time_bin'].dtype != '<M8[ns]':
        df['time_bin'] = pd.to_datetime(df['time_bin'])
    
    # Ensure the dataframe is sorted by 'time_bin'
    df = df.sort_values('time_bin')
    
    # Create a new column 'rolling_count' which calculates the rolling sum of 'count'
    df['rolling_count'] = df.groupby('spatial_bin')['count'].transform(lambda x: x.rolling(window_size, min_periods=1).sum())
    
    return df

# Usage
# df_rolling = rolling_count(df_transformed, '3H') # Here, '3H' indicates a 3 hour sliding window
#This function will add a new column 'rolling_count' to the dataframe, which calculates the rolling sum of the 'count' column within each spatial bin over a specified sliding window.
#Again, adjust the window size as per your requirements. This example uses a window size of 3 hours.
#Please note that the 'min_periods' parameter is set to 1, which means the rolling sum will output a value as long as there is at least one non-NA observation within the window size. If you would like to change this behavior (for example, to only output a value if there are at least N non-NA observations within the window size), you can adjust this parameter accordingly.

def calculate_baseline(df, baseline_period):
    # Convert 'time_bin' back to datetime format if it's not
    if df['time_bin'].dtype != '<M8[ns]':
        df['time_bin'] = pd.to_datetime(df['time_bin'])
    
    # Calculate the baseline as the mean of the 'count' variable over the baseline period
    df['baseline'] = df.groupby('spatial_bin')['count'].transform(lambda x: x.rolling(baseline_period, min_periods=1).mean())
    
    # Calculate the difference between the current count and the baseline
    df['difference_from_baseline'] = df['count'] - df['baseline']
    
    return df

# Usage
# df_baseline = calculate_baseline(df_transformed, '7D') # Here, '7D' indicates a 7 day baseline period
#This function will add two new columns to the dataframe:
#'baseline': this is the mean of the 'count' column within each spatial bin over a specified baseline period.
#'difference_from_baseline': this is the difference between the current count and the baseline.
#You can adjust the baseline period as per your requirements. This example uses a baseline period of 7 days.
#As for potential improvements:
#Smoothing: The calculated baseline might be sensitive to sudden changes in device activity. To mitigate this, you could apply a smoothing function to the 'count' column before calculating the baseline. One commonly used technique is exponential smoothing.
#Outlier Detection: Extreme values in the 'count' column might influence the baseline calculation. Implementing an outlier detection method could help to prevent this. After identifying outliers, you can treat them according to your needs - either by replacing them with NaNs, or capping them at certain upper and lower limits.
#Adjusting for Trends and Seasonality: If your data has a clear trend or seasonality, you might want to adjust for it before calculating the baseline. Methods like decomposition can help in this case.
#Segmenting Data: Instead of using all the data, you might want to segment it based on other variables (like time of day, day of week, etc.) and calculate the baseline for each segment separately. This might provide more insights if the activity changes significantly based on these factors.

#Below are a few methods you could use, along with Python code snippets:
#1. Statistical Methods: The simplest method might be to assume that your count data is normally distributed, and any data points that fall outside of a certain range are considered anomalies. A common choice is to use the mean plus or minus three standard deviations as a threshold.

# Calculate the mean and standard deviation of the 'count' column
mean = df['count'].mean()
std_dev = df['count'].std()

# Define a threshold for what counts as an anomaly (here, three standard deviations from the mean)
threshold = mean + 3*std_dev

# Create a new column 'anomaly' that is True for any rows where 'count' exceeds the threshold
df['anomaly'] = df['count'] > threshold
#2. Machine Learning Methods: There are many machine learning algorithms that can be used for anomaly detection. One of the most common ones is the Isolation Forest algorithm. The algorithm isolates observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.

from sklearn.ensemble import IsolationForest
# Initialize the model
clf = IsolationForest(contamination=0.01) # contamination is the expected proportion of outliers in the data
# Fit the model and predict anomalies
df['anomaly'] = clf.fit_predict(df[['count']])
#3. Time Series Methods: If you're working with time series data (which seems to be the case), methods like SARIMA or LSTM can also be used to detect anomalies. These methods will fit a model to your data, and then any points where the actual values deviate significantly from the predicted values can be considered anomalies.
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Fit a SARIMA model
model = SARIMAX(df['count'], order=(1, 0, 0), seasonal_order=(1, 1, 0, 12))
model_fit = model.fit(disp=False)
# Predict values
df['prediction'] = model_fit.predict()
# Calculate the residual (the difference between the actual and predicted values)
df['residual'] = df['count'] - df['prediction']
# Any points where the residual is large could be considered an anomaly
threshold = df['residual'].std() * 3
df['anomaly'] = abs(df['residual']) > threshold
#In all of these examples, the threshold for what counts as an anomaly can be adjusted to fit your needs. Note also that these methods will only work if the assumptions they make about your data are valid. If your data is not normally distributed, for example, the statistical method may not work well. Similarly, SARIMAX assumes that your data has a seasonal component, which may or may not be the case.
#The best method for anomaly detection will depend on the specifics of your data and problem, and it may be necessary to try several methods and compare their performance.


#There are several approaches to identify which device ids are significant based on their activity.
#One way could be to establish a threshold for activity level, such that any device id with activity below this threshold over a certain period of time is considered 'noise'. This approach is straightforward and computationally efficient, but the choice of threshold can be somewhat arbitrary and may require experimentation or domain knowledge.
#Here is a Python function that would do this:

def filter_ids(df, threshold):
    # Calculate the total count for each id
    id_counts = df['id'].value_counts()
    
    # Get a list of ids that are above the threshold
    significant_ids = id_counts[id_counts > threshold].index
    
    # Filter the dataframe to only include rows with significant ids
    df_filtered = df[df['id'].isin(significant_ids)]
    
    return df_filtered

# Usage
# df_filtered = filter_ids(df, 10) # Here, 10 is the threshold for minimum count
#Another more sophisticated approach could be to use statistical methods or machine learning to predict which ids will have significant activity in the future based on their past activity and other characteristics. This approach is more complex and computationally intensive, but it could potentially be more accurate and generalizable.
#For instance, you could train a classifier that takes as input features derived from each id's past activity (such as its mean, median, and standard deviation of 'count', the trend of 'count' over time, etc.) and predicts whether this id will have a 'count' above a certain threshold in the future. You could then use this classifier to filter your ids, keeping only the ones that are predicted to be significant.
#Here is a rough example of how you could do this:

from sklearn.ensemble import RandomForestClassifier

# Calculate features for each id
df_features = df.groupby('id').agg({
    'count': ['mean', 'median', 'std', 'last'], # Replace 'count' with your count column name
    'datetime': ['max'], # Replace 'datetime' with your datetime column name
})

# Define your target variable
df_features['target'] = (df_features['count', 'last'] > threshold).astype(int)

# Split your data into features (X) and target (y)
X = df_features.drop('target', axis=1)
y = df_features['target']

# Initialize and train your classifier
clf = RandomForestClassifier()
clf.fit(X, y)

# Predict which ids will be significant
predictions = clf.predict(X)

# Filter the dataframe to only include rows with significant ids
df_filtered = df[df['id'].isin(df_features[predictions == 1].index)]
#Please replace 'count' and 'datetime' with your actual column names, and adjust the features and model parameters according to your problem requirements. Also, remember to split your data into a training set and a test set to evaluate your model's performance before using it to filter ids.

#The PhiK correlation, or "correlation ratio", is a measure of association between two variables. Unlike Pearson's correlation, PhiK can be used for both categorical and numerical variables. This makes it particularly useful for your case, where you are dealing with both time (numerical) and spatial bin (categorical) data.

#Let's install the necessary library if not already done:

!pip install phik
#Then, we will apply this correlation measure to analyze your data:

import phik

def analyze_correlation(df, target_column):
    # Calculate the PhiK correlation matrix
    correlation_matrix = df.phik_matrix(interval_cols=[target_column])
    
    # Extract the column of the correlation matrix corresponding to the target variable
    correlations = correlation_matrix[target_column]
    
    # Remove the self-correlation entry
    correlations = correlations.drop(target_column)
    
    return correlations

# Usage
# correlations = analyze_correlation(df_transformed, 'event_occurred') # Here, 'event_occurred' indicates whether an event occurred or not
#This function will return a Series with the PhiK correlation between the target column and all other columns in the DataFrame.
#The target column should be a binary variable indicating whether an event occurred or not in each temporal or spatial bin.
#You can then inspect these correlation values to see which time windows have the strongest correlation with the occurrence of an event. Note that the PhiK correlation ranges from -1 (perfect inverse correlation) to 1 (perfect direct correlation), with 0 indicating no correlation.
#Keep in mind that the PhiK correlation, while powerful, is not infallible, and correlation does not imply causation. It is always a good idea to further investigate any strong correlations you find and ensure that they make sense in the context of your data and problem.


#Creating lag features is a common practice in time series analysis and they can be very helpful in capturing temporal patterns. Similarly, time till next event and time since last event can provide important context for each observation.
#Here's how you could implement these:
#Lag Features:
#A simple approach to create lag features is to use pandas shift() function which shifts the index of the DataFrame by a certain number of periods.

def create_lag_features(df, lag_periods):
    df = df.copy()
    for i in lag_periods:
        df['lag_'+str(i)] = df['count'].shift(i)
    return df

# Usage
# df_lagged = create_lag_features(df_transformed, [1, 2, 3, 7, 14]) # Here, lags of 1, 2, 3, 7, and 14 time periods are created
#In this function, lag_periods is a list of integers, each representing the number of periods to lag by.

#Time till next event and time since last event:

#Here's how you could create a 'time_till_next_event' and 'time_since_last_event' feature. This code assumes that your dataframe is sorted by time:

def create_event_time_features(df, event_column):
    df = df.copy()

    # Get the time since the last event
    df['time_since_last_event'] = df.loc[df[event_column]==1, 'datetime'].diff()

    # Get the time till the next event
    df['time_till_next_event'] = df.loc[df[event_column]==1, 'datetime'].diff().shift(-1)
    
    # Forward fill the values to propagate them through all rows between events
    df[['time_since_last_event', 'time_till_next_event']] = df[['time_since_last_event', 'time_till_next_event']].ffill()
    
    return df

# Usage
# df_events = create_event_time_features(df_transformed, 'event_occurred') # Here, 'event_occurred' indicates whether an event occurred or not
#In this function, event_column is the name of the column in your dataframe that indicates whether an event occurred or not. This column should be a binary variable where 1 indicates the occurrence of an event and 0 otherwise.
#Note: The resultant features 'time_since_last_event' and 'time_till_next_event' are in terms of difference in the 'datetime' values of the events. If 'datetime' is a timestamp, then the output would be in terms of time duration. If you want to convert it to other units (like minutes or hours), you can do so by using the appropriate conversion method. For example, df['time_since_last_event'].dt.total_seconds() / 60 will convert the time duration to minutes if 'datetime' is a timestamp.

#Mutual Information is a metric that quantifies the mutual dependence between two variables. It measures how much knowing one of these variables reduces uncertainty about the other. It can be particularly useful in this case as it is able to capture any kind of relationship, even if it is non-linear or non-monotonic.
#Here is how you could use the sklearn library to compute Mutual Information for your problem:

from sklearn.feature_selection import mutual_info_regression

def calculate_mutual_info(df, target_column):
    # Extract the features and target variables
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Calculate the Mutual Information
    mutual_info = mutual_info_regression(X, y)
    
    # Create a Series with the Mutual Information values, indexed by the feature names
    mutual_info_series = pd.Series(mutual_info, index=X.columns)
    
    # Sort the series by descending order of Mutual Information
    mutual_info_series.sort_values(ascending=False, inplace=True)
    
    return mutual_info_series

# Usage
# mutual_info = calculate_mutual_info(df_transformed, 'event_occurred') # Here, 'event_occurred' indicates whether an event occurred or not
#This function will return a pandas Series with the Mutual Information between each feature and the target column, sorted by descending order of Mutual Information.
#As with the PhiK correlation, the Mutual Information ranges from 0 (indicating no relationship) to 1 (indicating a perfect relationship). Note, however, that Mutual Information is not symmetric: the Mutual Information between X and Y is not necessarily the same as the Mutual Information between Y and X.
#In terms of interpreting the results, higher Mutual Information values indicate features that are more likely to be useful in predicting the target variable. However, like all feature importance measures, Mutual Information should be taken as a guide rather than an absolute truth: even features with low Mutual Information can contribute to the performance of a model, especially in combination with other features.

A common technique to reduce noise in high-dimensional categorical data like device IDs is to use feature hashing, also known as the "hashing trick". This method applies a hash function to the categories and reduces them into a lower dimensional space.

Feature hashing has a few key benefits:

It reduces the dimensionality of the data, which can make subsequent computations more efficient.
It can handle new categories that weren't seen during training, which makes it robust to changes in the data over time.
Here's how you could use feature hashing in Python using the FeatureHasher class from sklearn:

python
Copy code
from sklearn.feature_extraction import FeatureHasher

def hash_ids(df, n_features):
    # Initialize the hasher
    hasher = FeatureHasher(n_features=n_features, input_type='string')

    # Apply the hasher to the 'id' column
    hashed_features = hasher.transform(df['id'])

    # Convert the hashed features to a DataFrame
    df_hashed = pd.DataFrame(hashed_features.toarray())

    # Concatenate the hashed DataFrame with the original DataFrame, excluding the 'id' column
    df = pd.concat([df.drop(columns=['id']), df_hashed], axis=1)

    return df

# Usage
# df_hashed = hash_ids(df, 10) # Here, the number of hashed features is set to 10
#In this function, n_features is the number of features that the hashed IDs will be reduced to. This should be much smaller than the number of unique IDs in your data, but large enough to avoid too many hash collisions (which occur when different IDs get hashed to the same value).
#A couple of notes on using feature hashing:
#Because of hash collisions, feature hashing can introduce some noise into your data. You may need to experiment with different values of n_features to find a good trade-off between dimensionality reduction and data fidelity.
#The hashed features are not interpretable: they don't carry any intrinsic meaning, and you can't reverse-engineer a hashed ID from its hashed features. If interpretability is important for your project, you might want to use a different method.
#Feature hashing is a stateless operation: it doesn't need to learn anything from the data, so you can apply it independently to different datasets, and it will handle unseen IDs consistently.
#Feature hashing, also known as "the hashing trick", is a method used to convert high-dimensional categorical features into a lower-dimensional numeric format that can be used by machine learning algorithms. This is particularly useful for large datasets with many unique categories or for dealing with new categories that might appear in the future data.

#Here's a basic outline of how it works:
#A hash function is chosen. A hash function takes in input (in this case, a categorical feature like device ID) and outputs a hashed value. The output range is finite and typically much smaller than the input range.
#Each unique category in the feature is fed into the hash function, which outputs a numerical hashed value. Importantly, the same category will always produce the same hashed value.
#The hashed values are used as new features. If desired, multiple hashed features can be created by using different hash functions or by partitioning the output range of a single hash function.
#The original categorical feature can then be replaced by the hashed features in the dataset, reducing its dimensionality.
#The main advantage of feature hashing in this context is its ability to drastically reduce dimensionality. Your device ID data likely contains many unique IDs, each of which would need to be represented by a separate category in a traditional one-hot encoding scheme. This could result in a very large number of features, which can be computationally expensive for machine learning algorithms and can lead to overfitting.
#Feature hashing reduces this to a fixed, manageable number of features, regardless of how many unique categories there are. This makes it much more efficient and scalable.
#Another advantage of feature hashing is its ability to handle new categories. If a new device ID appears in the data that wasn't seen during training, the hash function can still produce a hashed value for it. This makes feature hashing robust to changes in the category distribution over time.
#However, a potential downside of feature hashing is the possibility of hash collisions, where different categories get mapped to the same hashed value. This can introduce some noise into the data. But in practice, with a well-chosen hash function and a sufficiently large output range, the impact of collisions can be minimized.
#Finally, it's important to note that the hashed features are not interpretable: they don't carry any intrinsic meaning about the original categories. If interpretability is important for your application, other methods like target encoding or bin counting might be more appropriate.


#Absolutely, visualizing the features and their relationship with the event occurrence can significantly help in communicating your ideas and findings with stakeholders. Here are some visualizations that you might consider:
#Heatmaps: These are great for showing the correlation between different variables. You could create a heatmap of the PhiK correlation or Mutual Information values between your features and the event occurrence. This can visually demonstrate which features have stronger relationships with the event occurrence.
#Time Series Plots: Visualizing the device counts, lag features, time till next event, and time since last event over time can provide insights into their behaviors and their relationship with the event occurrence. Overlaying these plots with the event occurrences can make it clear how these features change in relation to the events.
#Histograms or Boxplots: These can show the distribution of values for each feature, split by whether an event occurred or not. This can help illustrate how the distributions differ between when events do and don't occur, which can validate the usefulness of these features.
#Scatter plots: These can help visualize the relationship between two features. For example, a scatter plot of device density vs time since last event could reveal clusters or patterns that are indicative of events.
#Maps: If you have geographical data, plotting the device locations on a map and highlighting the locations where events occurred can be a powerful visual tool.

#Here is a code example of how to create a heatmap and a time series plot with matplotlib and seaborn:

import matplotlib.pyplot as plt
import seaborn as sns

def plot_heatmap(df, target_column):
    correlation_matrix = df.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

def plot_time_series(df, target_column):
    plt.figure(figsize=(10,8))
    plt.plot(df['datetime'], df[target_column], label=target_column)
    plt.plot(df['datetime'], df['event_occurred'], label='Event Occurrence')
    plt.title('Time Series Plot')
    plt.xlabel('Datetime')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

# Usage
# plot_heatmap(df_transformed, 'event_occurred')
# plot_time_series(df_transformed, 'device_count')
#Remember to carefully annotate your plots and provide sufficient context so that your stakeholders can understand them. It can also be helpful to provide concrete examples or case studies of how a certain feature helped to predict an event.