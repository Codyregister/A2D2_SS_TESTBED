# %%
import pandas as pd

# %%
info_df = pd.read_csv('data.csv')
event_df = pd.read_csv('event.csv')

# %%
info_df.head()

# %%
info_df['ID'].value_counts()

# %%
info_df['ID'].value_counts().describe()

# %%
#Drop all entries where the ID occurs less than the mean number of times
def drop_low_count(df, col_name):
    return df[df.groupby(col_name)[col_name].transform('count')>df[col_name].value_counts().mean()]

# %%
#Sorting into h3 bins
import h3
import numpy as np


info_df['h3_8'] = info_df.apply(lambda row: h3.geo_to_h3(row['Latitude'], row['Longitude'], 8), axis=1)

info_df.head()

# %%
info_df['h3_8'].describe()

# %%
import folium
from folium.plugins import HeatMap

#Create Heatmap
def create_heatmap(df, h3_col, zoom=8, heat_map=None):
    if heat_map is None:
        heat_map = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=zoom)
    hm = HeatMap(list(zip(df['Latitude'], df['Longitude'])), radius=8)
    hm.add_to(heat_map)
    return heat_map
    

# %%
create_heatmap(info_df, 'h3_8')

# %%
info_df.head()

# %%
df

# %%
df = info_df.copy()

# %%
drop_low_count(df, 'ID')

# %%
import pandas as pd
import numpy as np
import dask.dataframe as dd

# Convert 'Datetime' column to datetime data type
df['Datetime'] = pd.to_datetime(df['Datetime'])

# Create PeriodIndex with 1-hour time bins
periods = pd.period_range(start=df['Datetime'].min(), end=df['Datetime'].max(), freq='H')

# Create a MultiIndex from ID, H3 bin, and PeriodIndex
multi_index = pd.MultiIndex.from_product([df['ID'].unique(), df['h3_8'].unique(), periods],
                                         names=['ID', 'H3_bin', 'Period'])

# Create a Dask DataFrame from the original DataFrame
ddf = dd.from_pandas(df, npartitions=20)  # Adjust the number of partitions as per your available resources

# Set the 'Datetime' column as the index
ddf = ddf.set_index('Datetime')

# Create a Dask DataFrame with the MultiIndex
new_ddf = dd.from_pandas(pd.DataFrame(index=multi_index), npartitions=20)  # Adjust the number of partitions as per your available resources

# Iterate over the unique combinations of 'ID' and 'h3_8' and calculate the count
for id_val in df['ID'].unique():
    for h3_val in df['h3_8'].unique():
        filtered_ddf = ddf[(ddf['ID'] == id_val) & (ddf['h3_8'] == h3_val)]
        counts = filtered_ddf.resample('H').size().rename('Value')
        new_ddf = new_ddf.merge(counts, how='left', left_index=True, right_index=True)

# Compute the resulting DataFrame
new_df = new_ddf.compute()


# %%


# %%


# %%


# %%


# %%
event_df.describe()

# %%
#ARIMA Preproccesing

event_df['Datetime'] = pd.to_datetime(event_df['Datetime'])
info_df['Datetime'] = pd.to_datetime(info_df['Datetime'])

info_df['Date'] = info_df['Datetime'].dt.date
data = info_df.groupby('Date').size()

data.head()

# %%
size = int(len(data) * 0.8)
train, test = data[0:size], data[size:len(data)]
history = [x for x in train]
predictions = list()

# %%
from statsmodels.tsa.arima.model import ARIMA

for t in range(len(test)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(int(yhat))
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))

# %%
from sklearn.metrics import mean_squared_error

error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)

# %%
from matplotlib import pyplot

pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()

# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.preprocessing.sequence import TimeseriesGenerator

# %%
info_df = pd.read_csv('data.csv')
event_df = pd.read_csv('event.csv')

# %%
# Resample to get the number of device sightings per day
info_df_daily = info_df.resample('D').size()

# Create a binary time series for event occurrences
event_df_daily = event_df.resample('D').size().clip(upper=1)

# Make sure that both series have the same index
index = info_df_daily.index.union(event_df_daily.index)

info_df_daily = info_df_daily.reindex(index, fill_value=0)
event_df_daily = event_df_daily.reindex(index, fill_value=0)

# Merge the two series
data = pd.concat([info_df_daily, event_df_daily], axis=1)
data.columns = ['device_count', 'event_occurrence']



# %%
data


# %%
# Normalize data to be between 0 and 1
scaler = MinMaxScaler()
data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Define parameters
n_input = 14  # Number of steps to look back
n_features = data.shape[1]  # Number of features

# Split data into train and test sets
train_data = data.iloc[:-12]
test_data = data.iloc[-(12+n_input):]

# Create generators for training and testing
generator = TimeseriesGenerator(train_data.values, train_data['event_occurrence'].values, length=n_input, batch_size=6)
test_generator = TimeseriesGenerator(test_data.values, test_data['event_occurrence'].values, length=n_input, batch_size=1)


# %%
del(model)

# %%
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense

# Define inputs
inputs = Input(shape=(n_input, n_features))

# Define LSTM layer
lstm_out, hidden_state, cell_state = LSTM(200, activation='relu', return_sequences=True, return_state=True)(inputs)

# Define output layer
outputs = Dense(1, activation='sigmoid')(lstm_out)

# Define model
model = Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(generator, epochs=20)


# %%
# Predict probabilities of an event occurrence
predictions = model.predict(test_generator)

# Apply a threshold to get binary predictions
threshold = 0.5
binary_predictions = (predictions > threshold).astype(int)

# Compare binary_predictions with actual occurrences
accuracy = (binary_predictions == test_data['event_occurrence'].values[n_input:]).mean()
print(f'Test Accuracy: {accuracy}')


# %%
from tensorflow.keras.models import Model

# Define a new model that outputs the hidden states of the LSTM layer
hidden_state_model = Model(inputs=model.input, outputs=model.layers[0].output)

# Predict hidden states
hidden_states = hidden_state_model.predict(generator)

# Visualize hidden states for the first example in the batch
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 7))
plt.imshow(hidden_states[0], aspect='auto', cmap='jet')
plt.colorbar()
plt.title('Hidden states')
plt.show()


# %%


# %%
one_hot = pd.get_dummies(info_df['ID'])
info_df_one_hot = pd.concat([info_df, one_hot], axis=1)

info_df_on.head()

# %%
info_df_one_hot.drop(columns=['ID', 'Latitude', 'Longitude'], inplace=True)

# %%
info_df_daily = info_df_one_hot.groupby(info_df_one_hot.index).sum()


# %%



