# %%
import numpy as np
import pandas as pd
import random 
import string
import datetime as dt   


# %%
start_date = dt.date(2023,1,1)
end_date = dt.date(2023,3,31)

# %%
periods = pd.date_range(start_date, end_date, freq='1H')

# %%
num_identifiers = 1000
identifiers = [''.join(random.choices(string.ascii_uppercase + string.digits, k=10)) for _ in range(num_identifiers)]

# %%
data = []
event_counter = 0
event_target = np.zeros(len(periods))

# %%
event_hours = random.sample(range(len(periods)), 50)


# %%
for i, period in enumerate(periods):
    # Randomly choose the number of events in each hour
    num_events = np.random.randint(0, 3)  # You can adjust the range as per your requirements
    
    # Randomly select 'num_events' number of unique identifiers for this hour
    identifiers_observed = random.sample(identifiers, num_events)
    
    # Generate random observation counts for each identifier (between 0 and 10)
    for identifier in identifiers:
        count = np.random.randint(0, 11)
        row_data[identifier] = count
        if count > 0:
            identifiers_observed.extend([identifier] * count)
    
    # Create the row data for this hour
    row_data = {
        'Period': period,
        'EventOccurred': 1 if i in event_hours else 0,
    }
    
    # Set event occurrence to 1 for about 50 random hours
    if i in event_hours:
        # Randomly choose some identifiers to be somewhat correlated with the event
        correlated_identifiers = random.sample(identifiers, np.random.randint(1, 4))  # Adjust the range as needed
        identifiers_observed.extend(correlated_identifiers)
    
    # Count the occurrences of each identifier for this hour
    for identifier in identifiers:
        row_data[identifier] = identifiers_observed.count(identifier)
    
    data.append(row_data)

# %%
future_df = pd.DataFrame(data)

# %%
future_df.describe()

# %%
future_df.to_csv('future_data.csv', index=False)

# %%
backup = df.copy()

# %%
df = backup.copy()

# %%
# Step 2: Sort the DataFrame based on 'Period' column in descending order
df.sort_values(by='Period', ascending=False, inplace=True)

# Step 3: Calculate time_till until the next 'EventOccurred' equals 1
time_till_list = []
last_event_time = None

for index, row in df.iterrows():
    if row['EventOccurred'] == 1:
        last_event_time = row['Period']
        time_till_list.append(0)
    elif last_event_time is not None:
        time_till = (last_event_time - row['Period']).total_seconds() / 3600  # Convert to hours
        time_till_list.append(time_till)
    else:
        time_till_list.append(None)

# Reverse the time_till list to match the original DataFrame order
df['time_till'] = time_till_list

df.sort_values(by='Period', ascending=True, inplace=True)


# Display the resulting DataFrame
print(df)

# %%
df.to_csv('data.csv', index=False)

# %%
from sklearn.ensemble import RandomForestRegressor


# %%
X = df.drop(['Period', 'EventOccurred', 'time_till_next_event', 'time_till'], axis=1)
y = df['time_till']

# %%
n_train_rows = int(df.shape[0] * 0.7)


# %%
# Split the data
X_train = X[:n_train_rows]
y_train = y[:n_train_rows]
X_test = X[n_train_rows:]
y_test = y[n_train_rows:]

# %%
# Fit a model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# %%
feature_importances = rf.feature_importances_
feature_importances_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
feature_importances_df.sort_values('Importance', ascending=False, inplace=True)
print(feature_importances_df)


# %% [markdown]
# Lasso Regression 

# %%

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
# Define the model
lasso = Lasso(alpha=0.5)  # You may need to adjust the alpha parameter depending on your data

# Fit the model
lasso.fit(X_train_scaled, y_train)

# %%

# Get the feature importances
# Note that feature_importances will be a boolean array
feature_importances = np.abs(lasso.coef_) > 0.0

# Get the important features
important_features = X.columns[feature_importances]

# Print the important features
print(important_features)

# %% [markdown]
# Using AutoTS

# %%
df = backup.copy()
df

# %%
from autots import AutoTS


# %%
model = AutoTS(
    forecast_length=168,
    ensemble = 'simple',
    drop_data_older_than_periods=200,
    frequency='H',
    max_generations=10,
    num_validations=2,
    validation_method='backwards',
    model_interrupt=True,
    verbose=3,
)


# %%
df = backup.copy()

# %%
df['time_till'] = y

# %%
type(df['Period'][0])

# %%
df['Period'] = pd.to_datetime(df['Period'])

# %%
model.fit(df, date_col='Period', value_col='time_till')

# %% [markdown]
# So the way that AutoTS works is it will generate a forecasts for the next forecast_length. Adding new data and making new predictions requires retraining the model. Training longer forecasts windows takes longer, but you can access closer time periods with predictions[0] and so on. The drop periods operator determines how many past periods should be used to train. 

# %%


