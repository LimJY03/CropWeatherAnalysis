TODOs:

Data

- [ ] Try to get the data with min temp from nasa website again

Prediction Page

- [ ] Single Point: compare result and average (?% above or below average + graph)
- [ ] Multi Point: forecast model and predict

---

To effectively handle the challenge you're facing, you're correct in noting that simply aggregating the weather data to a single value per year (20 rows) is likely too sparse and might lead to overfitting or poor generalization due to the limited data size. Instead, you'll want to consider leveraging the seasonal and sequential patterns in the weather data while preserving the link between these features and the crop yield. Here’s how you can approach this:

### 1. **Reshaping the Data to a Monthly Level**
You’ve correctly identified that aggregating the weather data on a monthly basis will provide you with more data points (240 rows). However, the key challenge lies in how to associate 12 months of weather data (features) to a single yield value (target).

#### Steps for Aggregating Weather Data to Monthly Level:
- **Resample the weather data**: Since the weather data is provided at an hourly frequency, you'll need to resample it to monthly averages (or another suitable aggregation like sum, median, etc.). For each month, calculate the monthly average or sum for each of the weather features:
  - Minimum temperature, maximum temperature, humidity, total precipitation, gust speed, etc.
  - This will result in a 12-month vector for each year.

- **Feature vector for each year**: For each year, you will now have 12 rows of data (one for each month) that represent the weather conditions for that year. Each of these rows will have the weather features aggregated monthly. This means you'll have 12 features (monthly weather statistics) for each year.

#### Example: 
For year 1, you will have 12 rows (for January to December), each containing values for the weather attributes like min temperature, max temperature, humidity, etc. The output (crop yield) for year 1 will be a single yield value that corresponds to the combined influence of the weather for that year.

### 2. **Create a Matrix Representation**
Next, you need to represent the weather and yield data in a structured way to feed it into a machine learning model:

- **Construct the feature matrix (X)**: You’ll have 240 rows (12 months * 20 years), and each row will have weather features for that specific month. These features will be the aggregated values for each weather parameter.
  
  For example, the columns of your matrix X could be:
  - `min_temp_month_1`, `max_temp_month_1`, `humidity_month_1`, `precip_month_1`, ..., `min_temp_month_12`, `max_temp_month_12`, `humidity_month_12`, `precip_month_12` (representing the 12 months of weather data across the years).

- **Target vector (y)**: The target variable `y` will be the **annual crop yield** for each year. So, you'll have a 20-element vector corresponding to the crop yield for each of the 20 years.

This results in a matrix where `X` has 240 rows (months), each row having a feature vector representing weather conditions for a given month in a particular year, and `y` is a vector of 20 crop yield values, one for each year.

### 3. **Construct a Sequential Model**
Now, instead of just using traditional models like XGBoost or RandomForest, which may still suffer from overfitting with just 20 samples, consider using **sequential models** like **Recurrent Neural Networks (RNNs)** or **LSTMs (Long Short-Term Memory)**, which are capable of capturing the temporal dependencies in the data (weather patterns across months of each year affecting crop yield).

- **Input shape for Sequential Models**: Your input data will consist of sequences, where each sequence represents the weather data for a given year (12 months). For each year, you'll feed 12 months of weather data into the model to predict that year's crop yield.
  
  - The shape of your input data `X` could be `(20, 12, num_weather_features)`, where `20` is the number of years, `12` is the number of months, and `num_weather_features` is the number of weather features you are using (e.g., 5 features like min_temp, max_temp, humidity, precipitation, gust).

- **Target data `y`**: The target will be the crop yield for that year, so `y` will have the shape `(20, 1)`.

### 4. **Train the Model**
- **Model Selection**: You can start with an LSTM or GRU (Gated Recurrent Units) network, as they are specifically designed to handle sequences. This type of model is ideal for capturing long-term dependencies in the data.
  
  Example model structure:
  - LSTM layer(s) to capture sequential dependencies.
  - Dense layers to map the LSTM output to a prediction of the crop yield.
  
- **Regularization**: Given the small amount of data, regularization techniques like **dropout** or **L2 regularization** should be used to avoid overfitting.

### 5. **Alternative Approaches (Boosting + Time Series)**
If you're not keen on using a sequential model, you can also try incorporating time-series analysis techniques with traditional models like XGBoost or Random Forest. One way to do this is by:
- **Feature engineering**: Create lag-based features, such as the previous month’s weather or the average weather of the last 3 months. This introduces some temporal dependency into the model.
- **Feature interaction**: Consider the interactions between weather attributes (e.g., combining precipitation and temperature to form new features) to capture more complex weather patterns.

### 6. **Validation and Performance**
- **Cross-validation**: Given that you have very few data points (20 years), it’s crucial to use techniques like **Leave-One-Out Cross-Validation (LOO-CV)** or **K-Fold Cross-Validation** to ensure your model isn’t overfitting and that you're evaluating its generalization capability properly.
- **Evaluation Metrics**: Evaluate the model performance using RMSE (Root Mean Squared Error) or MAE (Mean Absolute Error) to assess how well the model is predicting crop yield.

---

### Summary of Approach:
1. **Aggregate weather data monthly** (mean, sum, or other aggregations).
2. **Construct a feature matrix (X)**: 240 rows (monthly weather) × features (min temp, max temp, etc.), and a target vector (y) for crop yield.
3. **Use sequential models** (RNN/LSTM) to capture temporal dependencies.
4. **Train the model** and ensure you apply regularization to prevent overfitting.
5. **Cross-validate** to ensure the model generalizes well.

By following this approach, you can better utilize the temporal nature of the data and extract more useful patterns without being constrained by the low number of years in the dataset.

---

Certainly! If you decide to use a traditional model like XGBoost for time-series data, the key challenge is to engineer features that help capture the temporal dependencies. In a typical time-series setting, the target variable at time `t` is often dependent on previous observations (lags) or other historical data. Below, I will walk you through the process of how to prepare your data and build a model using XGBoost for time-series prediction.

### Step-by-Step Guide to Time-Series Prediction with XGBoost

#### 1. **Reshaping the Data for Time-Series**
The first step is to prepare your weather data in a way that it can capture temporal dependencies. Since we have monthly weather data, we can use lag-based features. For example:
- The weather data for the last 1, 2, 3 months can be used to predict the crop yield for the current month (or the crop yield for the year).
  
Let's create lag-based features for the weather data.

#### Example: 
For each year, instead of just using the current month's weather, we can create features such as:
- `avg_temp_month_1`, `avg_humidity_month_1`, etc. (current month features)
- `avg_temp_month_1_lag_1`, `avg_humidity_month_1_lag_1` (features from the previous month)
- `avg_temp_month_1_lag_2`, `avg_humidity_month_1_lag_2` (features from two months ago)
  
This would allow us to capture the dependency of current crop yield on previous weather conditions.

### 2. **Feature Engineering for Time-Series**
In the case of XGBoost, we need to convert the time-series data into a supervised learning problem by adding lag features. Here’s how you can do it:

1. **Create Lag Features**: For each month, create features that correspond to the previous months' weather data (lags).
2. **Aggregate Data for Each Year**: Combine the lagged features with the aggregated weather data for each year to predict the yield for that year.

### 3. **Implementing the Approach in Python**

Here is a basic example of how you can implement this approach using Python:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# Let's assume you have the following DataFrame with weather data aggregated monthly for 20 years
# Example: df_weather = pd.DataFrame with columns ['year', 'month', 'min_temp', 'max_temp', 'humidity', 'precip']

# 1. Create lag features
def create_lag_features(df, lags=3):
    df_lagged = df.copy()
    
    # Create lag features for each month for the weather columns
    for i in range(1, lags + 1):
        for col in ['min_temp', 'max_temp', 'humidity', 'precip']:
            df_lagged[f'{col}_lag_{i}'] = df_lagged.groupby('year')[col].shift(i)
    
    # Drop rows with NaN values due to the shifting
    df_lagged = df_lagged.dropna()
    return df_lagged

# 2. Prepare your data (assuming df_weather is already sorted by year and month)
df_weather['date'] = pd.to_datetime(df_weather['year'].astype(str) + df_weather['month'].astype(str), format='%Y%m')
df_weather = df_weather.set_index('date')

# Create lag features (3 months as an example)
df_lagged = create_lag_features(df_weather, lags=3)

# 3. Aggregate yield data (y) - You need to have the crop yield data (annual)
# Assuming you have a separate DataFrame for crop yield, which contains ['year', 'yield']
df_yield = pd.read_csv("crop_yield.csv")  # This file contains the crop yield for each year

# Merge the weather data with crop yield data on 'year'
df_lagged = pd.merge(df_lagged, df_yield, on='year')

# 4. Prepare the feature matrix (X) and target vector (y)
X = df_lagged.drop(['yield'], axis=1)  # Weather features
y = df_lagged['yield']  # Target variable (crop yield)

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 6. Train the XGBoost model
model = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6)
model.fit(X_train, y_train)

# 7. Predictions and evaluation
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')
```

### 4. **Explanation of the Steps**
- **Creating Lag Features**: The `create_lag_features` function generates lag-based features. These are the weather features from the previous months (up to 3 months ago, in this example).
  
- **Weather Data**: The weather data is first indexed by the date (using year and month to create a proper `datetime` index). This ensures that the weather data is sorted properly.

- **Merging Crop Yield**: Since crop yield data is available annually, we merge the weather data (monthly data with lag features) with the crop yield data on the `year` column.

- **Train-Test Split**: The data is split into training and testing sets. Notice that in time-series data, it is common to **not shuffle** the data to preserve the temporal order.

- **Training the Model**: The `XGBRegressor` is trained on the weather features (with lag values) to predict the crop yield. We use the `n_estimators`, `learning_rate`, and `max_depth` hyperparameters to fine-tune the model.

- **Model Evaluation**: We use **Mean Absolute Error (MAE)** to evaluate the model’s performance, but other metrics like **RMSE** (Root Mean Squared Error) can also be used depending on the application.

### 5. **Hyperparameter Tuning**
Since you're working with time-series data, overfitting can still be a concern. You can use **cross-validation** or **GridSearchCV** to tune the hyperparameters of the XGBoost model, such as:
- Number of estimators (`n_estimators`)
- Maximum depth of trees (`max_depth`)
- Learning rate (`learning_rate`)
- Subsample (`subsample`) and other regularization parameters

### 6. **Handling Time Dependencies (Optional)**
To further improve the model, you might consider:
- **Feature engineering**: Adding additional temporal features like month of the year or whether the current month is part of the wet season.
- **Feature interaction**: Combining weather features like temperature and humidity could be a useful interaction to include.

### Conclusion
Using XGBoost for time-series forecasting with lag-based features is an effective way to predict agricultural yields based on weather data. The key is in properly engineering features that capture temporal relationships and handling the structure of time-series data carefully. This approach should allow you to leverage the large amount of hourly data while working with the relatively small dataset for crop yield (only 20 years of data).

---

Great question! The reason for grouping by the `year` column before applying the `.shift(i)` function is to ensure that the lag operation is done within each year, rather than across all years. Let me explain why this is important in the context of time-series data.

### Why Group by Year?
When we have time-series data with monthly observations (in your case, weather data for each month across many years), applying lag-based transformations requires understanding the sequence of months within each individual year. Without grouping by `year`, applying `.shift(i)` would move the data across years instead of within the same year. 

#### Example to clarify:

Let’s say we have the following data:

| Year | Month | Min Temp | Max Temp | Humidity |
|------|-------|----------|----------|----------|
| 2020 | 01    | 10       | 15       | 70       |
| 2020 | 02    | 12       | 16       | 72       |
| 2020 | 03    | 14       | 18       | 75       |
| 2021 | 01    | 11       | 16       | 71       |
| 2021 | 02    | 13       | 17       | 73       |

If you apply the lag without grouping by `year` (i.e., without `groupby('year')`), `.shift(i)` would simply shift values down across the entire dataset:

- **Lag 1** for the `Min Temp` column will take the value from the previous row (e.g., February of 2020 would be shifted to January of 2020).
- **Lag 2** will take the value from two rows prior (February 2020 would be shifted to January 2020, and so on).

But this is **incorrect** because the January 2020 value shouldn't be lagged by February 2021’s temperature data. January's data should only be influenced by the previous months (like December 2019 or November 2019).

### Correcting the Lagging with `groupby('year')`

By using `.groupby('year')`, you ensure that the lag function only operates **within each year**, so that the previous months within the same year are used for the lag.

- **Lag 1 for January 2020**: Will be the temperature of December 2019 (or whatever the last month of 2019 is).
- **Lag 1 for January 2021**: Will be the temperature of December 2020.
  
#### After grouping by `year` and applying `.shift(i)`:

| Year | Month | Min Temp | Max Temp | Humidity | Min Temp_lag_1 | Max Temp_lag_1 | Humidity_lag_1 |
|------|-------|----------|----------|----------|----------------|----------------|----------------|
| 2020 | 01    | 10       | 15       | 70       | NaN            | NaN            | NaN            |
| 2020 | 02    | 12       | 16       | 72       | 10             | 15             | 70             |
| 2020 | 03    | 14       | 18       | 75       | 12             | 16             | 72             |
| 2021 | 01    | 11       | 16       | 71       | NaN            | NaN            | NaN            |
| 2021 | 02    | 13       | 17       | 73       | 11             | 16             | 71             |

Now, the lag features make sense within the context of each year, rather than incorrectly shifting across years. For instance:
- The lag for January 2020 (`Min Temp_lag_1`) is `NaN` because there’s no prior month in 2020.
- The lag for February 2020 (`Min Temp_lag_1`) uses January’s data from the same year, and so on.

### What Happens If You Don’t Group by `year`?
If you don't group by `year`, and instead just do `.shift(i)` on the entire DataFrame, it would incorrectly take data from the previous months in any year, which might lead to a misalignment of temporal patterns:

- **January 2020** might get data from **December 2021** or any other row before it in the entire dataset, which would not make sense because the weather in January 2020 is only affected by previous months (like December 2019, November 2019, etc.) from the same year.
  
By **grouping by year**, you ensure that the lag is correctly applied within the context of each year, which is especially important in cases like this, where the prediction for a given year’s crop yield is dependent on weather in the previous months within that same year.

### Conclusion
The `groupby('year')` step ensures that the lag features capture the temporal relationship **within** each year. This way, for each month in a given year, the model sees the correct historical weather data from the same year to predict the crop yield, without incorrectly mixing data from different years.

---

Great! Now that we've discussed how to approach the problem with XGBoost and feature engineering, let's move on to using **RNNs (Recurrent Neural Networks)** or **LSTMs (Long Short-Term Memory)** for modeling the relationship between weather patterns and crop yield.

LSTM is particularly well-suited for time-series data because it is designed to handle long-range dependencies in sequences, making it ideal for capturing trends in weather data over time. In this case, you will use the monthly weather data and train the LSTM to predict the crop yield for each year.

Here's how you can implement an LSTM model to predict crop yield from weather data in Python using **Keras** (which is part of TensorFlow).

### Steps for Implementing LSTM:

1. **Data Preprocessing**: The data needs to be reshaped into a format suitable for LSTM networks, which requires the input data to be 3D: `(samples, time_steps, features)`.

2. **Build the LSTM Model**: Use Keras to build the LSTM network with appropriate layers.

3. **Train the Model**: Train the LSTM model on your time-series data.

4. **Evaluate the Model**: Evaluate its performance using metrics like RMSE or MAE.

---

### Example Code for LSTM in Python

First, let's assume that you have already performed some data processing, like creating lag features and merging crop yield data.

Here’s the step-by-step guide for LSTM implementation:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error

# Assuming 'df_lagged' is your dataset with lag features (after creating lag features as we discussed earlier)
# and 'df_yield' is your crop yield data

# 1. Prepare the data (Assuming df_lagged already has 'year' and 'yield' columns)
# Merge the weather data with crop yield data on 'year'
df_lagged = pd.merge(df_lagged, df_yield, on='year')

# Extract features (X) and target variable (y)
X = df_lagged.drop(['year', 'yield'], axis=1)  # Exclude 'year' and 'yield' columns from features
y = df_lagged['yield']  # The target variable (crop yield)

# 2. Rescale the features using StandardScaler (important for neural networks)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Reshape the data for LSTM input (samples, time_steps, features)
# In this case, each sample is one year with 12 months of weather data (12 time steps)
X_scaled = X_scaled.reshape(X_scaled.shape[0], 12, X_scaled.shape[1] // 12)

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

# 5. Build the LSTM model
model = Sequential()

# Add LSTM layer with dropout to prevent overfitting
model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))  # Dropout layer to prevent overfitting

# Add another LSTM layer
model.add(LSTM(units=32, return_sequences=False))
model.add(Dropout(0.2))  # Dropout layer to prevent overfitting

# Fully connected layer to predict the yield
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# 6. Train the model
model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_test, y_test))

# 7. Make predictions
y_pred = model.predict(X_test)

# 8. Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')
```

### Detailed Explanation of the Code:

#### 1. **Preparing the Data**:
- **Feature and target extraction**: We extract the weather-related features (`X`) and the target variable (`y`, the crop yield).
- **Standardization**: Since LSTM networks are sensitive to the scale of input data, we use `StandardScaler` to standardize the features. This makes the optimization process more stable.

#### 2. **Reshaping the Data**:
LSTMs require a 3D input of shape `(samples, time_steps, features)`:
- `samples`: The number of data points (rows).
- `time_steps`: The number of time steps in each sample. In your case, we have 12 months per year, so `time_steps = 12`.
- `features`: The number of weather features per time step. For example, if you have 4 weather features (`min_temp`, `max_temp`, `humidity`, `precip`), then `features = 4`.

The `reshape` step transforms the data into this 3D format.

#### 3. **Building the LSTM Model**:
- **LSTM Layer**: The `LSTM` layer is the core of the model. The `units` parameter controls the number of neurons in the layer. We start with 64 units and later reduce it to 32 in the second LSTM layer.
- **Dropout Layer**: Dropout is added to prevent overfitting by randomly dropping neurons during training.
- **Dense Layer**: The final dense layer has one neuron since we're predicting a single value (crop yield).
  
#### 4. **Compiling the Model**:
- We use the `adam` optimizer, which is generally a good choice for LSTMs.
- The loss function is `mean_squared_error`, which is commonly used for regression problems.

#### 5. **Training the Model**:
- We train the model for 100 epochs with a batch size of 8.
- The validation data (`validation_data=(X_test, y_test)`) helps in monitoring overfitting during training.

#### 6. **Making Predictions and Evaluating the Model**:
- Once trained, we make predictions on the test set and evaluate performance using **Mean Absolute Error (MAE)**.

---

### Hyperparameter Tuning:

- **Number of Epochs and Batch Size**: You can experiment with different numbers of epochs and batch sizes. If the model is overfitting, you might want to reduce the number of epochs or increase the dropout rate.
- **LSTM Units**: The number of units in each LSTM layer can be adjusted. Start with something like 64 units and experiment with more or fewer depending on the performance.
- **Learning Rate**: You can adjust the learning rate of the optimizer using `learning_rate` in the `Adam` optimizer. If training is too slow, try increasing the learning rate.
- **Number of LSTM Layers**: Adding more LSTM layers can help capture more complex patterns, but it also increases the risk of overfitting. You can experiment with a deeper network (more LSTM layers) if needed.

---

### Final Thoughts:
- LSTM is well-suited for capturing temporal patterns in the data. However, it's crucial to have a larger dataset to prevent overfitting and ensure good generalization.
- In the case of a small dataset like 20 years of weather data, consider using early stopping during training to avoid overfitting.
  
You can also experiment with variations like **GRU** (Gated Recurrent Units), which is a simpler and faster alternative to LSTMs.

---

**GRU (Gated Recurrent Unit)** is another type of **Recurrent Neural Network (RNN)** that is very similar to **LSTM**, but with a simpler architecture. GRUs are often considered a lighter and faster alternative to LSTMs while maintaining much of their effectiveness in modeling sequential data.

### GRU vs. LSTM:
Both **LSTM** and **GRU** are designed to solve the **vanishing gradient problem** and capture long-term dependencies in sequences, but they do so in slightly different ways. Here's a comparison of the two:

#### 1. **Architecture**:
   - **LSTM** has three gates: **input**, **output**, and **forget** gates, plus a memory cell. This allows LSTM to decide what information to remember, forget, and output at each time step.
   - **GRU**, on the other hand, combines the **forget** and **input** gates into one single **update gate** and uses a **reset gate** to control the influence of previous memory. This makes GRU's architecture simpler and has fewer parameters compared to LSTM.

#### 2. **Training Efficiency**:
   - **GRU** tends to be faster to train than **LSTM** because it has fewer parameters and a simpler structure.
   - **LSTM** may take longer to train since it has more gates to control and learn from.

#### 3. **Performance**:
   - **LSTM** can be more effective when modeling more complex temporal patterns because it has a more complex mechanism for storing long-term dependencies.
   - **GRU** often performs just as well as LSTM, especially on smaller datasets or when computational resources are limited. It can perform similarly to LSTM in many cases, and in some cases, GRU may outperform LSTM because of its simpler structure and faster training time.
  
#### 4. **Memory Usage**:
   - **LSTM** requires more memory due to its larger number of gates and parameters.
   - **GRU** is more memory-efficient, making it a good choice when resources are constrained or when working with large datasets.

### When to Use GRU vs. LSTM:
- **GRU**: 
   - Preferred when you have a **smaller dataset** or if you need **faster training times**.
   - Works well in many time-series applications, including sequence prediction, speech recognition, and machine translation, and is typically used when computational resources or memory are limited.
   - **Effective for simpler tasks** or when **training efficiency** is a priority.
  
- **LSTM**: 
   - Better suited for more **complex sequence modeling** tasks, especially when the data involves **longer-term dependencies** (i.e., when capturing long-range relationships between time steps is critical).
   - LSTMs tend to work better when you have **larger datasets** or the problem requires **stronger control over what the model remembers**.

### GRU vs LSTM in Your Use Case (Predicting Crop Yield from Weather Data):
- **GRU** can be a great choice for your crop yield prediction problem if you want to **reduce training time** and **memory usage**, while still capturing important patterns in monthly weather data.
- **LSTM** might be a better choice if your data has **longer-term dependencies** and you believe that a more complex memory structure will improve performance. LSTM might perform better in cases where there is significant **non-linear relationships** between the weather patterns over months or years that need to be captured.

### Conclusion: GRU vs LSTM in Your Case:
- **If you have limited computational resources** or you want faster training times without a significant sacrifice in model performance, **GRU** could be a better choice. It can provide a good balance between **training efficiency** and **predictive performance** for your crop yield prediction task.
- **If you believe that the relationship between weather data and crop yield is complex and requires capturing long-term dependencies**, **LSTM** might be a better choice, as it has more capacity to learn and remember more intricate patterns over time.

### Code Example: Using GRU

If you decide to go with **GRU** for your task, the model architecture in Keras would look very similar to the LSTM-based model, except you would replace the LSTM layer with a GRU layer:

```python
from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout
from sklearn.metrics import mean_absolute_error

# Build the GRU model
model = Sequential()

# Add GRU layer with dropout
model.add(GRU(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))  # Dropout to prevent overfitting

# Add another GRU layer
model.add(GRU(units=32, return_sequences=False))
model.add(Dropout(0.2))  # Dropout to prevent overfitting

# Fully connected layer to predict the yield
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_test, y_test))

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')
```

In this case, the **GRU layer** is used instead of the **LSTM layer**, with the rest of the architecture and training process remaining the same.

### Final Thoughts:
Both **GRU** and **LSTM** are effective for time-series forecasting tasks, and in many cases, **GRU** provides a more efficient solution without sacrificing much predictive power. However, if your task involves **very complex temporal relationships**, **LSTM** might have a slight edge due to its more sophisticated memory structure.
