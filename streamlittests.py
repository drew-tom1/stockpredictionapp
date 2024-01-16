import streamlit as st
import stockprice
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
import pandas as pd

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('Stock Price Prediction App')

st.sidebar.header('Input Parameters')

ticker = st.sidebar.text_input('Enter Stock Ticker (e.g., AAPL)', 'AAPL')
start_date = st.sidebar.text_input('Enter Start Date (YYYY-MM-DD)', '2022-01-01')
end_date = st.sidebar.text_input('Enter End Date (YYYY-MM-DD)', '2023-01-01')

stock_data = stockprice.get_stock_data(ticker, start_date, end_date)

stock_data_index = pd.date_range(start=start_date, periods=len(stock_data), freq='B')
stock_data_df = pd.DataFrame(stock_data, columns=[f'{ticker} Stock Price'], index=stock_data_index)

st.subheader(ticker + ' Stock Price Over Time')
st.line_chart(stock_data_df)

scaler = MinMaxScaler(feature_range=(0, 1))
normalized_data = scaler.fit_transform(stock_data)

look_back = 60

X, y = stockprice.create_sequences(normalized_data, look_back)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=10, batch_size=32)

predictions = model.predict(X)
predictions = scaler.inverse_transform(predictions)

time_range = pd.date_range(start=start_date, periods=len(stock_data) - look_back + len(y_test), freq='B')

stock_data = stock_data[:len(predictions)]
time_range = time_range[:len(predictions)]

df_chart = pd.DataFrame({'Actual': stock_data.flatten(), 'Predicted': predictions.flatten()}, index=time_range)

st.subheader('Original vs. Predicted Stock Prices')
st.line_chart(df_chart)
