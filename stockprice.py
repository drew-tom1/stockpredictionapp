import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

app = FastAPI()

class StockRequest(BaseModel):
    ticker: str
    start_date: str
    end_date: str

def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Close'].values.reshape(-1, 1)

def create_sequences(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

@app.post('/history')
async def history(request: StockRequest):
    stock_data = get_stock_data(request.ticker, request.start_date, request.end_date)
    stock_index = pd.date_range(start = request.start_date, periods = len(stock_data), freq = 'B')

    stock_data_df = pd.DataFrame(stock_data, columns=[f'{request.ticker} Stock Price'], index=stock_index)

    return stock_data_df.to_dict()


@app.post('/predict')
async def predict(request: StockRequest):
    stock_data = get_stock_data(request.ticker, request.start_date, request.end_date)

    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_data = scaler.fit_transform(stock_data)

    look_back = 60

    X, y = create_sequences(normalized_data, look_back)

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

    return df_chart.to_dict()

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)