import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from datetime import datetime, timedelta

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

@app.post('/history') # retrieve historical data and display
async def history(request: StockRequest):
    stock_data = get_stock_data(request.ticker, request.start_date, request.end_date)
    stock_index = pd.date_range(start = request.start_date, periods = len(stock_data), freq = 'B')

    stock_data_df = pd.DataFrame(stock_data, columns=[f'{request.ticker} Stock Price'], index=stock_index)

    return stock_data_df.to_dict()


@app.post('/predict')
async def predict(request: StockRequest):
    stock_data = get_stock_data(request.ticker, '2010-01-01', request.end_date)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_data = scaler.fit_transform(stock_data)
    
    look_back = 100
    X, y = create_sequences(normalized_data, look_back)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=30, batch_size=32)

    predictions = []
    last_sequence = X[-1]

    for _ in range(5):  # Predict the next 5 business days
        prediction = model.predict(last_sequence.reshape(1, look_back, 1))
        predictions.append(prediction[0, 0])
        
        last_sequence = np.append(last_sequence[1:], prediction)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    last_date = datetime.strptime(request.end_date, '%Y-%m-%d')
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=5, freq='B')

    df_chart = pd.DataFrame({'Predicted': predictions.flatten()}, index=future_dates)

    return df_chart.to_dict()

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)