# Import Libraries
import pandas as pd
import numpy as np
from nsepy import get_history
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from gnews import GNews

# Initialize Sentiment Analyzer and GNews API
analyzer = SentimentIntensityAnalyzer()
gnews = GNews(language="en", country="IN", max_results=100)

# Fetch Indian Stock Data
def fetch_stock_data(ticker, start_date, end_date):
    start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
    end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
    stock_data = get_history(symbol=ticker, start=start_date, end=end_date)

    if stock_data.empty:
        raise ValueError(f"No stock data available for {ticker} from {start_date} to {end_date}.")

    print("Stock Data Fetched Successfully:")
    print(stock_data.head())
    return stock_data[['Close', 'Volume']]

# Fetch News Sentiment for Indian Stocks
def fetch_news_sentiment(company_name):
    news = gnews.get_news(company_name)
    sentiments = [analyzer.polarity_scores(article['title'])['compound'] for article in news]
    avg_sentiment = np.mean(sentiments) if sentiments else 0
    print(f"Fetched Sentiment Score: {avg_sentiment}")
    return avg_sentiment

# Preprocess Stock Data
def preprocess_data(stock_data, sentiment_score):
    if stock_data.empty:
        raise ValueError("Stock data is empty. Ensure valid ticker and date range.")
    
    stock_data['Sentiment'] = sentiment_score

    if stock_data[['Close', 'Volume', 'Sentiment']].isnull().any().any():
        raise ValueError("Data contains missing values. Check stock and sentiment data.")

    print("Data before scaling:")
    print(stock_data[['Close', 'Volume', 'Sentiment']].head())
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data[['Close', 'Volume', 'Sentiment']])
    return scaled_data, scaler

# Prepare Data for LSTM
def prepare_lstm_data(data, look_back=60):
    X, y = [], []
    for i in range(look_back, len(data) - 1):
        X.append(data[i-look_back:i])
        y.append(data[i, 0])  # Predict 'Close' price
    return np.array(X), np.array(y)

# Build LSTM Model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))  # Output Layer
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Plot Results
def plot_results(actual, predicted):
    plt.figure(figsize=(10, 5))
    plt.plot(actual, label="Actual Prices")
    plt.plot(predicted, label="Predicted Prices")
    plt.title("Indian Stock Price Prediction with LSTM")
    plt.xlabel("Time")
    plt.ylabel("Stock Price (INR)")
    plt.legend()
    plt.show()

# Main Function
if __name__ == "__main__":
    # Parameters
    TICKER = "RELIANCE"  # NSE ticker (e.g., TCS, INFY, TATASTEEL)
    COMPANY_NAME = "Reliance Industries"  # Company Name
    END_DATE = datetime.now()
    START_DATE = END_DATE - timedelta(days=365)  # Last 1 year of data
    LOOK_BACK = 60  # Days to look back for LSTM

    # Fetch stock data
    try:
        print("Fetching stock data...")
        stock_data = fetch_stock_data(TICKER, START_DATE.strftime('%Y-%m-%d'), END_DATE.strftime('%Y-%m-%d'))
    except ValueError as e:
        print(f"Error fetching stock data: {e}")
        exit()

    # Fetch news sentiment
    print("Fetching news sentiment...")
    sentiment_score = fetch_news_sentiment(COMPANY_NAME)

    # Preprocess data
    print("Preprocessing data...")
    try:
        scaled_data, scaler = preprocess_data(stock_data, sentiment_score)
    except ValueError as e:
        print(f"Error preprocessing data: {e}")
        exit()

    # Prepare LSTM data
    X, y = prepare_lstm_data(scaled_data, look_back=LOOK_BACK)
    X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))  # Reshape for LSTM

    # Split data into train and test sets
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build and train model
    print("Building and training LSTM model...")
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_test, y_test))

    # Predict
    print("Making predictions...")
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(np.hstack((predictions, np.zeros((predictions.shape[0], 2)))))[:, 0]
    actual = scaler.inverse_transform(np.hstack((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], 2)))))[:, 0]

    # Plot results
    print("Plotting results...")
    plot_results(actual, predictions)
