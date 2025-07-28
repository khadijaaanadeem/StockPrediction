import gradio as gr
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

def train_model(ticker='AAPL', look_back=30):  # Reduced lookback for 6 months data
    try:
        print(f"\n{'='*50}")
        print(f"Starting model training for {ticker}")
        print(f"Using 6 months of historical data")
        print(f"Looking back {look_back} days for sequences")
        
        # Get exactly 6 months of data
        df = get_stock_data(ticker, days=180)
        
        if df.empty:
            raise ValueError(f"❌ No data returned for {ticker}. Please check the ticker symbol and try again.")
            
        if len(df) < look_back * 2:  # Need at least 2x lookback period
            raise ValueError(
                f"❌ Not enough historical data for {ticker}. "
                f"Got {len(df)} days, need at least {look_back * 2} days.\n"
                "Try a different ticker or increase the date range."
            )
        
        print(f"\n📊 Sample of the data (last 5 days):")
        print(df[['Open', 'High', 'Low', 'Close', 'Volume']].tail())
        
        # Clean data - remove any rows with NaN values
        df = df.dropna()
        if df.empty:
            raise ValueError("❌ No valid data remaining after removing NaN values. Please try a different ticker or date range.")
            
        close_data = df[['Close']].values
        print(f"\nTraining on {len(close_data)} data points")
        print(f"Data range: {df.index[0].date()} to {df.index[-1].date()}")
        print(f"Price range: ${close_data.min():.2f} - ${close_data.max():.2f}")
        
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_data)
        
        # Create training data
        X, y = [], []
        for i in range(look_back, len(scaled_data)):
            X.append(scaled_data[i-look_back:i, 0])
            y.append(scaled_data[i, 0])
        
        if not X or not y:
            raise ValueError("Not enough data points to create training sequences")
            
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        print(f"Training samples: {len(X)}")
        
        # Build a simpler model that works better with less data
        model = Sequential([
            LSTM(units=32, return_sequences=True, input_shape=(look_back, 1)),
            Dropout(0.1),
            LSTM(units=16, return_sequences=False),
            Dropout(0.1),
            Dense(units=8, activation='relu'),
            Dense(units=1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Use early stopping to prevent overfitting
        from tensorflow.keras.callbacks import EarlyStopping
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        # Train with validation split
        print("Starting model training...")
        history = model.fit(
            X, y, 
            batch_size=8, 
            epochs=30, 
            validation_split=0.2, 
            callbacks=[early_stop],
            verbose=1
        )
        
        # Plot training history
        plt.figure(figsize=(10, 4))
        plt.plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Training History')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()
        
        # Save the model and scaler
        model_filename = f'model_{ticker.lower()}.h5'
        model.save(model_filename)
        
        # Save the scaler parameters
        import joblib
        scaler_filename = f'scaler_{ticker.lower()}.pkl'
        joblib.dump(scaler, scaler_filename)
        
        print(f"Model trained and saved as {model_filename}")
        print(f"Scaler saved as {scaler_filename}")
        print("="*50 + "\n")
        
        return model, scaler
        
    except Exception as e:
        print(f"\nError during model training: {e}")
        import traceback
        traceback.print_exc()
        print("="*50 + "\n")
        return None, None

def load_or_train_model(model_path='model.h5', ticker='AAPL'):
    if os.path.exists(model_path):
        try:
            model = load_model(model_path)
            # Load or create a new scaler (we'll recreate it with current data when predicting)
            scaler = MinMaxScaler(feature_range=(0, 1))
            return model, scaler
        except:
            print("Error loading existing model, training a new one...")
            return train_model(ticker)
    else:
        print("Model not found, training a new one...")
        return train_model(ticker)

def get_stock_data(ticker, days=180):
    try:
        print(f"\n{'='*50}")
        print(f"Fetching 6 months of data for {ticker}...")
        
        # Clean the ticker symbol - remove any existing suffixes first
        ticker = ticker.upper().strip()
        for suffix in ['.NS', '.BO']:
            if ticker.endswith(suffix):
                ticker = ticker[:-len(suffix)]
        
        # Calculate date range for exactly 6 months
        end_date = datetime.today()
        start_date = end_date - timedelta(days=days)
        
        # Try different exchange suffixes if needed
        suffixes = ['', '.NS', '.BO']
        
        for suffix in suffixes:
            current_ticker = ticker + suffix if suffix else ticker
            print(f"Trying ticker: {current_ticker}")
            try:
                # Try with explicit date range
                df = yf.download(
                    current_ticker,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    progress=False
                )
                
                # If we got data, verify it has enough data points
                if not df.empty and len(df) >= 30:  # Need at least 30 days of data
                    print(f"✓ Successfully fetched {len(df)} days of data for {current_ticker}")
                    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
                    print(f"Latest Close: ${df['Close'].iloc[-1]:.2f}")
                    return df
                elif not df.empty:
                    print(f"✗ Not enough data for {current_ticker} (only {len(df)} days)")
                
            except Exception as e:
                error_msg = str(e).split('\n')[0]
                print(f"✗ Error with {current_ticker}: {error_msg}")
                continue
        
        print(f"\n❌ Could not fetch data for {ticker} with any exchange suffix.")
        print("\nCommon ticker formats:")
        print("- US Stocks: 'AAPL', 'MSFT', 'GOOGL'")
        print("- NSE (India): 'RELIANCE.NS', 'TCS.NS'")
        print("- BSE (India): 'TATASTEEL.BO', 'HDFCBANK.BO'")
        print("\nPlease ensure you're using a valid ticker symbol.")
        return pd.DataFrame()
        
    except Exception as e:
        print(f"❌ Unexpected error in get_stock_data: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()
        return pd.DataFrame()

def prepare_data(df, look_back=60):
    close_data = df[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_data)
    
    # Prepare the input data for prediction
    input_data = scaled_data[-look_back:]
    x_input = input_data.reshape((1, look_back, 1))
    
    return x_input, scaler, close_data

def predict_next_days(model, x_input, scaler, days=5):
    predictions = []
    current_batch = x_input.reshape((1, 60, 1))
    
    for _ in range(days):
        current_pred = model.predict(current_batch)[0]
        predictions.append(current_pred[0])
        current_batch = np.append(current_batch[:,1:,:], [[current_pred]], axis=1)
    
    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions)
    return predictions.flatten()

def plot_predictions(actual, predicted, ticker):
    plt.figure(figsize=(10, 6))
    plt.plot(actual, label='Historical Data')
    future_days = range(len(actual), len(actual) + len(predicted))
    plt.plot(future_days, predicted, 'r--', label='Predicted')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Days')
    plt.ylabel('Price ($)')
    plt.legend()
    return plt

def predict_stock(ticker, days_to_predict):
    try:
        if not ticker or not ticker.strip():
            return None, "❌ Please enter a valid stock ticker."
            
        ticker = ticker.strip().upper()
        
        if days_to_predict < 1 or days_to_predict > 14:
            return None, "❌ Please choose between 1-14 days for prediction with 6 months data."
        
        print(f"\n{'='*50}")
        print(f"📈 Starting prediction for {ticker} - {days_to_predict} days")
        
        # Get stock data first to validate ticker
        print("📊 Fetching stock data...")
        df = get_stock_data(ticker)
        
        if df.empty or len(df) < 30:  # Reduced minimum data requirement
            return None, (
                f"❌ Not enough historical data for {ticker}.\n\n"
                "Possible reasons:\n"
                "• The ticker symbol may be incorrect\n"
                "• The stock may be newly listed\n"
                "• There might be no recent trading activity\n\n"
                "Try these examples instead:\n"
                "• US: 'AAPL', 'MSFT', 'GOOGL'\n"
                "• NSE: 'RELIANCE.NS', 'TCS.NS'\n"
                "• BSE: 'TATASTEEL.BO', 'HDFCBANK.BO'"
            )
        
        # Load or train model
        print("🤖 Loading or training model...")
        try:
            model, _ = load_or_train_model(ticker=ticker)
            if model is None:
                return None, "❌ Failed to initialize the prediction model. Please try again later."
        except Exception as e:
            print(f"❌ Model error: {str(e)}")
            return None, "❌ Error initializing the prediction model. Please check the console for details."
        
        # Prepare data for prediction
        print("🔍 Preparing data for prediction...")
        try:
            x_input, scaler, close_data = prepare_data(df)
            if x_input is None or scaler is None or close_data is None:
                return None, "❌ Error preparing data for prediction. Please try a different ticker."
        except Exception as e:
            print(f"❌ Data preparation error: {str(e)}")
            return None, "❌ Error processing the stock data. Please try again."
        
        # Make predictions
        print("🔮 Making predictions...")
        try:
            predictions = predict_next_days(model, x_input, scaler, days=days_to_predict)
            if predictions is None or len(predictions) == 0:
                return None, "❌ Failed to generate predictions. The model couldn't process the data."
        except Exception as e:
            print(f"❌ Prediction error: {str(e)}")
            return None, "❌ Error generating predictions. Please try again later."
        
        # Create plot
        print("🎨 Generating visualization...")
        try:
            plot = plot_predictions(close_data[-60:], predictions, ticker)
        except Exception as e:
            print(f"❌ Plot generation error: {str(e)}")
            return None, "❌ Error generating the prediction chart. The prediction was completed but the visualization failed."
        
        # Prepare prediction text
        last_price = close_data[-1][0] if len(close_data) > 0 else 0
        prediction_text = (
            f"## 📊 {ticker} Stock Prediction 📈\n\n"
            f"**Current Price:** ${last_price:.2f}\n"
            f"**Prediction for Next {days_to_predict} Day{'s' if days_to_predict > 1 else ''}:**\n"
        )
        
        for i, price in enumerate(predictions, 1):
            change = ((price - last_price) / last_price) * 100
            prediction_text += f"Day {i}: ${price:.2f} ({change:+.2f}%)"
            if i < len(predictions):
                prediction_text += "\n"
        
        print("Prediction completed successfully!")
        print('='*50 + '\n')
        
        return plot, prediction_text
        
    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return None, error_msg

def create_interface():
    with gr.Blocks(theme=gr.themes.Soft(), title="AI Stock Predictor") as demo:
        gr.Markdown("""
        # 📈 AI Stock Price Predictor
        
        Predict future stock prices using advanced LSTM neural networks. This tool analyzes historical price patterns 
        to forecast potential price movements for the next 1-14 days.
        
        **Supported Markets:**
        - US Stocks (e.g., AAPL, MSFT, GOOGL)
        - NSE India (e.g., RELIANCE.NS, TCS.NS)
        - BSE India (e.g., TATASTEEL.BO, HDFCBANK.BO)
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                ticker = gr.Textbox(
                    label="Stock Ticker Symbol",
                    placeholder="e.g., AAPL, RELIANCE.NS, TATASTEEL.BO",
                    max_lines=1
                )
                days = gr.Slider(
                    minimum=1,
                    maximum=14,
                    value=5,
                    step=1,
                    label="Prediction Horizon (Days)",
                    info="How many days ahead to predict"
                )
                submit_btn = gr.Button("🔮 Predict", variant="primary")
                
                # Example tickers
                gr.Examples(
                    examples=[
                        ["AAPL", 5],
                        ["MSFT", 7],
                        ["RELIANCE.NS", 3],
                        ["TATASTEEL.BO", 5]
                    ],
                    inputs=[ticker, days],
                    label="Try these examples:",
                    examples_per_page=4
                )
                
            with gr.Column(scale=2):
                gr.Markdown("### 💡 Tips")
                gr.Markdown("""
                - For US stocks, just enter the symbol (e.g., AAPL)
                - For NSE India, add .NS (e.g., RELIANCE.NS)
                - For BSE India, add .BO (e.g., TATASTEEL.BO)
                - More data = better predictions (min 30 days required)
                - Predictions are more reliable for liquid stocks
                """)
        
        with gr.Row():
            plot_out = gr.Plot(label="Price Prediction")
            
        with gr.Row():
            text_out = gr.Markdown()
        
        # Status indicator
        status = gr.Textbox(
            label="Status",
            interactive=False,
            visible=False
        )
        
        # Handle form submission
        submit_btn.click(
            fn=predict_stock,
            inputs=[ticker, days],
            outputs=[plot_out, text_out],
            api_name="predict"
        )
        
        return demo

# Create and launch the interface
iface = create_interface()
if __name__ == "__main__":
    iface.launch()
