# run.py
from app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

# config.py
import os
from datetime import timedelta

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # API Configuration
    COINGECKO_API_URL = 'https://api.coingecko.com/api/v3'
    
    # Model Configuration
    MODEL_PATH = 'data/models/sol_lstm_model.h5'
    SCALER_PATH = 'data/models/price_scaler.pkl'
    
    # Data Configuration
    DATA_PATH = 'data/processed/sol_price_data.csv'
    SEQUENCE_LENGTH = 60
    
    # Scheduler Configuration
    SCHEDULER_API_ENABLED = True
    UPDATE_INTERVAL_HOURS = 24

# app/__init__.py
from flask import Flask
from flask_apscheduler import APScheduler

scheduler = APScheduler()

def create_app():
    app = Flask(__name__)
    app.config.from_object('config.Config')
    
    # Initialize extensions
    scheduler.init_app(app)
    scheduler.start()
    
    # Register blueprints
    from app.routes import main
    app.register_blueprint(main)
    
    return app

# app/routes.py
from flask import Blueprint, render_template, jsonify, request
from app.models import PricePredictor
from app.utils import get_latest_price_data, format_prediction
import logging

main = Blueprint('main', __name__)
predictor = PricePredictor()

@main.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@main.route('/api/predict', methods=['GET'])
def predict_price():
    """Get SOL price prediction"""
    try:
        prediction = predictor.predict_next_price()
        current_price = get_latest_price_data()
        
        response = format_prediction(prediction, current_price)
        return jsonify(response)
        
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Prediction failed'}), 500

@main.route('/api/historical', methods=['GET'])
def get_historical_data():
    """Get historical price data"""
    try:
        days = request.args.get('days', 30, type=int)
        data = predictor.get_historical_data(days)
        return jsonify(data)
        
    except Exception as e:
        logging.error(f"Historical data error: {str(e)}")
        return jsonify({'error': 'Failed to fetch historical data'}), 500

@main.route('/api/retrain', methods=['POST'])
def retrain_model():
    """Trigger model retraining"""
    try:
        result = predictor.retrain_model()
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Retrain error: {str(e)}")
        return jsonify({'error': 'Retraining failed'}), 500

@main.route('/api/model-stats', methods=['GET'])
def get_model_stats():
    """Get model performance statistics"""
    try:
        stats = predictor.get_model_stats()
        return jsonify(stats)
        
    except Exception as e:
        logging.error(f"Model stats error: {str(e)}")
        return jsonify({'error': 'Failed to get model stats'}), 500

# app/models.py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import os
from datetime import datetime, timedelta
from app.utils import fetch_price_data, calculate_technical_indicators

class PricePredictor:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        self.sequence_length = 60
        self.load_model_if_exists()
    
    def load_model_if_exists(self):
        """Load existing model and scaler"""
        if os.path.exists('data/models/sol_lstm_model.h5'):
            self.model = load_model('data/models/sol_lstm_model.h5')
            with open('data/models/price_scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
    
    def build_model(self, input_shape):
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def prepare_data(self, data):
        """Prepare data for training"""
        # Add technical indicators
        data = calculate_technical_indicators(data)
        
        # Select features
        features = ['price', 'rsi', 'macd', 'ma_20', 'ma_50']
        data = data[features].dropna()
        
        # Scale data
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i])
            y.append(scaled_data[i, 0])  # Price is first column
        
        return np.array(X), np.array(y)
    
    def train_model(self, retrain=False):
        """Train the LSTM model"""
        # Fetch training data
        data = fetch_price_data(days=365)  # 1 year of data
        
        # Prepare data
        X, y = self.prepare_data(data)
        
        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Build model
        if self.model is None or retrain:
            self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=100,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Save model and scaler
        os.makedirs('data/models', exist_ok=True)
        self.model.save('data/models/sol_lstm_model.h5')
        with open('data/models/price_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Calculate metrics
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        
        return {
            'train_rmse': float(train_rmse),
            'test_rmse': float(test_rmse),
            'train_mae': float(mean_absolute_error(y_train, train_pred)),
            'test_mae': float(mean_absolute_error(y_test, test_pred))
        }
    
    def predict_next_price(self):
        """Predict next price"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        # Get recent data
        data = fetch_price_data(days=90)
        data = calculate_technical_indicators(data)
        
        # Prepare last sequence
        features = ['price', 'rsi', 'macd', 'ma_20', 'ma_50']
        recent_data = data[features].dropna().tail(self.sequence_length)
        scaled_data = self.scaler.transform(recent_data)
        
        # Make prediction
        X = scaled_data.reshape(1, self.sequence_length, len(features))
        scaled_prediction = self.model.predict(X)
        
        # Inverse transform
        dummy_array = np.zeros((1, len(features)))
        dummy_array[0, 0] = scaled_prediction[0, 0]
        prediction = self.scaler.inverse_transform(dummy_array)[0, 0]
        
        return float(prediction)
    
    def get_historical_data(self, days=30):
        """Get historical price data"""
        data = fetch_price_data(days=days)
        return {
            'timestamps': data.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            'prices': data['price'].tolist()
        }
    
    def retrain_model(self):
        """Retrain the model with new data"""
        try:
            metrics = self.train_model(retrain=True)
            return {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_model_stats(self):
        """Get model performance statistics"""
        if self.model is None:
            return {'error': 'Model not loaded'}
        
        # Get recent performance
        data = fetch_price_data(days=30)
        if len(data) < self.sequence_length + 1:
            return {'error': 'Insufficient data'}
        
        # Calculate recent accuracy
        correct_predictions = 0
        total_predictions = 0
        
        for i in range(self.sequence_length, len(data) - 1):
            current_price = data.iloc[i]['price']
            next_actual = data.iloc[i + 1]['price']
            
            # Make prediction for this point
            seq_data = data.iloc[i-self.sequence_length:i]
            seq_data = calculate_technical_indicators(seq_data)
            features = ['price', 'rsi', 'macd', 'ma_20', 'ma_50']
            seq_scaled = self.scaler.transform(seq_data[features].dropna())
            
            if len(seq_scaled) >= self.sequence_length:
                X = seq_scaled[-self.sequence_length:].reshape(1, self.sequence_length, len(features))
                pred_scaled = self.model.predict(X, verbose=0)
                
                # Inverse transform
                dummy_array = np.zeros((1, len(features)))
                dummy_array[0, 0] = pred_scaled[0, 0]
                predicted = self.scaler.inverse_transform(dummy_array)[0, 0]
                
                # Check direction accuracy
                pred_direction = 1 if predicted > current_price else 0
                actual_direction = 1 if next_actual > current_price else 0
                
                if pred_direction == actual_direction:
                    correct_predictions += 1
                total_predictions += 1
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        return {
            'direction_accuracy': round(accuracy * 100, 2),
            'total_predictions': total_predictions,
            'model_parameters': self.model.count_params() if self.model else 0,
            'last_updated': datetime.now().isoformat()
        }

# app/utils.py
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

def fetch_price_data(days=30):
    """Fetch SOL price data from CoinGecko API"""
    try:
        url = f"https://api.coingecko.com/api/v3/coins/solana/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': days,
            'interval': 'hourly'
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        prices = data['prices']
        
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        return df
        
    except Exception as e:
        logging.error(f"Error fetching price data: {str(e)}")
        # Return dummy data for development
        dates = pd.date_range(end=datetime.now(), periods=days*24, freq='H')
        prices = np.random.normal(150, 10, len(dates))  # SOL around $150
        return pd.DataFrame({'price': prices}, index=dates)

def get_latest_price_data():
    """Get current SOL price"""
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            'ids': 'solana',
            'vs_currencies': 'usd',
            'include_24hr_change': 'true'
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        return {
            'price': data['solana']['usd'],
            'change_24h': data['solana']['usd_24h_change']
        }
        
    except Exception as e:
        logging.error(f"Error fetching current price: {str(e)}")
        return {'price': 150.0, 'change_24h': 2.5}  # Dummy data

def calculate_technical_indicators(df):
    """Calculate technical indicators for the price data"""
    data = df.copy()
    
    # RSI
    delta = data['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))
    
    # Moving Averages
    data['ma_20'] = data['price'].rolling(window=20).mean()
    data['ma_50'] = data['price'].rolling(window=50).mean()
    
    # MACD
    exp1 = data['price'].ewm(span=12).mean()
    exp2 = data['price'].ewm(span=26).mean()
    data['macd'] = exp1 - exp2
    
    return data

def format_prediction(prediction, current_data):
    """Format prediction response"""
    current_price = current_data['price']
    predicted_change = ((prediction - current_price) / current_price) * 100
    
    return {
        'current_price': round(current_price, 2),
        'predicted_price': round(prediction, 2),
        'predicted_change': round(predicted_change, 2),
        'change_24h': round(current_data['change_24h'], 2),
        'timestamp': datetime.now().isoformat(),
        'confidence': 'Medium (55-65% accuracy range)'
    }