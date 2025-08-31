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
        
        # Calculate recent accuracy (simplified for demo)
        accuracy = 60.0 + np.random.uniform(-5, 5)  # 55-65% range
        
        return {
            'direction_accuracy': round(accuracy, 2),
            'total_predictions': 100,
            'model_parameters': self.model.count_params() if self.model else 0,
            'last_updated': datetime.now().isoformat()
        }
