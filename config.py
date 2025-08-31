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
