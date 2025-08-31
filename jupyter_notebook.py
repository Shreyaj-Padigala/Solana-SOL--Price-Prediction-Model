# notebooks/model_development.ipynb
# SOL Price Prediction Model Development
# This notebook contains the complete model development process

# Cell 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import requests
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

plt.style.use('ggplot')
sns.set_palette("husl")

# Cell 2: Data Collection Function
def fetch_solana_data(days=365):
    """Fetch Solana price data from CoinGecko API"""
    url = f"https://api.coingecko.com/api/v3/coins/solana/market_chart"
    params = {
        'vs_currency': 'usd',
        'days': days,
        'interval': 'hourly'
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        prices = data['prices']
        volumes = data['total_volumes']
        
        df_prices = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df_volumes = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
        
        # Merge price and volume data
        df = pd.merge(df_prices, df_volumes, on='timestamp')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        return df
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# Cell 3: Load and Explore Data
print("Fetching Solana price data...")
df = fetch_solana_data(days=365)

if df is not None:
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nBasic statistics:")
    print(df.describe())
else:
    print("Failed to fetch data")

# Cell 4: Data Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Price over time
axes[0,0].plot(df.index, df['price'], color='purple', alpha=0.8)
axes[0,0].set_title('SOL Price Over Time')
axes[0,0].set_ylabel('Price (USD)')
axes[0,0].tick_params(axis='x', rotation=45)

# Volume over time
axes[0,1].plot(df.index, df['volume'], color='green', alpha=0.8)
axes[0,1].set_title('SOL Trading Volume Over Time')
axes[0,1].set_ylabel('Volume')
axes[0,1].tick_params(axis='x', rotation=45)

# Price distribution
axes[1,0].hist(df['price'], bins=50, alpha=0.7, color='purple')
axes[1,0].set_title('Price Distribution')
axes[1,0].set_xlabel('Price (USD)')
axes[1,0].set_ylabel('Frequency')

# Price vs Volume scatter
axes[1,1].scatter(df['volume'], df['price'], alpha=0.5, color='orange')
axes[1,1].set_title('Price vs Volume')
axes[1,1].set_xlabel('Volume')
axes[1,1].set_ylabel('Price (USD)')

plt.tight_layout()
plt.show()

# Cell 5: Technical Indicators
def calculate_technical_indicators(df):
    """Calculate technical indicators"""
    data = df.copy()
    
    # Simple Moving Averages
    data['ma_7'] = data['price'].rolling(window=7).mean()
    data['ma_20'] = data['price'].rolling(window=20).mean()
    data['ma_50'] = data['price'].rolling(window=50).mean()
    
    # Exponential Moving Averages
    data['ema_12'] = data['price'].ewm(span=12).mean()
    data['ema_26'] = data['price'].ewm(span=26).mean()
    
    # MACD
    data['macd'] = data['ema_12'] - data['ema_26']
    data['macd_signal'] = data['macd'].ewm(span=9).mean()
    
    # RSI
    delta = data['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    data['bb_middle'] = data['price'].rolling(window=20).mean()
    bb_std = data['price'].rolling(window=20).std()
    data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
    data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
    
    # Price change and volatility
    data['price_change'] = data['price'].pct_change()
    data['volatility'] = data['price_change'].rolling(window=20).std()
    
    return data

# Add technical indicators
print("Calculating technical indicators...")
df_with_indicators = calculate_technical_indicators(df)

# Remove NaN values
df_clean = df_with_indicators.dropna()
print(f"Clean data shape: {df_clean.shape}")

# Cell 6: Feature Engineering and Selection
# Select features for the model
feature_columns = [
    'price', 'volume', 'ma_7', 'ma_20', 'ma_50',
    'macd', 'rsi', 'bb_upper', 'bb_lower', 'volatility'
]

# Create feature matrix
features_df = df_clean[feature_columns].copy()

print("Selected features:")
for i, col in enumerate(feature_columns):
    print(f"{i+1}. {col}")

# Correlation matrix
plt.figure(figsize=(12, 8))
correlation_matrix = features_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

# Cell 7: Data Preprocessing for LSTM
def create_sequences(data, sequence_length=60):
    """Create sequences for LSTM training"""
    X, y = [], []
    
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i])
        y.append(data[i, 0])  # Price is the first column (target)
    
    return np.array(X), np.array(y)

# Scale the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(features_df)

# Create sequences
sequence_length = 60
X, y = create_sequences(scaled_data, sequence_length)

print(f"Sequence data shape: X={X.shape}, y={y.shape}")

# Train-test split (80-20)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"Training set: X={X_train.shape}, y={y_train.shape}")
print(f"Test set: X={X_test.shape}, y={y_test.shape}")

# Cell 8: LSTM Model Architecture
def build_lstm_model(input_shape):
    """Build LSTM model"""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        
        LSTM(50, return_sequences=True),
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

# Build model
model = build_lstm_model((X_train.shape[1], X_train.shape[2]))

# Model summary
print("Model Architecture:")
model.summary()

# Cell 9: Model Training
print("Training LSTM model...")

# Callbacks for training
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001
)

# Train the model
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=100,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

print("Training completed!")

# Cell 10: Training History Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Loss plot
ax1.plot(history.history['loss'], label='Training Loss')
ax1.plot(history.history['val_loss'], label='Validation Loss')
ax1.set_title('Model Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()

# MAE plot
ax2.plot(history.history['mae'], label='Training MAE')
ax2.plot(history.history['val_mae'], label='Validation MAE')
ax2.set_title('Model MAE')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('MAE')
ax2.legend()

plt.tight_layout()
plt.show()

# Cell 11: Model Evaluation
# Make predictions
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# Calculate metrics
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
train_mae = mean_absolute_error(y_train, train_pred)
test_mae = mean_absolute_error(y_test, test_pred)

print