# Solana-SOL--Price-Prediction-Model
A full-stack web application that uses LSTM neural networks to predict Solana (SOL) cryptocurrency price movements with ~55-65% accuracy.

# Model-Features
LSTM Neural Network: Deep learning model built with TensorFlow/Keras for time series prediction
Real-time Data: Fetches live SOL price data from CoinGecko API
Web Interface: Flask backend with responsive JavaScript frontend
AWS Deployment: Production-ready deployment on AWS EC2
Daily Updates: Automated model retraining with new market data

# Model-Performance
Accuracy Range: 55-65%
Architecture: LSTM with dropout layers for regularization
Training Period: August 2024 - November 2024
Prediction Window: Next 7 days price movements

# Backend:
Python 3.8+
TensorFlow/Keras
Flask
Pandas, NumPy
Scikit-learn
APScheduler

# Frontend:
HTML5/CSS3
JavaScript (ES6+)
Chart.js for visualizations
Bootstrap for responsive design

# Deployment:
AWS EC2
Gunicorn WSGI server
Nginx reverse proxy
