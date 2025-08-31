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
