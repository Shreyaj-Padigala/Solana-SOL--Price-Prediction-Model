"""
Standalone script to train the LSTM model
Usage: python scripts/train_model.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models import PricePredictor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Train the LSTM model"""
    logger.info("Starting model training...")
    
    try:
        predictor = PricePredictor()
        metrics = predictor.train_model()
        
        logger.info("Training completed!")
        logger.info(f"Training RMSE: {metrics['train_rmse']:.4f}")
        logger.info(f"Test RMSE: {metrics['test_rmse']:.4f}")
        logger.info(f"Training MAE: {metrics['train_mae']:.4f}")
        logger.info(f"Test MAE: {metrics['test_mae']:.4f}")
        
        logger.info("Model saved to data/models/")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
