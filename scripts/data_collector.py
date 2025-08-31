"""
Script to collect and store historical price data
Usage: python scripts/data_collector.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils import fetch_price_data
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Collect and save historical price data"""
    logger.info("Fetching historical price data...")
    
    try:
        # Fetch 1 year of data
        data = fetch_price_data(days=365)
        
        # Create data directory if it doesn't exist
        os.makedirs('data/raw', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)
        
        # Save raw data
        raw_path = 'data/raw/sol_price_raw.csv'
        data.to_csv(raw_path)
        logger.info(f"Raw data saved to {raw_path}")
        
        # Process and save
        processed_path = 'data/processed/sol_price_data.csv'
        data.to_csv(processed_path)
        logger.info(f"Processed data saved to {processed_path}")
        
        logger.info(f"Data shape: {data.shape}")
        logger.info(f"Date range: {data.index.min()} to {data.index.max()}")
        logger.info(f"Price range: ${data['price'].min():.2f} - ${data['price'].max():.2f}")
        
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
