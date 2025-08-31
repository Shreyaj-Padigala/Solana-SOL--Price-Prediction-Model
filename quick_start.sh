#!/bin/bash

# Quick start script for SOL Price Prediction
echo "=== SOL Price Prediction Quick Start ==="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed"
    exit 1
fi

print_status "Python 3 found: $(python3 --version)"

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    print_error "pip is not installed"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Install requirements
print_status "Installing requirements..."
pip install -r requirements.txt

# Run startup check
print_status "Running startup check..."
python startup_check.py

# Create necessary directories
print_status "Creating directories..."
mkdir -p data/models data/raw data/processed logs

# Ask if user wants to collect data
echo ""
read -p "Do you want to collect fresh data? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Collecting data..."
    python scripts/data_collector.py
fi

# Ask if user wants to train model
echo ""
read -p "Do you want to train the model? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Training model (this may take several minutes)..."
    python scripts/train_model.py
fi

# Start the application
echo ""
print_status "Starting Flask application..."
print_status "Open your browser to http://localhost:5000"
print_warning "Press Ctrl+C to stop the server"

python run.py
