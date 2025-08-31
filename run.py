from app import create_app
import os

app = create_app()

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('data/models', exist_ok=True)
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5000)
