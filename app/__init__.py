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
