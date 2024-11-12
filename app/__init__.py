from flask import Flask
from .routes import setup_routes


def create_app():
    app = Flask(__name__)

    # Настройки из config.py
    app.config.from_pyfile('config.py')

    # Устанавливаем маршруты
    setup_routes(app)

    return app
