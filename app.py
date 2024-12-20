from app import create_app
from flask_cors import CORS

# ngrok firstly-super-chigger.ngrok-free.app 5000

# Запуск Flask-приложения
app = create_app()
CORS(app, origins="*")
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
