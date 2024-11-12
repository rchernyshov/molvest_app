from flask import jsonify, request
from .models.prediction import PredictionModel
from .models.data_loader import DataLoader

# Загружаем и подготавливаем данные
data_loader = DataLoader()
data = data_loader.load_data()

# Создаем экземпляр модели
model = PredictionModel(data)
model.train_model()


def setup_routes(app):
    @app.route("/predict", methods=["GET"])
    def predict():
        date_str = request.args.get("date", None)
        prediction = model.predict(date_str)
        return jsonify(prediction)

    @app.route("/stats", methods=["GET"])
    def stats():
        stats_data = model.get_stats()
        return jsonify(stats_data)