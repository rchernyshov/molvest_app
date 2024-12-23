from flask import jsonify, request
from .model.prediction import PredictionModel
from .model.data_loader import DataLoader

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

    # Новый маршрут для получения статистики
    @app.route("/data/summary", methods=["GET"])
    def summary_statistics():
        stats = model.get_summary_statistics()
        return jsonify(stats)

    @app.route("/predict/next7days", methods=["GET"])
    def predict_next_7_days():
        predictions = model.predict_next_7_days()
        return jsonify(predictions)

    @app.route("/predict/monthly", methods=["GET"])
    def predict_monthly():
        predictions = model.predict_monthly()
        return jsonify(predictions)

    @app.route("/data/raw", methods=["GET"])
    def get_raw_data():

        # Получаем данные в их исходном виде, без обработки
        raw_data = data_loader.load_data()

        # Преобразуем данные в формат, который будет возвращен в ответе
        result = raw_data[["Date", "Sales_kg"]].to_dict(orient="records")

        return jsonify(result)

    @app.route("/predict/train_model", methods=["POST"])
    def train_model():
        try:
            result = model.train_model()
            return jsonify(result)
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/data/correlation", methods=["GET"])
    def get_correlation():
        # Получаем корреляцию
        correlation_matrix = data_loader.calculate_correlation()

        # Преобразуем корреляцию в словарь и возвращаем как JSON
        correlation_dict = correlation_matrix.to_dict()
        return jsonify(correlation_dict)
