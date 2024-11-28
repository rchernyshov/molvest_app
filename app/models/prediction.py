from prophet import Prophet
import pandas as pd


class PredictionModel:
    def __init__(self, data):
        self.data = data
        self.model = None  # Инициализация модели Prophet
        self.is_trained = False

    def preprocess_data(self):
        # Подготовка данных для Prophet
        self.data.rename(columns={"Date": "ds", "Sales_kg": "y"}, inplace=True)
        self.data = self.data[["ds", "y"]]  # Оставляем только нужные столбцы
        self.data = self.data[self.data["y"] > 0]  # Исключаем некорректные значения (если есть)
        return self.data

    def train_model(self):
        # Предварительная обработка данных
        self.data = self.preprocess_data()

        # Создание и обучение модели Prophet
        self.model = Prophet()
        self.model.fit(self.data)
        self.is_trained = True
        print("Model trained successfully.")

    def predict_next_7_days(self):
        if not self.is_trained:
            raise Exception("Model is not trained. Call train_model() first.")

        # Создание будущей даты на 7 дней
        future = self.model.make_future_dataframe(periods=7)
        forecast = self.model.predict(future)

        # Возвращаем только предсказания на 7 дней
        predictions = forecast[["ds", "yhat"]].iloc[-7:]
        result = [{"day": i + 1, "predicted_sales_kg": round(row["yhat"], 2)}
                  for i, row in predictions.iterrows()]
        return result

    def predict_monthly(self):
        if not self.is_trained:
            raise Exception("Model is not trained. Call train_model() first.")

        # Прогноз на следующие 12 месяцев
        future = self.model.make_future_dataframe(periods=365)  # Прогноз на год
        forecast = self.model.predict(future)

        # Группировка прогнозов по месяцам
        forecast["month"] = forecast["ds"].dt.to_period("M")
        monthly_forecast = forecast.groupby("month")["yhat"].mean().reset_index()

        # Возвращаем предсказания на следующие 12 месяцев
        result = [{"month": row["month"].strftime("%Y-%m"),
                   "predicted_sales_kg": round(row["yhat"], 2)}
                  for _, row in monthly_forecast.head(12).iterrows()]
        return result

    def predict(self, date_str):
        if not self.is_trained:
            raise Exception("Model is not trained. Call train_model() first.")

        # Прогноз на определённую дату
        future = pd.DataFrame({"ds": [pd.to_datetime(date_str)]})
        forecast = self.model.predict(future)
        predicted_sales = forecast.loc[0, "yhat"]
        return {"date": date_str, "predicted_sales_kg": round(predicted_sales, 2)}

    def get_stats(self):
        # Статистика по данным
        avg_sales = self.data["y"].mean()
        max_sales = self.data["y"].max()
        min_sales = self.data["y"].min()
        return {
            "average_sales_kg": round(avg_sales, 2),
            "max_sales_kg": round(max_sales, 2),
            "min_sales_kg": round(min_sales, 2)
        }