from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

class PredictionModel:
    def __init__(self, data):
        self.data = data
        self.model = LinearRegression()

    def train_model(self):
        # Простая тренировка модели на временных данных
        X = np.array(range(len(self.data))).reshape(-1, 1)
        y = self.data["Продажи, кг"]
        self.model.fit(X, y)

    def predict(self, date_str):
        # Предсказание на основе даты
        if date_str:
            # Преобразование даты в индекс
            date = pd.to_datetime(date_str, format="%Y-%m-%d")
            date_index = (date - self.data["Дата"].min()).days
        else:
            date_index = len(self.data)
        predicted_sales = self.model.predict([[date_index]])[0]
        return {"date": date_str or "следующий день", "predicted_sales_kg": round(predicted_sales, 2)}

    def get_stats(self):
        # Статистика по данным
        avg_sales = self.data["Продажи, кг"].mean()
        max_sales = self.data["Продажи, кг"].max()
        min_sales = self.data["Продажи, кг"].min()
        return {
            "average_sales_kg": round(avg_sales, 2),
            "max_sales_kg": round(max_sales, 2),
            "min_sales_kg": round(min_sales, 2)
        }
