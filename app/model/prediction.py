import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import timedelta

class PredictionModel:
    def __init__(self, data):
        self.data = data
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False

    def preprocess_data(self):
        # Исключаем строки с некорректными значениями (ноль или отрицательные значения)
        self.data = self.data[self.data["Sales_kg"] > 0]

        # Исключение выбросов на основе метода межквартильного размаха (IQR)
        q1 = self.data["Sales_kg"].quantile(0.25)  # Первый квартиль
        q3 = self.data["Sales_kg"].quantile(0.75)  # Третий квартиль
        iqr = q3 - q1  # Межквартильный размах

        lower_bound = q1 - 3 * iqr  # Нижняя граница
        upper_bound = q3 + 3 * iqr  # Верхняя граница

        # Оставляем только значения в пределах границ
        self.data = self.data[(self.data["Sales_kg"] >= lower_bound) & (self.data["Sales_kg"] <= upper_bound)]
        return self.data

    def create_sequences(self, data, sequence_length):
        sequences = []
        targets = []
        for i in range(len(data) - sequence_length):
            sequences.append(data[i:i + sequence_length])
            targets.append(data[i + sequence_length])
        return np.array(sequences), np.array(targets)

    def train_model(self):
        # Предварительная обработка данных
        self.data = self.preprocess_data()

        # Масштабируем данные
        sales_data = self.data["Sales_kg"].values.reshape(-1, 1)
        sales_scaled = self.scaler.fit_transform(sales_data)

        # Создаем последовательности
        sequence_length = 30  # Длина временной последовательности
        X, y = self.create_sequences(sales_scaled, sequence_length)

        # Разделяем на тренировочную и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

        # Создаем модель
        self.model = Sequential([
            LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')

        # Обучаем модель
        self.model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test), verbose=1)
        self.is_trained = True
        print("Model trained successfully.")

    def get_summary_statistics(self):
        summary = {
            "mean": round(self.data["Sales_kg"].mean(), 2),
            "std_dev": round(self.data["Sales_kg"].std(), 2),
            "min": round(self.data["Sales_kg"].min(), 2),
            "max": round(self.data["Sales_kg"].max(), 2),
         }
        return summary

    def predict_next_7_days(self):
        
        if not self.is_trained:
            raise Exception("Model is not trained. Call train_model() first.")

        # Последние данные для прогноза
        recent_data = self.data["Sales_kg"].values[-30:].reshape(-1, 1)
        recent_scaled = self.scaler.transform(recent_data)

        # Последняя дата в данных
        last_date = self.data["Date"].max()

        # Прогнозируем на 7 дней
        predictions = []
        for i in range(7):
            input_data = recent_scaled[-30:].reshape(1, 30, 1)
            prediction = self.model.predict(input_data)[0][0]
            predictions.append(float(prediction))  # Преобразуем в тип float
            recent_scaled = np.append(recent_scaled, [[prediction]], axis=0)

        # Обратное масштабирование
        predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

        # Преобразуем результат в тип float для сериализации
        predictions = [float(pred[0]) for pred in predictions]

        # Возвращаем дату и прогноз
        return [{"date": (last_date + timedelta(days=i)).strftime("%d.%m.%Y"), "predicted_sales_kg": round(pred, 2)} for
                i, pred in enumerate(predictions)]

    def predict_monthly(self):
        if not self.is_trained:
            raise Exception("Model is not trained. Call train_model() first.")

        # Используем данные для прогнозирования на месяц
        recent_data = self.data["Sales_kg"].values[-30:].reshape(-1, 1)
        recent_scaled = self.scaler.transform(recent_data)

        # Последняя дата в данных
        last_date = self.data["Date"].max()

        predictions = []
        for i in range(30):  # Прогнозируем на 30 дней
            input_data = recent_scaled[-30:].reshape(1, 30, 1)
            prediction = self.model.predict(input_data)[0][0]
            predictions.append(float(prediction))  # Преобразуем в тип float
            recent_scaled = np.append(recent_scaled, [[prediction]], axis=0)

        # Обратное масштабирование
        predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        predictions = [float(pred[0]) for pred in predictions]  # Преобразуем в тип float для JSON

        # Возвращаем дату и прогноз
        return [{"date": (last_date + timedelta(days=i)).strftime("%d.%m.%Y"), "predicted_sales_kg": round(pred, 2)} for
                i, pred in enumerate(predictions)]

