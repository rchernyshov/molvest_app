from app import create_app
from app.models.data_loader import DataLoader
from app.models.prediction import PredictionModel
import plotly.express as px

# Загрузка данных
loader = DataLoader()
data = loader.load_data()

# Создание и обучение модели
model = PredictionModel(data)
model.train_model()

# Предсказания
print("Статистика продаж:", model.get_stats())
print("Прогноз на 7 ближайших дней:", model.predict_next_7_days())
print("Прогноз на следующие месяцы:", model.predict_monthly())
print("Прогноз на определённую дату:", model.predict("2024-12-25"))

data.rename(columns={'ds': 'Date', 'y': 'Sales_kg'}, inplace=True)

# print(data.columns)  # Выведет имена всех столбцов

fig = px.line(data, x='Date', y='Sales_kg', title="Динамика продаж", labels={'Sales_kg': 'Продажи (кг)', 'Date': 'Дата'})
fig.show()


# Запуск Flask-приложения
app = create_app()
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
