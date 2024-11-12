import pandas as pd

class DataLoader:
    def load_data(self):
        # Загрузка данных и подготовка
        data = pd.read_csv("data.csv", sep=";", encoding="cp1251")
        data['column_name'] = data['column_name'].str.replace(',', '.').astype(float)
        # Пример предобработки данных
        data["Дата"] = pd.to_datetime(data["Дата"], format="%d.%m.%Y")
        return data
