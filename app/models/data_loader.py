import pandas as pd
from app.config import DATA_PATH


class DataLoader:
    def load_data(self):
        # Загрузка данных и подготовка
        data = pd.read_csv(DATA_PATH, delimiter=';', header=0, names=['Date', 'Product_Category', 'Product', 'City', 'Customer_Group', 'Point_Format', 'Sales_kg'])

        data['Sales_kg'] = data['Sales_kg'].str.replace(',', '.').astype(float)
        # Пример предобработки данных
        data["Date"] = pd.to_datetime(data["Date"], format="%d.%m.%Y")
        return data
