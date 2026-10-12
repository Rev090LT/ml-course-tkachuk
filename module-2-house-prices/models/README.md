# 🏠 Module 2 — House Prices: предобработка и регрессия

**Автор:** Ткачук Семён Сергеевич, группа ПКТб-23-1  
**Дисциплина:** Машинное обучение и ИИ  
**Дата сдачи:** 2026-10-13  
**Датасет:** [House Prices - Advanced Regression Techniques (Kaggle)](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)  

---

## 📊 Результаты

| Модель | RMSE (log) | MAE (log) | R² | RMSE ($) | Время обучения |
|---|---|---|---|---|---|
| Linear Regression | 0.135 | 0.095 | 0.88 | $28 000 | ~0.01 сек |
| Ridge | 0.128 | 0.090 | 0.90 | $26 000 | ~0.01 сек |
| **Lasso** | **0.125** | **0.088** | **0.91** | **$25 000** | ~0.02 сек |
| ElasticNet | 0.126 | 0.089 | 0.91 | $25 500 | ~0.02 сек |

> [!NOTE]
> Лучший результат по метрике R² показала модель **Lasso** (R² = 0.91), которая также успешно выполнила отбор признаков, обнулив более 50% неинформативных весов.

---

## 🚀 Быстрый старт (инференс из репозитория)

```python
import io
import joblib
import requests
import numpy as np
import pandas as pd

BASE_URL = "https://raw.githubusercontent.com/Rev090LT/ml-course-tkachuk/tree/main/module-2-house-prices"

# 1. Загрузка обученной модели Lasso
model_url = f"{BASE_URL}/models/lasso_model.pkl"
lasso_model = joblib.load(io.BytesIO(requests.get(model_url).content))

# 2. Загрузка scaler и списка признаков
scaler = joblib.load(io.BytesIO(requests.get(f"{BASE_URL}/models/scaler.pkl").content))
feature_cols = requests.get(f"{BASE_URL}/models/feature_cols.json").json()

print(f"✅ Модель успешно загружена! Количество признаков: {len(feature_cols)}")

# 3. Пример предсказания
def predict_price(house_data, model, scaler, feature_cols):
    X_new = pd.DataFrame([house_data])
    for col in feature_cols:
        if col not in X_new.columns:
            X_new[col] = 0
    X_new = X_new[feature_cols]
    X_new_scaled = scaler.transform(X_new)
    y_pred_log = model.predict(X_new_scaled)[0]
    return np.expm1(y_pred_log)


# test_house = {'OverallQual': 8, 'TotalSF': 3000, 'GrLivArea': 2000, 'YearBuilt': 2010, 'GarageCars': 2, 'TotalBath': 2.5}
# print(f"Предсказанная цена: ${predict_price(test_house, lasso_model, scaler, feature_cols):,.0f}")


## 📁 Структура модуля

module-2-house-prices/
├── README.md               # Этот файл: отчёт и документация
├── notebook.ipynb          # Полный Jupyter Notebook (14 разделов)
├── data/
│   ├── train.csv           # Обучающая выборка (1460 объектов)
│   ├── test.csv            # Тестовая выборка (1459 объектов)
│   └── house_prices_info.md # Описание датасета
├── models/
│   ├── lr_model.pkl        # Веса Linear Regression
│   ├── ridge_model.pkl     # Веса Ridge
│   ├── lasso_model.pkl     # Веса Lasso (лучшая модель)
│   ├── enet_model.pkl      # Веса ElasticNet
│   ├── scaler.pkl          # Обученный StandardScaler
│   ├── feature_cols.json   # Точный порядок и список входных признаков
│   └── metrics.json        # Итоговые метрики всех моделей
├── examples/
│   ├── eda_plots.png       # Аналитическая сетка графиков первичного анализа
│   ├── predictions_scatter.png # График реальных vs предсказанных цен
│   └── residuals.png       # График анализа остатков
└── requirements.txt        # Список зависимостей