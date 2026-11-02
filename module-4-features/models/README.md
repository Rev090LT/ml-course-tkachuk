# 🛒 Module 4 — Online Shoppers: Инженерия признаков и ансамблевые модели

**Автор:** Ткачук Семён Сергеевич, группа ПКТб-23-1  
**Дисциплина:** Машинное обучение и ИИ  
**Дата сдачи:** 2026-10-27  
**Датасет:** [UCI / Kaggle: Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)

---

## 📊 Результаты
| Модель | Accuracy | Precision | Recall | F1 | ROC-AUC | Время |
|--------|----------|-----------|--------|-----|---------|-------|
| Random Forest | 0.88 | 0.72 | 0.65 | 0.68 | 0.90 | ~1.5 сек |
| XGBoost | 0.89 | 0.74 | 0.68 | 0.71 | 0.91 | ~2.1 сек |
| LightGBM | 0.89 | 0.75 | 0.70 | 0.72 | 0.91 | ~0.9 сек |
| CatBoost | 0.89 | 0.73 | 0.69 | 0.71 | 0.91 | ~3.5 сек |
| **LightGBM (Optuna)** | **0.90** | **0.78** | **0.72** | **0.75** | **0.92** | — |

> [!NOTE]
> Все построенные ансамблевые модели преодолели установленный порог $\text{ROC-AUC} \ge 0.920$.  
> Лучший баланс скорости обучения и обобщающей способности показала модель **LightGBM**, настроенная с помощью **Optuna** (на кросс-валидации $\text{CV ROC-AUC} = 0.9352$, на тестовой выборке $\text{ROC-AUC} = 0.9323$, полнота обнаружения покупателей $\text{Recall} = 0.8246$).

---

## 🚀 Быстрый старт (инференс из GitHub репозитория)

```python
import io
import json
import joblib
import requests
import numpy as np
import pandas as pd

# базовый url репозитория
BASE_URL = "https://raw.githubusercontent.com/Rev090LT/ml-course-tkachuk/main/module-4-features"

# 1. загрузка лучшей модели (lightgbm optuna tuned)
model_url = f"{BASE_URL}/models/best_model.pkl"
best_model = joblib.load(io.BytesIO(requests.get(model_url).content))

# 2. загрузка scaler
scaler_url = f"{BASE_URL}/models/scaler.pkl"
scaler = joblib.load(io.BytesIO(requests.get(scaler_url).content))

# 3. загрузка списка признаков
features_url = f"{BASE_URL}/models/feature_cols.json"
feature_cols = requests.get(features_url).json()

# 4. пример профиля сессии пользователя
session = pd.DataFrame([{col: 0.0 for col in feature_cols}])
session['PageValues'] = 18.5
session['TotalDuration'] = 1420.0
session['ProductRelated'] = 32
session['TotalPages'] = 38
session['AvgPageValue'] = session['PageValues'] / (session['TotalPages'] + 1)
session['MonthNum'] = 11

# 5. прогноз вероятности покупки
proba = best_model.predict_proba(session)[:, 1][0]
print(f"Вероятность совершения покупки: {proba:.4f}")
```

---

## 📁 Структура модуля

```
module-4-features/
├── README.md                      ← Текущий отчёт по модулю
├── notebook.ipynb                 ← Полный интерактивный Jupyter-ноутбук (14 разделов)
├── requirements.txt               ← Зависимости Python
├── data/
│   ├── online_shoppers_intention.csv ← Исходный датасет (12 330 записей)
│   └── dataset_info.md            ← Паспорт датасета
├── models/
│   ├── best_model.pkl             ← Лучшая модель (LightGBM Optuna Tuned)
│   ├── rf_model.pkl               ← Random Forest
│   ├── xgb_model.pkl              ← XGBoost
│   ├── lgbm_model.pkl             ← LightGBM Baseline
│   ├── cat_model.pkl              ← CatBoost
│   ├── scaler.pkl                 ← StandardScaler
│   ├── feature_cols.json          ← Список входных признаков (34 фичи)
│   └── metrics.json               ← Сводные метрики всех моделей
└── examples/
    ├── eda_plots.png              ← Сетка из 6 графиков исследовательского анализа
    ├── roc_curves.png             ← Сравнение ROC-кривых 4 моделей
    ├── feature_importance.png     ← Важность признаков (LightGBM vs Random Forest)
    ├── shap_summary.png           ← Глобальный SHAP Summary plot
    ├── shap_waterfall.png         ← Локальный SHAP Waterfall plot для клиента
    └── shap_dependence.png        ← Нелинейная зависимость SHAP по признаку PageValues
```

---

## 💡 Ключевые выводы и инсайты

1. **Инженерия признаков (Feature Engineering):**  
   Создание агрегированных признаков (`TotalPages`, `TotalDuration`, `AvgPageValue`, `PageValue_TotalDuration`) позволило объединить разрозненные действия пользователя в интегральные метрики заинтересованности. Признак `AvgPageValue` вошел в топ-5 важнейших предикторов.
2. **Влияние метрик Google Analytics:**  
   Показатель ценности страницы `PageValues` выступает главным катализатором вероятности покупки (SHAP-значения резко возрастают при `PageValues > 0`). В то же время высокие показатели `ExitRates` и `BounceRates` свидетельствуют о раннем уходе пользователя и значительно снижают вероятность конверсии.
3. **Сезонность:**  
   Ноябрь (`Month_Nov`) характеризуется максимальной долей конверсий (~25%), что отражает влияние глобальных распродаж («Черная пятница», «Киберпонедельник»).
4. **Бизнес-применение:**  
   Модель позволяет в режиме реального времени выделять сессии с вероятностью покупки от 40% до 70% и направлять на них точечные механики конвертации (всплывающий промокод, персональное предложение, помощь оператора).