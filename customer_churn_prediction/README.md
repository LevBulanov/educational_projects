# Прогнозирование оттока клиентов телеком-оператора

#### Краткое описание
- **Задача**: бинарная классификация — прогнозирование оттока клиентов  
- **Бизнес-цель**: вовремя выявлять клиентов с риском ухода и предлагать им удерживающие акции  
- **Результат**: лучшая модель CatBoost достигает **AUC-ROC ≈ 0.83** на тесте (целевой порог ≥ 0.80)  

#### Данные
- **Источник**: выгрузки заказчика (анонимизированы)  
- **Файлы**:  
  - `contract_new.csv` — договоры и платежи (BeginDate, EndDate, Type, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges)  
  - `personal_new.csv` — персональные данные (gender, SeniorCitizen, Partner, Dependents)  
  - `internet_new.csv` — интернет-услуги (InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies)  
  - `phone_new.csv` — телефония (MultipleLines)  
- **Объем**: 7 043 клиента (после объединения)  
- **Таргет**:  
  - `label = 1`, если EndDate ≠ “No” (клиент ушёл)  
  - `label = 0`, если EndDate == “No” (клиент остался)  

#### Предобработка и EDA (ключевые моменты)
- **Приведение типов**:  
  - даты → `datetime`  
  - `TotalCharges` → `float` (строки `' '` заменены на 0)  
- **Обработка EndDate**:  
  1. Заменили “No” на `None` для расчётов  
  2. Заполнили `None` датой выгрузки (`2020-02-01`) для получения производных признаков  
  3. Оригинальные даты исключили из модели, чтобы избежать утечек  
- **Объединение**: по `customerID`  
- **Логичные пропуски** (отсутствие услуги) заменены на категории:  
  - `Dont use internet service`  
  - `Dont use phone service`  
- **Итоговый датасет**: 21 признак + таргет, доля положительного класса ≈ 15%  

#### Фичи в модели
- **Числовые**: `MonthlyCharges`, `TotalCharges`  
- **Категориальные**:  
  `Type`, `PaperlessBilling`, `PaymentMethod`, `gender`, `SeniorCitizen`, `Partner`, `Dependents`,  
  `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`,  
  `StreamingTV`, `StreamingMovies`, `MultipleLines`  
- **Исключены**: `customerID`, `BeginDate`, `EndDate` (во избежание утечек)  

#### Метрики и цель
- **Основная метрика**: AUC-ROC (цель ≥ 0.80)  
- **Дополнительно**: F1-score, Average Precision (AUC-PR), матрица ошибок  

#### Модели и обучение
- **Разбиение**: train/test = 75%/25% (со стратификацией)  
- **Алгоритмы**:  
  - CatBoostClassifier (Grid Search, `Pool` для категориальных)  
  - RandomForestClassifier (OrdinalEncoder в `ColumnTransformer` для категориальных)  
- **Поиск гиперпараметров**:  
  - **CatBoost (лучшие)**:  
    - `depth=4`, `one_hot_max_size=10`, `l2_leaf_reg=2`  
    - `iterations=500`, `learning_rate=0.1`, `random_seed=170723`  
  - **RandomForest**:  
    - `n_estimators` ∈ [40, 80, 100]  
    - `max_depth` ∈ [2, 4, 8]  
    - `max_features` ∈ [2, 4]  

#### Результаты
- **Кросс-валидация**:  
  - CatBoost AUC-ROC ≈ 0.814  
  - RandomForest AUC-ROC ≈ 0.800  
- **Тестовая выборка (CatBoost)**:  
  - AUC-ROC ≈ 0.828  
  - Порог вероятности 0.23 (по F1 на кросс-валидации)  
  - F1 ≈ 0.484  
  - Average Precision ≈ 0.515  
  - **Матрица ошибок**:  
    - TN: 1323, FP: 163, FN: 135, TP: 140  
- **Важность признаков (CatBoost, по убыванию)**:  
  1. `TotalCharges`, `MonthlyCharges`  
  2. `Type`, `PaymentMethod`, `MultipleLines`, `Partner`, `Dependents`  
  3. `InternetService`, `StreamingTV`, `OnlineBackup`, `TechSupport`, `StreamingMovies`,  
     `DeviceProtection`, `OnlineSecurity`, `PaperlessBilling`, `SeniorCitizen`, `gender`  

# Используемые библиотеки 
*pandas, matplotlib, seaborn, phik, catboost, numpy, sklearn*
