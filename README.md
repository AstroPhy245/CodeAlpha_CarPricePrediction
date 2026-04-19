# 🚗 Car Price Prediction with Machine Learning
### CodeAlpha Data Science Internship — Task 3

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-1.3+-green?logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 📌 Project Overview

This project trains a machine learning regression model to **predict the price of a car** based on features such as brand, age, mileage, horsepower, fuel type, transmission, and condition. It covers the full ML pipeline — data generation, EDA, feature engineering, preprocessing, model training, and evaluation.

---

## 🗂 Dataset

- **Samples:** 1,000 car records (synthetically generated with realistic distributions)
- **Features:** 9 input features

| Feature | Type | Description |
|---|---|---|
| Brand | Categorical | Car manufacturer (Toyota, BMW, Mercedes, etc.) |
| Year | Numerical | Manufacturing year (2005–2023) |
| Mileage_km | Numerical | Total kilometers driven |
| Horsepower | Numerical | Engine horsepower (70–500 HP) |
| Engine_Size | Numerical | Engine displacement in litres (1.0–5.0L) |
| Fuel_Type | Categorical | Petrol / Diesel / Electric / Hybrid |
| Transmission | Categorical | Manual / Automatic |
| Condition | Categorical | New / Like New / Good / Fair / Poor |
| Num_Owners | Numerical | Number of previous owners |

- **Target Variable:** `Price_USD` (car selling price in US dollars)

---

## 🧪 Models Trained

| Model | MAE | RMSE | R² Score |
|---|---|---|---|
| Linear Regression | $7,670 | $10,519 | 0.2325 |
| Decision Tree | $2,966 | $5,191 | 0.8131 |
| Random Forest | $2,130 | $3,796 | 0.9000 |
| **Gradient Boosting** ✅ | **$2,182** | **$3,640** | **0.9081** |

> ✅ Best performing model: **Gradient Boosting Regressor** with R² = 0.9081

---

## 📊 Visualizations Generated

| File | Description |
|---|---|
| `car_eda.png` | Price distribution, avg price by brand, price vs year/mileage, price by fuel & condition |
| `car_correlation.png` | Feature correlation heatmap |
| `car_model_results.png` | Actual vs Predicted, Residual plot, R² comparison chart |
| `car_feature_importance.png` | Top features ranked by importance (Random Forest) |

---

## ⚙️ Feature Engineering

Three additional features were derived to improve model performance:
- `Car_Age` = 2024 − Year
- `Mileage_per_Year` = Mileage_km ÷ Car_Age
- `Price_per_HP` = Price_USD ÷ Horsepower (for EDA insight)

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/CodeAlpha_CarPricePrediction.git
cd CodeAlpha_CarPricePrediction
```

### 2. Install Dependencies
```bash
pip install scikit-learn pandas matplotlib seaborn numpy
```

### 3. Run the Script
```bash
python task3_car_price_prediction.py
```

---

## 📁 Project Structure

```
CodeAlpha_CarPricePrediction/
│
├── task3_car_price_prediction.py  # Main Python script
├── car_eda.png                    # Exploratory data analysis plots
├── car_correlation.png            # Feature correlation heatmap
├── car_model_results.png          # Model performance visualizations
├── car_feature_importance.png     # Feature importance chart
└── README.md                      # Project documentation
```

---

## 🔍 Key Findings

- **Brand, Car Age, and Mileage** are the strongest predictors of car price
- Electric and Hybrid vehicles command a **price premium** over petrol/diesel
- Automatic transmission cars are priced **~8% higher** on average than manual
- Gradient Boosting outperformed all other models with an **R² of 0.908**, explaining over 90% of price variance
- Linear Regression performed poorly (R² = 0.23), confirming the **non-linear** nature of car pricing

---

## 🛠 Tech Stack

- **Language:** Python 3.8+
- **Libraries:** Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn

---

## 👤 Author

**[Priyadarshini Lodh]**

[LinkedIn Profile]([https://linkedin.com](https://www.linkedin.com/in/priyadarshini-lodh-b63737332?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app))

---

*This project was completed as part of the CodeAlpha Data Science Internship Program.*
