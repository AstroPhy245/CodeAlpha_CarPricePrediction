# ============================================================
# TASK 3: Car Price Prediction with Machine Learning
# CodeAlpha Data Science Internship
# ============================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ── 1. Create Realistic Dataset ──────────────────────────────
np.random.seed(42)
n = 1000

brands = {
    'Toyota':  (1.0, 15000), 'Honda':  (0.95, 14000),
    'Ford':    (0.85, 13000), 'BMW':    (1.8,  35000),
    'Mercedes':(2.0,  45000), 'Audi':   (1.7,  32000),
    'Hyundai': (0.75, 12000), 'Kia':    (0.70, 11000),
    'Chevrolet':(0.80,13500), 'Nissan': (0.82, 12500)
}
fuel_types    = ['Petrol', 'Diesel', 'Electric', 'Hybrid']
transmissions = ['Manual', 'Automatic']
conditions    = ['New', 'Like New', 'Good', 'Fair', 'Poor']

brand_list    = np.random.choice(list(brands.keys()), n)
fuel          = np.random.choice(fuel_types, n, p=[0.45, 0.30, 0.10, 0.15])
transmission  = np.random.choice(transmissions, n, p=[0.40, 0.60])
condition     = np.random.choice(conditions, n, p=[0.15, 0.25, 0.30, 0.20, 0.10])
year          = np.random.randint(2005, 2024, n)
mileage       = np.abs(np.random.normal(50000, 30000, n)).astype(int)
horsepower    = np.random.randint(70, 500, n)
engine_size   = np.round(np.random.uniform(1.0, 5.0, n), 1)
num_owners    = np.random.randint(1, 5, n)

# Price formula
prices = []
for i in range(n):
    b = brand_list[i]
    goodwill, base = brands[b]
    age_factor   = max(0.3, 1 - (2024 - year[i]) * 0.055)
    cond_factor  = {'New':1.0,'Like New':0.92,'Good':0.80,'Fair':0.65,'Poor':0.45}[condition[i]]
    fuel_factor  = {'Electric':1.15,'Hybrid':1.10,'Diesel':1.05,'Petrol':1.0}[fuel[i]]
    trans_factor = 1.08 if transmission[i] == 'Automatic' else 1.0
    mile_factor  = max(0.5, 1 - mileage[i] / 250000)
    hp_bonus     = horsepower[i] * 30
    owner_factor = max(0.75, 1 - (num_owners[i] - 1) * 0.07)
    price = (base * goodwill * age_factor * cond_factor * fuel_factor
             * trans_factor * mile_factor * owner_factor + hp_bonus)
    price += np.random.normal(0, price * 0.05)
    prices.append(max(3000, round(price, 2)))

df = pd.DataFrame({
    'Brand': brand_list, 'Year': year, 'Mileage_km': mileage,
    'Horsepower': horsepower, 'Engine_Size': engine_size,
    'Fuel_Type': fuel, 'Transmission': transmission,
    'Condition': condition, 'Num_Owners': num_owners, 'Price_USD': prices
})

print("=" * 55)
print("        CAR PRICE PREDICTION PROJECT")
print("=" * 55)
print(f"\nDataset shape  : {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nMissing values :\n{df.isnull().sum()}")
print(f"\nPrice statistics:\n{df['Price_USD'].describe().round(0)}")

# ── 2. EDA ───────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Car Price Prediction — EDA', fontsize=16, fontweight='bold')

# Price distribution
axes[0,0].hist(df['Price_USD'], bins=40, color='#2196F3', edgecolor='white', alpha=0.85)
axes[0,0].set_title('Price Distribution'); axes[0,0].set_xlabel('Price (USD)')
axes[0,0].axvline(df['Price_USD'].mean(), color='red', linestyle='--', label='Mean')
axes[0,0].legend()

# Avg price by brand
brand_avg = df.groupby('Brand')['Price_USD'].mean().sort_values(ascending=False)
axes[0,1].bar(brand_avg.index, brand_avg.values,
              color=plt.cm.viridis(np.linspace(0.2,0.8,len(brand_avg))), edgecolor='white')
axes[0,1].set_title('Avg Price by Brand'); axes[0,1].set_ylabel('Avg Price (USD)')
axes[0,1].tick_params(axis='x', rotation=45, labelsize=8)

# Price vs Year
axes[0,2].scatter(df['Year'], df['Price_USD'], alpha=0.3, color='#4CAF50', s=10)
axes[0,2].set_title('Price vs Year'); axes[0,2].set_xlabel('Year')
axes[0,2].set_ylabel('Price (USD)')

# Price vs Mileage
axes[1,0].scatter(df['Mileage_km'], df['Price_USD'], alpha=0.3, color='#FF5722', s=10)
axes[1,0].set_title('Price vs Mileage'); axes[1,0].set_xlabel('Mileage (km)')

# Price by Fuel Type
df.boxplot(column='Price_USD', by='Fuel_Type', ax=axes[1,1])
axes[1,1].set_title('Price by Fuel Type'); axes[1,1].set_xlabel('')
plt.sca(axes[1,1]); plt.title('Price by Fuel Type')

# Price by Condition
df.boxplot(column='Price_USD', by='Condition', ax=axes[1,2])
axes[1,2].set_title('Price by Condition'); axes[1,2].set_xlabel('')
plt.sca(axes[1,2]); plt.title('Price by Condition')
axes[1,2].tick_params(axis='x', rotation=20, labelsize=8)

plt.suptitle('Car Price Prediction — EDA', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/car_eda.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[Saved] car_eda.png")

# ── 3. Feature Engineering ───────────────────────────────────
df_ml = df.copy()
df_ml['Car_Age']        = 2024 - df_ml['Year']
df_ml['Price_per_HP']   = df_ml['Price_USD'] / df_ml['Horsepower']
df_ml['Mileage_per_Year'] = df_ml['Mileage_km'] / (df_ml['Car_Age'] + 1)

# Encode categoricals
le = LabelEncoder()
for col in ['Brand', 'Fuel_Type', 'Transmission', 'Condition']:
    df_ml[col + '_enc'] = le.fit_transform(df_ml[col])

features = ['Car_Age', 'Mileage_km', 'Horsepower', 'Engine_Size',
            'Num_Owners', 'Mileage_per_Year',
            'Brand_enc', 'Fuel_Type_enc', 'Transmission_enc', 'Condition_enc']
X = df_ml[features]
y = df_ml['Price_USD']

# ── 4. Correlation Heatmap ───────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
corr_data = df_ml[features + ['Price_USD']]
mask = np.triu(np.ones_like(corr_data.corr(), dtype=bool))
sns.heatmap(corr_data.corr(), annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=ax, mask=mask, linewidths=0.5)
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/car_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print("[Saved] car_correlation.png")

# ── 5. Train/Test Split ──────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\nTraining samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")

# ── 6. Train Models ──────────────────────────────────────────
models = {
    'Linear Regression'     : LinearRegression(),
    'Decision Tree'         : DecisionTreeRegressor(max_depth=8, random_state=42),
    'Random Forest'         : RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting'     : GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}
print("\n" + "=" * 55)
print("           MODEL PERFORMANCE COMPARISON")
print("=" * 55)
print(f"{'Model':<25} {'MAE':>8} {'RMSE':>10} {'R²':>8}")
print("-" * 55)

for name, model in models.items():
    model.fit(X_train_sc, y_train)
    preds = model.predict(X_test_sc)
    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    results[name] = {'model': model, 'preds': preds, 'mae': mae, 'rmse': rmse, 'r2': r2}
    print(f"{name:<25} {mae:>8,.0f} {rmse:>10,.0f} {r2:>8.4f}")

best_name = max(results, key=lambda k: results[k]['r2'])
best = results[best_name]
print(f"\nBest Model : {best_name}  (R² = {best['r2']:.4f})")

# ── 7. Results Visualization ─────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f'Car Price Prediction — {best_name}', fontsize=14, fontweight='bold')

# Actual vs Predicted
axes[0].scatter(y_test, best['preds'], alpha=0.4, color='#2196F3', s=15)
mn, mx = y_test.min(), y_test.max()
axes[0].plot([mn, mx], [mn, mx], 'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Price'); axes[0].set_ylabel('Predicted Price')
axes[0].set_title('Actual vs Predicted'); axes[0].legend()

# Residuals
residuals = y_test.values - best['preds']
axes[1].scatter(best['preds'], residuals, alpha=0.4, color='#FF5722', s=15)
axes[1].axhline(0, color='black', linestyle='--', lw=1.5)
axes[1].set_xlabel('Predicted Price'); axes[1].set_ylabel('Residuals')
axes[1].set_title('Residual Plot')

# R² comparison
names = list(results.keys())
r2s   = [results[n]['r2'] for n in names]
bar_colors = ['#4CAF50' if n == best_name else '#90CAF9' for n in names]
bars = axes[2].barh(names, r2s, color=bar_colors, edgecolor='white', height=0.5)
axes[2].set_xlim(0, 1.1)
axes[2].set_xlabel('R² Score')
axes[2].set_title('R² Score Comparison')
for bar, r2 in zip(bars, r2s):
    axes[2].text(r2 + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{r2:.3f}', va='center', fontsize=9)
axes[2].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/car_model_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("[Saved] car_model_results.png")

# ── 8. Feature Importance ────────────────────────────────────
rf_model = results['Random Forest']['model']
feat_imp = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 5))
feat_imp.plot(kind='barh', ax=ax, color='#2196F3', edgecolor='white')
ax.set_title('Feature Importance — Random Forest', fontsize=13, fontweight='bold')
ax.set_xlabel('Importance Score')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('/home/claude/car_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("[Saved] car_feature_importance.png")

print("\n" + "=" * 55)
print("  TASK 3 COMPLETE — Car Price Prediction")
print("=" * 55)
