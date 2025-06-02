import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

# 1. 加载数据
data = pd.read_csv(r'D:\paq\GEE\problem\bj19_2.csv')
print("数据样例：\n", data.head())

# 2. 选择特征和目标
# 增加了新的特征: NDVI, NDWI, VIRS, POP, GAIA
features = [ 'NDVI', 'NDWI', 'VIRS', 'POP', 'GAIA']
X = data[features].values
y = data['LST'].values

# 3. 数据预处理 - 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. 划分训练和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 5. 模型训练
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='rmse',
    random_state=42
)
model.fit(X_train, y_train)

# 6. 模型预测与评估
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

# 7. 特征重要性可视化
plt.figure(figsize=(10, 6))
importance = model.feature_importances_
sorted_idx = np.argsort(importance)
plt.barh(np.array(features)[sorted_idx], importance[sorted_idx])
plt.xlabel("XGBoost Feature Importance")
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
plt.show()

# 8. 排列重要性 - 更可靠的特征重要性评估方法
result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
sorted_idx = result.importances_mean.argsort()

plt.figure(figsize=(10, 6))
plt.boxplot(result.importances[sorted_idx].T, vert=False, tick_labels=np.array(features)[sorted_idx])
plt.title("Permutation Importance (test set)")
plt.tight_layout()
plt.savefig('permutation_importance.png', dpi=300)
plt.show()

# 9. 预测值与实际值对比
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')