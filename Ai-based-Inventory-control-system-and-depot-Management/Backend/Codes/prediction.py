import pandas as pd
import joblib

# -----------------------------
# 1️⃣ Load new inventory data
# -----------------------------
new_inventory_df = pd.read_csv(r"D:\Inventory Project\Backend\Dataset\realistic_inventory_10000_for_prediction.csv")

# -----------------------------
# 2️⃣ Load saved XGBoost model and label encoder
# -----------------------------
xgb_model = joblib.load(r"D:\Inventory Project\Backend\Models\xgboost_inventory_model.pkl")
le = joblib.load(r"D:\Inventory Project\Backend\Models\label_encoder.pkl")

# -----------------------------
# 3️⃣ Predict stock levels
# -----------------------------
pred_numeric = xgb_model.predict(new_inventory_df)
pred_labels = le.inverse_transform(pred_numeric)
new_inventory_df['predicted_stock_level'] = pred_labels

# -----------------------------
# 4️⃣ Add restock alert column
# -----------------------------
def restock_alert(stock_level):
    if stock_level in ['Very Low', 'Low']:
        return 'Restock Needed'
    elif stock_level in ['High', 'Very High']:
        return 'Overstock'
    else:  # Perfect
        return 'Stock OK'

new_inventory_df['restock_alert'] = new_inventory_df['predicted_stock_level'].apply(restock_alert)

# -----------------------------
# 5️⃣ Add priority column
# -----------------------------
def priority_level(stock_level):
    if stock_level == 'Very Low':
        return 'High Priority'
    elif stock_level == 'Low':
        return 'Medium Priority'
    elif stock_level == 'Perfect':
        return 'Normal'
    else:  # High or Very High
        return 'Check/Overstock'

new_inventory_df['priority'] = new_inventory_df['predicted_stock_level'].apply(priority_level)

# -----------------------------
# 6️⃣ Preview results
# -----------------------------
print(new_inventory_df.sample(10))

# -----------------------------
# 7️⃣ Save to CSV
# -----------------------------
path = r"D:\Inventory Project\Backend\Prediction Output in CSV\predicted_inventory_with_alerts_and_priority.csv"
new_inventory_df.to_csv(path, index=False)
print(f"Predictions saved to {path}")

