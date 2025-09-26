import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
import joblib

# -----------------------------
# 1️⃣ Load dataset
# -----------------------------
df = pd.read_csv(r"D:\Inventory Project\trial 01\inventory_training_5000.csv")

# Split features and target
X = df.drop(columns=['stock_level'])
y = df['stock_level']

# -----------------------------
# 2️⃣ Encode target labels
# -----------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Label mapping:", label_mapping)

# -----------------------------
# 3️⃣ Identify numeric and categorical columns
# -----------------------------
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# -----------------------------
# 4️⃣ Preprocessing pipeline
# -----------------------------
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# -----------------------------
# 5️⃣ Train/Test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# -----------------------------
# 6️⃣ Define and train XGBoost model
# -----------------------------
xgb_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    eval_metric='mlogloss',
    random_state=42
)

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', xgb_model)])

pipeline.fit(X_train, y_train)

# -----------------------------
# 7️⃣ Evaluate model
# -----------------------------
y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)

print("\n✅ XGBoost Model Evaluation")
print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
print("Confusion Matrix:\n", cm)

# -----------------------------
# 8️⃣ Retrain on full dataset
# -----------------------------
pipeline.fit(X, y_encoded)

# -----------------------------
# 9️⃣ Save the model and label encoder
# -----------------------------
joblib.dump(pipeline, 'xgboost_inventory_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
print("\n✅ XGBoost model saved as 'xgboost_inventory_model.pkl'")
print("✅ Label encoder saved as 'label_encoder.pkl'")
