#srikar is a bum
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, cross_val_predict, LeaveOneOut
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np


data = pd.read_csv("data.csv")


data_rename = data.rename(columns={
    "Turbidity (NTU)": "Turbidity",
    "DO (mg/L)": "DO_mg/L",
    "Temperature (°C)": "Temperature_C",
    "Ammonium Ion": "Ammonium_Ion",
    "Amount (mg/L)": "Amount"
})


data_clean = data_rename.drop(["DO_mg/L", "Tds", "Ammonium_Ion", "Nitrate"], axis=1)


data_clean["Contaminant"] = data_clean["Contaminant"].str.strip()
data_clean = data_clean[data_clean["Contaminant"].notna() & (data_clean["Contaminant"] != "")]


data_clean["Amount"] = data_clean["Amount"].replace(",", "", regex=True)
data_clean["Amount"] = pd.to_numeric(data_clean["Amount"], errors="coerce")

print("=== BEFORE IMPUTATION ===")
print(f"Total rows: {len(data_clean)}")
print(f"Rows with complete metric data: {data_clean.dropna(subset=['pH', 'Conductivity', 'Turbidity', 'Temperature_C']).shape[0]}")
print()


data_imputed = data_clean.copy()
metrics = ["pH", "Conductivity", "Turbidity", "Temperature_C"]

for metric in metrics:
    data_imputed[metric] = data_imputed.groupby("Contaminant")[metric].transform(
        lambda x: x.fillna(x.mean())
    )

still_missing = data_imputed[data_imputed[metrics].isna().any(axis=1)]
if len(still_missing) > 0:
    print("=== ROWS THAT CANNOT BE IMPUTED (no group data) ===")
    print(still_missing[["Contaminant", "Amount"]])
    print()


data_imputed = data_imputed.dropna(subset=metrics)
data_imputed = data_imputed.reset_index(drop=True)

print("=== AFTER IMPUTATION ===")
print(f"Total samples: {len(data_imputed)}")
print(f"Samples per class:\n{data_imputed['Contaminant'].value_counts()}")
print()
print("=== IMPUTED DATA ===")
print(data_imputed)
print()


X = data_imputed[["pH", "Conductivity", "Turbidity", "Temperature_C"]]
y = data_imputed["Contaminant"]


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

loo = LeaveOneOut()

models = {
    "XGBoost": XGBClassifier(
        n_estimators=50,
        max_depth=2,
        learning_rate=0.1,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss'
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=3,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=50,
        max_depth=2,
        learning_rate=0.1,
        random_state=42
    ),
    "KNN (k=2)": KNeighborsClassifier(n_neighbors=2),
    "KNN (k=3)": KNeighborsClassifier(n_neighbors=3),
    "SVM (RBF)": SVC(kernel='rbf', C=1.0, random_state=42),
    "SVM (Linear)": SVC(kernel='linear', C=1.0, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=42)
}

print("=== MODEL COMPARISON (Loo CV) ===")
print("-" * 45)

results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_scaled, y_encoded, cv=loo)
    accuracy = scores.mean()
    results[name] = accuracy
    correct = int(accuracy * len(y_encoded))
    print(f"{name:25} Accuracy: {accuracy:.2f} ({correct}/{len(y_encoded)} correct)")

best_model_name = max(results, key=results.get)
best_accuracy = results[best_model_name]
print("-" * 45)
print(f"Best model: {best_model_name} ({best_accuracy:.2f})")
print()

best_model = models[best_model_name]
y_pred = cross_val_predict(best_model, X_scaled, y_encoded, cv=loo)

print(f"=== DETAILED RESULTS FOR {best_model_name.upper()} ===")
print("\nClassification Report:")
print(classification_report(y_encoded, y_pred,
                            target_names=label_encoder.classes_,
                            zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_encoded, y_pred))

best_model.fit(X_scaled, y_encoded)


if hasattr(best_model, 'feature_importances_'):
    print("\nFeature Importance:")
    for feature, importance in zip(X.columns, best_model.feature_importances_):
        print(f"  {feature}: {importance:.3f}")


print("\n=== EXAMPLE PREDICTION ===")
new_sample = [[7.0, 60.0, 30.0, 18.5]]
new_sample_scaled = scaler.transform(new_sample)
prediction = best_model.predict(new_sample_scaled)
print(f"Input: pH=7.0, Conductivity=60, Turbidity=30, Temp=18.5")
print(f"Predicted contaminant: {label_encoder.inverse_transform(prediction)[0]}")
