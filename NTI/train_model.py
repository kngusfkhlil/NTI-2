import pandas as pd
import numpy as np
import os
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
os.makedirs('plots', exist_ok=True)

# === Load dataset ===
dataset_path = "heart_cleveland_upload.csv"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"❌ Dataset file '{dataset_path}' not found.")

df = pd.read_csv(dataset_path)
print("🔎 Dataset columns:", df.columns.tolist())
print("🔎 Unique thal values:", df['thal'].unique())

# === Visualize class distribution before preprocessing ===
plt.figure(figsize=(6, 4))
df['condition'].value_counts(normalize=True).plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Class Distribution Before Preprocessing')
plt.xlabel('Condition (0 = No Disease, 1 = Disease)')
plt.ylabel('Proportion')
plt.xticks(rotation=0)
plt.savefig('plots/class_distribution_before.png', bbox_inches='tight')
plt.close()
print("✅ Saved: plots/class_distribution_before.png")
print("Class distribution before outlier removal:\n", df['condition'].value_counts(normalize=True))

# === Rename target column ===
if 'condition' not in df.columns:
    for col in ['HeartDisease', 'target', 'Heart Disease', 'output']:
        if col in df.columns:
            df.rename(columns={col: 'condition'}, inplace=True)
            break
    else:
        raise ValueError("❌ Target column not found. Expected 'HeartDisease', 'target', 'condition', or 'output'.")

print("✅ Dataset loaded. Shape:", df.shape)
print("🧼 Missing values:\n", df.isnull().sum())

# === Visualize numerical feature distributions ===
numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
for col in numerical_cols:
    if col in df.columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True, color='skyblue')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.savefig(f'plots/{col}_distribution.png', bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: plots/{col}_distribution.png")

# === Visualize correlation heatmap ===
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Features')
plt.savefig('plots/correlation_heatmap.png', bbox_inches='tight')
plt.close()
print("✅ Saved: plots/correlation_heatmap.png")

# === Remove outliers from numerical features ===
def remove_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

for col in numerical_cols:
    if col in df.columns:
        df = remove_outliers(df, col)

print("✅ Shape after outlier removal:", df.shape)
print("Class distribution after outlier removal:\n", df['condition'].value_counts(normalize=True))

# === Features and target ===
X = df.drop("condition", axis=1)
y = df["condition"]

# === One-hot encode categorical columns ===
cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
cat_cols = [col for col in cat_cols if col in X.columns]
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# === Save feature names for Streamlit input alignment ===
feature_names = X.columns.tolist()
joblib.dump(feature_names, "feature_names.pkl")
print("✅ Saved: feature_names.pkl")
print("🔎 Training feature names:", feature_names)

# === Scale the data ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.pkl")
print("✅ Saved: scaler.pkl")

# === Apply SMOTE to balance classes ===
smote = SMOTE(random_state=42)
X_scaled, y = smote.fit_resample(X_scaled, y)

# === Visualize class distribution after SMOTE ===
plt.figure(figsize=(6, 4))
pd.Series(y).value_counts(normalize=True).plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Class Distribution After SMOTE')
plt.xlabel('Condition (0 = No Disease, 1 = Disease)')
plt.ylabel('Proportion')
plt.xticks(rotation=0)
plt.savefig('plots/class_distribution_after_smote.png', bbox_inches='tight')
plt.close()
print("✅ Saved: plots/class_distribution_after_smote.png")
print("Class distribution after SMOTE:\n", pd.Series(y).value_counts(normalize=True))

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === Save high-risk test samples for validation ===
high_risk_samples = pd.DataFrame(X_test[y_test == 1], columns=feature_names)
high_risk_samples['true_label'] = 1
high_risk_samples.to_csv('high_risk_test_samples.csv', index=False)
print("✅ Saved: high_risk_test_samples.csv")

# === Define models with class weighting ===
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights = {0: class_weights[0], 1: class_weights[1]}

models = {
    "Logistic Regression": LogisticRegression(random_state=42, class_weight='balanced'),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVC": SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced'),
    "Decision Tree": DecisionTreeClassifier(criterion='entropy', random_state=42, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42, class_weight='balanced'),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, scale_pos_weight=class_weights[1]/class_weights[0]),
    "Naive Bayes": GaussianNB()
}

# === Create models directory ===
os.makedirs('models', exist_ok=True)

# === Train and evaluate models ===
model_metrics = []
print("\n🔍 Model Evaluation:")
for name, model in models.items():
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        high_risk_recall = report['1']['recall']
        print(f"\n✅ {name} Accuracy: {acc*100:.2f}%")
        print(classification_report(y_test, y_pred))
        
        # Save model
        model_filename = f"models/{name.replace(' ', '_').lower()}_model.pkl"
        joblib.dump(model, model_filename)
        print(f"✅ Saved: {model_filename}")
        
        # Test on high-risk samples
        if os.path.exists('high_risk_test_samples.csv'):
            high_risk_samples = pd.read_csv('high_risk_test_samples.csv')
            X_high_risk = high_risk_samples.drop('true_label', axis=1)
            high_risk_pred = model.predict(X_high_risk)
            high_risk_acc = (high_risk_pred == 1).mean()
            print(f"✅ {name} High-Risk Accuracy: {high_risk_acc*100:.2f}%")
        
        model_metrics.append({
            'Model': name,
            'Accuracy': acc,
            'High-Risk Recall': high_risk_recall
        })
        
        # Save feature importance for Random Forest
        if name == "Random Forest":
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)
            plt.figure(figsize=(8, 6))
            sns.barplot(x='Importance', y='Feature', data=feature_importance)
            plt.title('Random Forest Feature Importance')
            plt.savefig('plots/random_forest_feature_importance.png', bbox_inches='tight')
            plt.close()
            print("✅ Saved: plots/random_forest_feature_importance.png")
        
    except Exception as e:
        print(f"❌ Error with {name}: {e}")

# === Save model metrics for Streamlit ===
joblib.dump(model_metrics, "model_metrics.pkl")
print("✅ Saved: model_metrics.pkl")

# === Visualize model performance comparison ===
metrics_df = pd.DataFrame(model_metrics)
plt.figure(figsize=(10, 6))
metrics_df.set_index('Model')[['Accuracy', 'High-Risk Recall']].plot(kind='bar')
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.legend()
plt.savefig('plots/model_performance_comparison.png', bbox_inches='tight')
plt.close()
print("✅ Saved: plots/model_performance_comparison.png")

# === Find best model ===
best_model = None
best_model_name = ""
best_accuracy = 0.0
for metric in model_metrics:
    if metric['Accuracy'] > best_accuracy:
        best_model_name = metric['Model']
        best_accuracy = metric['Accuracy']

print(f"\n🏆 Best model: {best_model_name} ({best_accuracy*100:.2f}%)")
print("\n✅ All tasks completed. Files saved:")
print("   - scaler.pkl")
print("   - feature_names.pkl")
print("   - high_risk_test_samples.csv")
print("   - model_metrics.pkl")
print("   - models/ (all model files)")
print("   - plots/ (visualizations)")