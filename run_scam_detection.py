import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("SCAM DETECTION IN DATING APPS - MODEL TRAINING & EVALUATION")
print("="*70)

# 1. Load the Data
print("\n[1/8] Loading data...")
df = pd.read_csv('dating_app_behavior_dataset_extended1.csv')
print(f"✓ Dataset loaded: {df.shape[0]:,} records, {df.shape[1]} features")

# 2. FEATURE ENGINEERING: Message Velocity
print("\n[2/8] Engineering Message_Velocity feature...")
df['Message_Velocity'] = df['message_sent_count'] / (df['app_usage_time_min'] + 1)
print(f"✓ New feature created")
print(f"   Message_Velocity statistics:")
print(f"   - Mean: {df['Message_Velocity'].mean():.4f}")
print(f"   - Std Dev: {df['Message_Velocity'].std():.4f}")
print(f"   - Min: {df['Message_Velocity'].min():.4f}")
print(f"   - Max: {df['Message_Velocity'].max():.4f}")

# 3. TARGET RE-MAPPING
print("\n[3/8] Creating binary target variable (Is_Scam_Suspect)...")
predatory_outcomes = ['Catfished', 'Blocked', 'Ghosted']
df['Is_Scam_Suspect'] = df['match_outcome'].isin(predatory_outcomes).astype(int)

target_counts = df['Is_Scam_Suspect'].value_counts()
print(f"✓ Binary target created")
print(f"   - Class 0 (Legitimate):   {target_counts[0]:,} ({target_counts[0]/len(df)*100:.1f}%)")
print(f"   - Class 1 (Scam Suspect): {target_counts[1]:,} ({target_counts[1]/len(df)*100:.1f}%)")

# 4. FEATURE SELECTION
print("\n[4/8] Selecting behavioral features...")
selected_features = [
    'Message_Velocity', 
    'app_usage_time_min', 
    'message_sent_count', 
    'bio_length', 
    'profile_pics_count', 
    'swipe_right_ratio'
]
print(f"✓ Selected {len(selected_features)} features:")
for i, feat in enumerate(selected_features, 1):
    print(f"   {i}. {feat}")

X = df[selected_features]
y = df['Is_Scam_Suspect']

# 5. Train-Test Split
print("\n[5/8] Splitting data into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"✓ Data split completed")
print(f"   - Training set: {X_train.shape[0]:,} records (70%)")
print(f"   - Test set: {X_test.shape[0]:,} records (30%)")

# 6. Train the Model
print("\n[6/8] Training Random Forest Classifier...")
model = RandomForestClassifier(random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print(f"✓ Model training complete")
print(f"   - Algorithm: Random Forest")
print(f"   - Number of trees: {model.n_estimators}")

# 7. Make Predictions
print("\n[7/8] Evaluating model on test set...")
preds = model.predict(X_test)
preds_proba = model.predict_proba(X_test)[:, 1]

# 8. Calculate Metrics
print("\n[8/8] Computing performance metrics...")
accuracy = accuracy_score(y_test, preds)
roc_auc = roc_auc_score(y_test, preds_proba)

print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
print(f"\n🎯 ACCURACY: {accuracy * 100:.2f}%")
print(f"   ROC-AUC Score: {roc_auc:.4f}")

print("\n" + "-"*70)
print("CLASSIFICATION REPORT")
print("-"*70)
print(classification_report(y_test, preds, 
                          target_names=['Legitimate', 'Scam Suspect'],
                          digits=4))

print("-"*70)
print("CONFUSION MATRIX")
print("-"*70)
cm = confusion_matrix(y_test, preds)
print("\n                 Predicted")
print("              Legit    Scam")
print(f"Actual Legit  {cm[0,0]:6d}   {cm[0,1]:6d}")
print(f"       Scam   {cm[1,0]:6d}   {cm[1,1]:6d}")

print("\n" + "-"*70)
print("INTERPRETATION")
print("-"*70)
tn, fp, fn, tp = cm.ravel()
print(f"True Negatives (Correct Legitimate): {tn:,}")
print(f"False Positives (False Alarms): {fp:,}")
print(f"False Negatives (Missed Scams): {fn:,}")
print(f"True Positives (Caught Scams): {tp:,}")

print("\n" + "-"*70)
print("FEATURE IMPORTANCE")
print("-"*70)
feature_importance = pd.DataFrame({
    'Feature': selected_features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nRanking (by predictive importance):")
for idx, row in feature_importance.iterrows():
    print(f"  {idx+1}. {row['Feature']:<25s}: {row['Importance']:.4f}")

# Visualization
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Feature Importance
ax1 = axes[0]
colors = plt.cm.viridis(np.linspace(0, 1, len(feature_importance)))
bars = ax1.barh(feature_importance['Feature'], feature_importance['Importance'], color=colors)
ax1.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
ax1.set_title('Feature Importance for Scam Detection', fontsize=12, fontweight='bold')
ax1.invert_yaxis()
for i, (v) in enumerate(feature_importance['Importance']):
    ax1.text(v + 0.002, i, f'{v:.4f}', va='center', fontsize=10)

# Plot 2: Confusion Matrix Heatmap
ax2 = axes[1]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2, 
            xticklabels=['Legitimate', 'Scam'],
            yticklabels=['Legitimate', 'Scam'],
            cbar_kws={'label': 'Count'})
ax2.set_ylabel('True Label', fontsize=11, fontweight='bold')
ax2.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
ax2.set_title('Confusion Matrix', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('scam_detection_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved as 'scam_detection_analysis.png'")

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)
