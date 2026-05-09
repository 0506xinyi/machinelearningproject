import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import chi2_contingency, spearmanr, pearsonr
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('dating_app_behavior_dataset_cleaned.csv')

print("=" * 80)
print("FEATURE SELECTION ANALYSIS - DATING APP BEHAVIOR DATASET")
print("=" * 80)
print(f"\nDataset shape: {df.shape}")
print(f"Target variable: 'match_outcome'")
print(f"\nTarget class distribution:")
print(df['match_outcome'].value_counts())

# Separate features and target
X = df.drop('match_outcome', axis=1)
y = df['match_outcome']

# Identify feature types
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

print(f"\n{'='*80}")
print("FEATURE TYPES")
print(f"{'='*80}")
print(f"\nNumerical features ({len(numerical_cols)}): {numerical_cols}")
print(f"\nCategorical features ({len(categorical_cols)}): {categorical_cols}")

# ============================================================================
# 1. CORRELATION ANALYSIS (Numerical Features)
# ============================================================================
print(f"\n{'='*80}")
print("1. CORRELATION ANALYSIS (Numerical Features)")
print(f"{'='*80}")

# Encode target for correlation
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

# Calculate correlations with target
correlations = {}
for col in numerical_cols:
    corr, p_value = spearmanr(X[col], y_encoded)
    correlations[col] = {'correlation': abs(corr), 'p_value': p_value}

# Sort by correlation
corr_df = pd.DataFrame(correlations).T.sort_values('correlation', ascending=False)
print("\nNumerical Features Correlation with Target (Spearman):")
print(corr_df)

# ============================================================================
# 2. MUTUAL INFORMATION & FEATURE IMPORTANCE
# ============================================================================
print(f"\n{'='*80}")
print("2. FEATURE IMPORTANCE ANALYSIS (Random Forest)")
print(f"{'='*80}")

# Encode all features
X_encoded = X.copy()
le_dict = {}

for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X[col])
    le_dict[col] = le

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_scaled, y_encoded)

# Get feature importances
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance Scores:")
print(feature_importance.to_string(index=False))

print("\nTop 10 Most Important Features:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"  {idx+1}. {row['feature']}: {row['importance']:.4f}")

# ============================================================================
# 3. CATEGORICAL FEATURES - CHI-SQUARE TEST
# ============================================================================
print(f"\n{'='*80}")
print("3. CATEGORICAL FEATURES - CHI-SQUARE TEST")
print(f"{'='*80}")

chi2_results = {}
for col in categorical_cols:
    contingency_table = pd.crosstab(X[col], y)
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    chi2_results[col] = {'chi2_stat': chi2, 'p_value': p_value, 'dof': dof}

chi2_df = pd.DataFrame(chi2_results).T.sort_values('chi2_stat', ascending=False)
print("\nChi-Square Test Results (Categorical Features):")
print(chi2_df)

# ============================================================================
# 4. KEY RELATIONSHIPS & INSIGHTS
# ============================================================================
print(f"\n{'='*80}")
print("4. KEY RELATIONSHIPS & INSIGHTS")
print(f"{'='*80}")

# Top numerical features
print("\n📊 TOP NUMERICAL FEATURES (Strong correlations):")
top_numerical = corr_df.head(5)
for idx, (feat, row) in enumerate(top_numerical.iterrows(), 1):
    print(f"  {idx}. {feat}")
    print(f"     - Correlation: {row['correlation']:.4f}")
    print(f"     - P-value: {row['p_value']:.4e}")
    print(f"     - Statistically significant: {'✓ Yes' if row['p_value'] < 0.05 else '✗ No'}")

# Top categorical features
print("\n📊 TOP CATEGORICAL FEATURES (Chi-Square significance):")
top_categorical = chi2_df.head(5)
for idx, (feat, row) in enumerate(top_categorical.iterrows(), 1):
    print(f"  {idx}. {feat}")
    print(f"     - Chi-Square stat: {row['chi2_stat']:.4f}")
    print(f"     - P-value: {row['p_value']:.4e}")
    print(f"     - Degrees of freedom: {row['dof']}")
    print(f"     - Statistically significant: {'✓ Yes' if row['p_value'] < 0.05 else '✗ No'}")

# ============================================================================
# 5. INTERESTING FEATURE INTERACTIONS
# ============================================================================
print(f"\n{'='*80}")
print("5. INTERESTING FEATURE RELATIONSHIPS")
print(f"{'='*80}")

# Numeric feature relationships
print("\n🔗 Top Feature Pairs (Numerical):")
numeric_corr_matrix = X[numerical_cols].corr()
# Get upper triangle
mask = np.triu(np.ones_like(numeric_corr_matrix, dtype=bool))
pairs = []
for i in range(len(numeric_corr_matrix.columns)):
    for j in range(i+1, len(numeric_corr_matrix.columns)):
        pairs.append({
            'feature1': numeric_corr_matrix.columns[i],
            'feature2': numeric_corr_matrix.columns[j],
            'correlation': numeric_corr_matrix.iloc[i, j]
        })
pairs_df = pd.DataFrame(pairs).sort_values('correlation', ascending=False, key=abs)
print(pairs_df.head(10).to_string(index=False))

# ============================================================================
# 6. RECOMMENDATIONS
# ============================================================================
print(f"\n{'='*80}")
print("6. FEATURE SELECTION RECOMMENDATIONS")
print(f"{'='*80}")

# High importance features
high_importance = feature_importance[feature_importance['importance'] > feature_importance['importance'].quantile(0.75)]
print(f"\n✅ RECOMMENDED FEATURES FOR MODEL (Top 25% by importance):")
for idx, row in high_importance.iterrows():
    print(f"  • {row['feature']}")

# Statistical significance
significant_categorical = chi2_df[chi2_df['p_value'] < 0.05]
print(f"\n✅ STATISTICALLY SIGNIFICANT CATEGORICAL FEATURES (p < 0.05):")
if len(significant_categorical) > 0:
    for feat in significant_categorical.index:
        print(f"  • {feat}")
else:
    print("  (None)")

significant_numerical = corr_df[corr_df['p_value'] < 0.05]
print(f"\n✅ STATISTICALLY SIGNIFICANT NUMERICAL FEATURES (p < 0.05):")
if len(significant_numerical) > 0:
    for feat in significant_numerical.index:
        print(f"  • {feat}")
else:
    print("  (None)")

# Low variance features to potentially remove
low_importance = feature_importance[feature_importance['importance'] < feature_importance['importance'].quantile(0.25)]
print(f"\n⚠️  LOW IMPORTANCE FEATURES (Consider removing):")
for idx, row in low_importance.iterrows():
    print(f"  • {row['feature']} (importance: {row['importance']:.4f})")

# ============================================================================
# 7. CREATE VISUALIZATIONS
# ============================================================================
print(f"\n{'='*80}")
print("7. GENERATING VISUALIZATIONS")
print(f"{'='*80}")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Feature Importance
ax1 = axes[0, 0]
top_features = feature_importance.head(15)
ax1.barh(range(len(top_features)), top_features['importance'])
ax1.set_yticks(range(len(top_features)))
ax1.set_yticklabels(top_features['feature'])
ax1.set_xlabel('Importance Score')
ax1.set_title('Top 15 Features by Importance', fontsize=12, fontweight='bold')
ax1.invert_yaxis()

# Plot 2: Correlation with Target
ax2 = axes[0, 1]
top_corr = corr_df.head(10)
colors = ['green' if x > 0 else 'red' for x in top_corr['correlation']]
ax2.barh(range(len(top_corr)), top_corr['correlation'], color=colors)
ax2.set_yticks(range(len(top_corr)))
ax2.set_yticklabels(top_corr.index)
ax2.set_xlabel('Absolute Correlation')
ax2.set_title('Top Numerical Features (Correlation with Target)', fontsize=12, fontweight='bold')
ax2.invert_yaxis()

# Plot 3: Chi-Square Results
ax3 = axes[1, 0]
top_chi2 = chi2_df.head(10)
ax3.barh(range(len(top_chi2)), top_chi2['chi2_stat'], color='steelblue')
ax3.set_yticks(range(len(top_chi2)))
ax3.set_yticklabels(top_chi2.index)
ax3.set_xlabel('Chi-Square Statistic')
ax3.set_title('Top Categorical Features (Chi-Square Test)', fontsize=12, fontweight='bold')
ax3.invert_yaxis()

# Plot 4: Distribution of Target Variable
ax4 = axes[1, 1]
target_counts = y.value_counts()
colors_pie = plt.cm.Set3(np.linspace(0, 1, len(target_counts)))
ax4.pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%', colors=colors_pie)
ax4.set_title('Distribution of Match Outcomes', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('feature_selection_analysis.png', dpi=300, bbox_inches='tight')
print("\n✅ Visualization saved as 'feature_selection_analysis.png'")

# ============================================================================
# 8. SAVE ANALYSIS RESULTS
# ============================================================================
print(f"\n{'='*80}")
print("8. SAVING ANALYSIS RESULTS")
print(f"{'='*80}")

# Save feature importance
feature_importance.to_csv('feature_importance.csv', index=False)
print("✅ Feature importance saved to 'feature_importance.csv'")

# Save correlation analysis
corr_df.to_csv('numerical_correlations.csv')
print("✅ Correlation analysis saved to 'numerical_correlations.csv'")

# Save chi-square results
chi2_df.to_csv('categorical_chi2_results.csv')
print("✅ Chi-square results saved to 'categorical_chi2_results.csv'")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE!")
print(f"{'='*80}\n")
