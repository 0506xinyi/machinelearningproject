# Feature Selection Analysis - Dating App Behavior Dataset

## Executive Summary
Analysis of 50,000 dating app user records with 24 features across 10 match outcome categories.

---

## 🎯 TOP RECOMMENDED FEATURES FOR YOUR MODEL

### **Tier 1: Highest Priority Features** (Strong importance + Statistical significance)
1. **interest_tags** (Importance: 5.85%)
   - Most important categorical feature
   - User interests are highly predictive of match outcomes
   
2. **bio_length** (Importance: 5.73%)
   - Strong discriminative power
   - Correlates with engagement level
   
3. **weight_kg** (Importance: 5.68%)
   - Only numerically significant feature (p = 0.012)
   - Physical attributes matter in matching
   
4. **likes_received** (Importance: 5.54%)
   - Strongly indicates user appeal and engagement
   - Related to mutual_matches (correlation: 0.206)

### **Tier 2: Important Features** (High importance)
5. **app_usage_time_min** (Importance: 5.52%)
   - User commitment/activity level
   
6. **message_sent_count** (Importance: 5.41%)
   - Engagement metric, predicts interaction outcomes
   
7. **emoji_usage_rate** (Importance: 5.23%)
   - Communication style indicator
   
8. **swipe_right_ratio** (Importance: 5.21%)
   - User preference pattern

### **Tier 3: Supporting Features** (Good importance)
9. **age** (Importance: 5.01%)
10. **height_cm** (Importance: 5.00%)

---

## 📊 KEY RELATIONSHIPS IN YOUR DATA

### **Strong Feature Correlations**
- **height_cm ↔ weight_kg** (0.784): Physical attributes are strongly correlated
- **likes_received ↔ mutual_matches** (0.206): Attractiveness influences matching success

### **Statistically Significant Relationships**
| Feature | P-value | Significance |
|---------|---------|-------------|
| weight_kg | 0.0123 | ✓ YES (1.2%) |
| gender | 0.0102 | ✓ YES (1.0%) |

---

## ⚠️ FEATURES TO CONSIDER REMOVING

**Low Importance Features** (Bottom 6):
- swipe_right_label (1.43%) - Redundant with swipe_right_ratio
- app_usage_time_label (2.09%) - Redundant with app_usage_time_min
- swipe_time_of_day (3.00%)
- gender (3.02%)
- body_type (3.10%)
- relationship_intent (3.11%)

**Reason**: These features add little predictive value and may introduce noise.

---

## 💡 INSIGHTS FOR FEATURE SELECTION

### **What Matters Most:**
1. **User Profile Quality**
   - interest_tags + bio_length together indicate profile completeness
   - Better profiles get more matches

2. **User Activity & Engagement**
   - app_usage_time_min, message_sent_count, likes_received
   - Active users have better outcomes

3. **Communication Style**
   - emoji_usage_rate, message patterns
   - How users communicate affects results

4. **Physical Characteristics**
   - weight_kg, height_cm (correlated at 0.78)
   - Consider using only one to avoid multicollinearity

5. **Swiping Behavior**
   - swipe_right_ratio indicates user selectivity
   - Balance between pickiness and success

---

## 📋 RECOMMENDED FEATURE SETS FOR MODELING

### **Option 1: Minimal Feature Set (High Performance, Low Complexity)**
Use **Top 6 Features**:
```
1. interest_tags
2. bio_length
3. weight_kg
4. likes_received
5. app_usage_time_min
6. message_sent_count
```
**Best for**: Quick models, production systems

### **Option 2: Balanced Feature Set (Recommended for Production)**
Use **Top 10 Features**:
```
1. interest_tags
2. bio_length
3. weight_kg
4. likes_received
5. app_usage_time_min
6. message_sent_count
7. emoji_usage_rate
8. swipe_right_ratio
9. age
10. height_cm (or exclude due to correlation with weight_kg)
```
**Best for**: General-purpose models

### **Option 3: Full Feature Set (Maximum Information)**
Use **All except low-importance features**:
- Exclude: swipe_right_label, app_usage_time_label
- Keep: 22 features
**Best for**: Complex ensemble models

---

## 🚨 Data Quality Observations

1. **Multicollinearity Alert**
   - height_cm and weight_kg are highly correlated (0.78)
   - Consider using BMI instead or dropping one
   
2. **Balanced Classes**
   - Target variable is well-balanced across 10 outcomes
   - No class imbalance issues

3. **Low Statistical Significance**
   - Most numerical features have weak correlations with target
   - Suggests non-linear relationships
   - **Recommendation**: Use tree-based models (Random Forest, XGBoost)

---

## ✅ NEXT STEPS FOR IMPLEMENTATION

1. **Create training dataset** with recommended features
2. **Handle multicollinearity** (e.g., BMI instead of height+weight)
3. **Encode categorical variables** (interest_tags, gender, etc.)
4. **Consider feature engineering**:
   - BMI = weight_kg / (height_cm/100)²
   - Profile completeness score
   - Engagement score
5. **Use tree-based models** (Random Forest, XGBoost) for non-linear relationships
6. **Validate** with cross-validation on train/test sets

---

## 📁 Generated Files

- `feature_importance.csv` - Complete importance rankings
- `numerical_correlations.csv` - Correlation analysis
- `categorical_chi2_results.csv` - Statistical test results
- `feature_selection_analysis.png` - Visualization plots
