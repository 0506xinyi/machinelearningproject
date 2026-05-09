# RomanceScam - Dating App Top-Tier User Prediction

## Problem Statement

Dating apps have millions of users, which makes it hard for people to find a great match. To help, apps create 'top-lists' to show the best profiles first. The problem is that it's very hard for a computer to guess who is actually a 'Top-Tier' user. If the AI makes a mistake and fills the top-list with average or boring profiles, the app looks bad and users get frustrated. We need a smart system that is extremely picky, ensuring that only the truly popular profiles are recommended.

## Research Questions

**Primary:** Can we build an AI that looks at profile effort and body metrics (BMI) to accurately pick out the top 20% of users without accidentally recommending the wrong people to the top-list?

**Supporting:**
1. Which matters more? Does putting effort into your bio and photos help more than your age or BMI when it comes to being popular?
2. Which AI is the smartest? Out of the three models we tested, which one is the best at being 'picky' so it doesn't make mistakes in our top-list?
3. Can it handle 'random' tastes? Can the AI still find the best profiles even when human dating tastes are messy and unpredictable?

## Objectives

1. **Organize the Data:** Create a clean list of features like "Profile Effort" and "BMI" from the raw user data.
2. **The Model Shootout:** Test three different AI 'brains' to see which one is the most accurate at predicting top-list status.
3. **Be Extra Picky:** Adjust the AI so it only recommends someone when it is 80% sure they belong in the top-list.
4. **Check the Work:** Make sure the AI is actually learning real patterns and hasn't just memorized the answers (the "Overfitting" check).

## Methodology

### 1. Data Cleaning & Feature Engineering (`clean_data.py`)

- **Input:** Raw dating app dataset (50,000 users, 24 features)
- **Steps:**
  - Duplicate removal and missing value handling
  - Text standardization (stripping whitespaces)
  - Outlier capping using IQR method on `app_usage_time_min`
  - Feature engineering:
    - `BMI` = weight_kg / (height_cm/100)^2
    - `Profile_Effort` = bio_length x profile_pics_count
  - Desirability logic injection: effort (60%) + healthy BMI (40%) -> likes_received
  - Target variable: `Is_Top_Tier` (top 20% by likes_received, with 15% label noise to simulate real-world unpredictability)
- **Output:** `processed_desirability_data.csv` with 9 features + target

### 2. Model Shootout & Diagnosis (`analysis.py`)

- **Models tested:**
  - Logistic Regression (Baseline, balanced class weights)
  - Random Forest (Pruned: max_depth=5, min_samples_leaf=15)
  - Gradient Boosting (Pruned: max_depth=4, min_samples_leaf=15)
- **Train/Test Split:** 70/30
- **Decision Threshold:** 0.80 (only recommend if 80% confident)
- **Overfitting Check:** Compare training vs. test accuracy (flag if gap > 5%)

### 3. Feature Selection Analysis (`feature_selection_analysis.py`)

- Random Forest feature importance ranking
- Numerical correlation analysis
- Chi-squared tests for categorical features
- See `FEATURE_SELECTION_SUMMARY.md` for detailed results

## Features Used

| Feature | Description |
|---------|-------------|
| `BMI` | Body Mass Index (engineered from height/weight) |
| `age` | User age |
| `Profile_Effort` | bio_length x profile_pics_count |
| `app_usage_time_min` | Daily app usage (IQR-capped) |
| `swipe_right_ratio` | Right-swipe proportion |
| `emoji_usage_rate` | Emoji usage in messages |
| `profile_pics_count` | Number of profile photos |
| `bio_length` | Biography text length |

## Setup

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Step 1: Clean data and generate processed dataset
python clean_data.py

# Step 2: Run model shootout with overfitting diagnosis
python analysis.py
```

## Dependencies

See [requirements.txt](requirements.txt) for full list. Key libraries:
- pandas, numpy - Data manipulation
- scikit-learn - Machine learning models and metrics
- imbalanced-learn - Class imbalance handling (SMOTE)
- matplotlib, seaborn - Visualization

## Repository Structure

```
RomanceScam/
├── clean_data.py                  # Data cleaning & feature engineering pipeline
├── analysis.py                    # Model shootout & overfitting diagnosis
├── feature_selection_analysis.py  # Feature importance & statistical analysis
├── FEATURE_SELECTION_SUMMARY.md   # Feature selection findings
├── requirements.txt               # Python dependencies
└── .gitignore
```
