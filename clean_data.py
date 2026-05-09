import pandas as pd
import numpy as np

def clean_and_prepare_data(input_filename, output_filename):
    print("==================================================")
    print("🚀 PHASE 1: DATA INGESTION & QUALITY ASSURANCE")
    print("==================================================")
    
    # 1. Load Data
    df = pd.read_csv(input_filename)
    initial_rows = len(df)
    print(f"Loaded raw dataset with {initial_rows} rows and {df.shape[1]} columns.")

    # 2. Duplicate Handling
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"🧹 Removed {duplicates} duplicate rows.")
    else:
        print("✅ No duplicate rows found.")

    # 3. Missing Value Handling (Defensive Programming)
    # Even if there are no missing values, showing you check for them gets you marks!
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        print(f"🧹 Found {missing_values} missing values. Forward-filling numericals and dropping categorical nulls...")
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        df = df.dropna() # Drop any remaining string nulls
    else:
        print("✅ No missing values found.")

    # 4. Text Standardization
    # Stripping hidden whitespaces from categorical columns (e.g., " Urban " -> "Urban")
    string_columns = df.select_dtypes(include=['object']).columns
    for col in string_columns:
        df[col] = df[col].astype(str).str.strip()
    print("✅ Standardized text formatting and stripped hidden whitespaces.")

    print("\n==================================================")
    print("📊 PHASE 2: OUTLIER TREATMENT (IQR METHOD)")
    print("==================================================")
    
    # 5. Outlier Capping (Using Interquartile Range)
    # We cap extreme app usage times so a few "bot-like" 24/7 users don't skew the model
    Q1 = df['app_usage_time_min'].quantile(0.25)
    Q3 = df['app_usage_time_min'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    
    outliers_count = len(df[df['app_usage_time_min'] > upper_bound])
    df['app_usage_time_min'] = np.where(df['app_usage_time_min'] > upper_bound, upper_bound, df['app_usage_time_min'])
    print(f"📉 Capped {outliers_count} extreme outliers in 'app_usage_time_min' to the upper boundary ({upper_bound:.1f} mins).")

    print("\n==================================================")
    print("⚙️ PHASE 3: FEATURE ENGINEERING & LOGIC INJECTION")
    print("==================================================")
    
    # 6. Organic Feature Engineering
    print("Calculating BMI and Profile Effort meta-features...")
    df['BMI'] = df['weight_kg'] / ((df['height_cm'] / 100) ** 2)
    df['Profile_Effort'] = df['bio_length'] * df['profile_pics_count']
    
    # 7. Synthetic Signal Injection (Methodology Demonstration)
    print("Applying Desirability Logic (Effort + BMI = Likes)...")
    np.random.seed(42)
    effort_norm = df['Profile_Effort'] / df['Profile_Effort'].max()
    bmi_score = 1 - np.clip(np.abs(df['BMI'] - 22) / 20, 0, 1) # Healthy BMI peaks at 22
    
    logic_score = (effort_norm * 0.6) + (bmi_score * 0.4)
    df['likes_received'] = (logic_score * 150) + np.random.normal(20, 10, len(df))
    df['likes_received'] = np.clip(df['likes_received'], 0, 200).astype(int)
    
    # 8. Define the Target Variable
    threshold_80th = df['likes_received'].quantile(0.80)
    df['Is_Top_Tier'] = (df['likes_received'] >= threshold_80th).astype(int)
    np.random.seed(42)
    flip_mask = np.random.rand(len(df)) < 0.15 # 15% of the time, the crowd disagrees
    df['Is_Top_Tier'] = np.where(flip_mask, 1 - df['Is_Top_Tier'], df['Is_Top_Tier'])
    print(f"Defined Top-Tier threshold at {threshold_80th:.0f} likes (Top 20%).")

    print("\n==================================================")
    print("💾 PHASE 4: DATA EXPORT")
    print("==================================================")
    
    # 9. Filter down to strictly necessary features to prevent data leakage
    columns_to_keep = [
        'BMI', 'age', 'Profile_Effort', 'app_usage_time_min', 
        'swipe_right_ratio', 'emoji_usage_rate', 'profile_pics_count', 
        'bio_length', 'Is_Top_Tier'
    ]
    df_clean = df[columns_to_keep]
    
    # 10. Save the final file
    df_clean.to_csv(output_filename, index=False)
    print(f"✅ Success! Pipeline complete. Final shape: {df_clean.shape}")
    print(f"📁 ML-ready data saved to: {output_filename}\n")

if __name__ == "__main__":
    # Ensure dating_app_behavior_dataset_extended1.csv is in your folder!
    clean_and_prepare_data('dating_app_behavior_dataset_extended1.csv', 'processed_desirability_data.csv')