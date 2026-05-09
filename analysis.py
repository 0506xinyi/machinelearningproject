import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score

def run_shootout_and_diagnose(input_filename):
    print(f"Loading noisy, realistic data from {input_filename}...")
    df = pd.read_csv(input_filename)
    
    # 1. Feature Selection and Target
    X = df.drop('Is_Top_Tier', axis=1)
    y = df['Is_Top_Tier']
    
    # 2. Strict 70/30 Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 3. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Initialize the Competitors (Notice the pruned parameters to fight overfitting!)
    models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', random_state=42),
        "Random Forest (Pruned)": RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=15, class_weight='balanced', random_state=42),
        "Gradient Boosting (Pruned)": GradientBoostingClassifier(n_estimators=100, max_depth=4, min_samples_leaf=15, random_state=42)
    }
    
    STRICT_THRESHOLD = 0.80
    results = []
    
    print("\n==========================================================")
    print("🩺 PHASE 1: OVERFITTING DIAGNOSIS (Train vs. Test)")
    print("==========================================================")
    
    # 5. Train, Diagnose, and Evaluate Each Model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        
        # --- THE OVERFITTING CHECK ---
        train_preds = model.predict(X_train_scaled)
        train_acc = accuracy_score(y_train, train_preds)
        
        test_preds = model.predict(X_test_scaled)
        test_acc = accuracy_score(y_test, test_preds)
        
        print(f"   -> Training Accuracy (Memorized): {train_acc * 100:.2f}%")
        print(f"   -> Testing Accuracy  (Unseen):    {test_acc * 100:.2f}%")
        
        if (train_acc - test_acc) > 0.05:
            print("   ⚠️ WARNING: Model might still be overfitting slightly!")
        else:
            print("   ✅ GOOD FIT: Model is generalizing well to new data.")
        
        # --- VIP FEED BUSINESS LOGIC ---
        probabilities = model.predict_proba(X_test_scaled)[:, 1]
        vip_predictions = (probabilities >= STRICT_THRESHOLD).astype(int)
        
        # Calculate Final Business Metrics
        prec = precision_score(y_test, vip_predictions, zero_division=0)
        rec = recall_score(y_test, vip_predictions)
        
        results.append({
            "Model": name,
            "Precision": prec * 100,
            "Recall": rec * 100
        })
    
    # 6. Display the Final Comparison Table
    print("\n==========================================================")
    print(f"🏆 PHASE 2: FINAL VIP LEADERBOARD (Threshold = {STRICT_THRESHOLD})")
    print("==========================================================")
    print(f"{'Model Name':<30} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 56)
    
    # Sort results by highest Precision
    results = sorted(results, key=lambda x: x['Precision'], reverse=True)
    
    for r in results:
        print(f"{r['Model']:<30} | {r['Precision']:>5.2f}%    | {r['Recall']:>5.2f}%")

if __name__ == "__main__":
    run_shootout_and_diagnose('processed_desirability_data.csv')