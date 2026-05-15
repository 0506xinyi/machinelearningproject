import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier 
from xgboost import XGBClassifier 
from flaml import AutoML
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

THRESHOLD = 0.50

def run_shootout_and_diagnose(input_filename):
    print(f"Loading noisy, realistic data from {input_filename}...")
    df = pd.read_csv(input_filename)
    
    X = df.drop('Is_Top_Tier', axis=1)
    y = df['Is_Top_Tier']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=7, class_weight='balanced', random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=7, class_weight='balanced', random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=5, random_state=42, eval_metric='logloss')
    } 
    
    results = []
    
    print("\n")
    print("***PHASE 1: OVERFITTING & UNDERFITTING DIAGNOSIS (Train vs Test)***")
    print("")

    for name, model in models.items():
        print(f"\nTraining {name}...")

        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            train_data = X_train_scaled
            test_data = X_test_scaled
        else:
            model.fit(X_train, y_train)
            train_data = X_train
            test_data = X_test

        train_probs = model.predict_proba(train_data)[:, 1]
        test_probs = model.predict_proba(test_data)[:, 1]

        train_preds = (train_probs >= THRESHOLD).astype(int)
        test_preds = (test_probs >= THRESHOLD).astype(int)

        train_f1 = f1_score(y_train, train_preds, zero_division=0)
        test_f1 = f1_score(y_test, test_preds, zero_division=0)
        
        train_precision = precision_score(y_train, train_preds, zero_division=0)
        test_precision = precision_score(y_test, test_preds, zero_division=0)
        train_recall = recall_score(y_train, train_preds, zero_division=0)
        test_recall = recall_score(y_test, test_preds, zero_division=0)
        train_auc = roc_auc_score(y_train, train_probs)
        test_auc = roc_auc_score(y_test, test_probs)

        f1_gap = train_f1 - test_f1
        if train_f1 >= 0.85 and f1_gap >= 0.10:
            print(f"⚠️ OVERFITTING: Train F1 {train_f1*100:.1f}%, Test F1 {test_f1*100:.1f}% (Gap: {f1_gap*100:.1f}%)")
        elif train_f1 < 0.60 and test_f1 < 0.60:
            print(f"⚠️ UNDERFITTING: Train F1 {train_f1*100:.1f}%, Test F1 {test_f1*100:.1f}%")
        else:
            print(f"✅ GOOD GENERALIZATION: Train F1 {train_f1*100:.1f}%, Test F1 {test_f1*100:.1f}%")

        results.append({
            "Model": name,
            "Precision": test_precision * 100,
            "Recall": test_recall * 100,
            "F1 Score": test_f1 * 100,
            "ROC-AUC": test_auc * 100
        })

    print("\n")
    print(f"***PHASE 2: FINAL LEADERBOARD (Threshold = {THRESHOLD})***")
    print("")

    print(f"{'Model Name':<22} | {'Precision':>9} | {'Recall':>9} | {'F1 Score':>9} | {'ROC-AUC':>9}")
    print("-" * 70)
    
    results = sorted(results, key=lambda x: x['F1 Score'], reverse=True)
    
    for r in results:
        print(f"{r['Model']:<22} | {r['Precision']:>8.2f}% | {r['Recall']:>8.2f}% | {r['F1 Score']:>8.2f}% | {r['ROC-AUC']:>8.2f}%")


    best_model_name = results[0]['Model']
    print(f"\nThe Winner is: {best_model_name}!")
    return best_model_name

if __name__ == "__main__":

    champion_model = run_shootout_and_diagnose('processed_desirability_data.csv')

    print("\n")
    print(f"***PHASE 3: HYPERPARAMETER TUNING FOR [{champion_model.upper()}]***")
    print("")

    df = pd.read_csv('processed_desirability_data.csv')
    X = df.drop('Is_Top_Tier', axis=1)
    y = df['Is_Top_Tier']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    if champion_model == "Random Forest":
        base_model = RandomForestClassifier(class_weight='balanced', random_state=42)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_leaf': [1, 2, 5]
        }
        train_data_p3, test_data_p3 = X_train, X_test

    elif champion_model == "XGBoost":
        base_model = XGBClassifier(random_state=42, eval_metric='logloss')
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1]
        }
        train_data_p3, test_data_p3 = X_train, X_test

    elif champion_model == "Gradient Boosting":
        base_model = GradientBoostingClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 150, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'min_samples_leaf': [5, 10]
        }
        train_data_p3, test_data_p3 = X_train, X_test

    elif champion_model == "Decision Tree":
        base_model = DecisionTreeClassifier(class_weight='balanced', random_state=42)
        param_grid = {
            'max_depth': [5, 10, 15, None],
            'min_samples_leaf': [1, 5, 10]
        }
        train_data_p3, test_data_p3 = X_train, X_test

    elif champion_model == "Logistic Regression":
        base_model = LogisticRegression(class_weight='balanced', random_state=42)
        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0]
        }

        scaler = StandardScaler()
        train_data_p3 = scaler.fit_transform(X_train)
        test_data_p3 = scaler.transform(X_test)

    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=5,
        n_jobs=-1,
        scoring='f1'
    )

    print(f"Optimizing F1 Score for {champion_model}...")
    grid_search.fit(train_data_p3, y_train)

    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best Cross-Validated F1 Score: {grid_search.best_score_ * 100:.2f}%")

    best_model = grid_search.best_estimator_
    test_probs = best_model.predict_proba(test_data_p3)[:, 1]
    test_preds = (test_probs >= THRESHOLD).astype(int)

    final_precision = precision_score(y_test, test_preds, zero_division=0)
    final_recall = recall_score(y_test, test_preds, zero_division=0)
    final_f1 = f1_score(y_test, test_preds, zero_division=0)
    final_auc = roc_auc_score(y_test, test_probs)

    print("\n")
    print(f"***FINAL TEST PERFORMANCE ({champion_model} - BEST MODEL OUT OF THE FIVE)***")
    print(f"Precision : {final_precision * 100:.2f}%")
    print(f"Recall    : {final_recall * 100:.2f}%")
    print(f"F1 Score  : {final_f1 * 100:.2f}%")
    print(f"ROC-AUC   : {final_auc * 100:.2f}%")


    print("\n")
    print("***COMPARING MANUAL TUNING OF OUR MODEL WITH FLAML AUTOML BENCHMARK***")
    print("Initializing FLAML environment...")

    # 1. Prepare the exact same data split
    df = pd.read_csv('processed_desirability_data.csv')
    X = df.drop('Is_Top_Tier', axis=1)
    y = df['Is_Top_Tier']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # 2. Configure FLAML to optimize for F1-Score
    automl = AutoML()
    settings = {
        "time_budget": 60,      
        "metric": 'f1',          
        "task": 'classification',
        "estimator_list": ['rf', 'xgboost', 'extra_tree'], 
        "log_file_name": 'flaml_dating_app.log',
        "seed": 42,
        "verbose": 0             
    }

    # 3. Train
    automl.fit(X_train=X_train, y_train=y_train, **settings)

    # 4. Evaluate
    test_preds = automl.predict(X_test)
    test_probs = automl.predict_proba(X_test)[:, 1]

    print(f"\nBest Automated Model Found: {automl.best_estimator}")
    print(f"Precision : {precision_score(y_test, test_preds, zero_division=0) * 100:.2f}%")
    print(f"Recall    : {recall_score(y_test, test_preds, zero_division=0) * 100:.2f}%")
    print(f"F1 Score  : {f1_score(y_test, test_preds, zero_division=0) * 100:.2f}%")
    print(f"ROC-AUC   : {roc_auc_score(y_test, test_probs) * 100:.2f}%")