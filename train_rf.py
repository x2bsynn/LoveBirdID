# train_rf.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, LeaveOneOut
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

def train_random_forest():
    # 1. Load the data
    data_file = 'data/lovebird_weights.csv'
    if not os.path.exists(data_file):
        print(f"Error: Data file {data_file} not found!")
        return None
    
    df = pd.read_csv(data_file)
    print("Dataset Head:")
    print(df.head())
    print(f"\nDataset Size: {len(df)}")
    print(f"Class Counts:\n{df['sex'].value_counts()}")
    
    if len(df) < 4:
        print("Warning: Very small dataset! Results may be unreliable.")
        print("Consider collecting more data before training.")

    # 2. Encode the labels
    df['sex_encoded'] = df['sex'].map({'female': 0, 'male': 1})

    # 3. Define features (X) and target (y)
    X = df[['weight']]
    y = df['sex_encoded']

    # 4. Create the Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    # 5. Use Leave-One-Out Cross-Validation for small datasets
    print("\nPerforming Leave-One-Out Cross-Validation...")
    loo = LeaveOneOut()

    # Get cross-validation predictions and probabilities
    true_labels, pred_labels, probas = [], [], []
    for train_index, test_index in loo.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Train and predict
        rf_model.fit(X_train, y_train)
        pred = rf_model.predict(X_test)[0]
        proba = rf_model.predict_proba(X_test)[0]
        
        true_labels.append(y_test.values[0])
        pred_labels.append(pred)
        probas.append(proba)

    # 6. Evaluate the model
    accuracy = accuracy_score(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels, target_names=['female', 'male'])
    conf_matrix = confusion_matrix(true_labels, pred_labels)

    print("\n" + "="*50)
    print("RANDOM FOREST MODEL EVALUATION (Leave-One-Out CV)")
    print("="*50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:\n{report}")
    print(f"Confusion Matrix:\n{conf_matrix}")

    # 7. Train final model on ALL data for deployment
    print("\nTraining final model on entire dataset...")
    rf_model_final = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model_final.fit(X, y)

    # 8. Save the evaluation report
    os.makedirs('models', exist_ok=True)
    with open('models/rf_model_evaluation.txt', 'w') as f:
        f.write("Random Forest Model Evaluation (Leave-One-Out CV)\n")
        f.write("="*50 + "\n")
        f.write(f"Dataset Size: {len(df)}\n")
        f.write(f"Class Distribution:\n{df['sex'].value_counts()}\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write("Confusion Matrix:\n")
        f.write(str(conf_matrix) + "\n")

    # 9. Save the final trained model
    joblib.dump(rf_model_final, 'models/lovebird_rf_model.pkl')
    print("\nFinal model trained on all data and saved as 'models/lovebird_rf_model.pkl'")
    print("Evaluation report saved as 'models/rf_model_evaluation.txt'")

    # 10. Show sample predictions with probabilities
    print("\nSample Predictions with Confidence:")
    for i in range(min(5, len(df))):
        true_sex = 'female' if true_labels[i] == 0 else 'male'
        pred_sex = 'female' if pred_labels[i] == 0 else 'male'
        female_prob, male_prob = probas[i]
        status = "✓" if pred_sex == true_sex else "✗"
        print(f"{status} Sample {i+1}: True={true_sex}, Pred={pred_sex}, "
              f"P(female)={female_prob:.3f}, P(male)={male_prob:.3f}")
    
    return rf_model_final

if __name__ == "__main__":
    train_random_forest()