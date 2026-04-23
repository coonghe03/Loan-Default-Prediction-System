# src/models.py
# ONLY Member 2's Decision Tree Model

import joblib
from sklearn.tree import DecisionTreeClassifier

def train_decision_tree(X_train, y_train):
    """Train only Decision Tree model - Member 2 responsibility"""
    print("Training Decision Tree (Member 2 responsibility)...")
    
    dt_model = DecisionTreeClassifier(
        max_depth=10, 
        random_state=42,
        min_samples_split=10,   # Helps prevent overfitting
        min_samples_leaf=5
    )
    
    dt_model.fit(X_train, y_train)
    
    # Save your model
    joblib.dump(dt_model, '../models/decision_tree.pkl')
    print("✅ Decision Tree model trained and saved to models/decision_tree.pkl")
    
    return dt_model