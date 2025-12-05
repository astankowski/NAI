"""
This code features classification problem made with decision trees and SVM (Support Vector Machines)

Datasets used for classification:
    1. Ionosphere Dataset: https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data
    2. Breast Cancer Wisconsin (Diagnostic): https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data

Authors:
- Aleksander Stankowski (s27549)
- Daniel Bieliński (s27292)

Environment Setup:
    1. Create a virtual environment:
       python3 -m venv venv
    2. Activate the virtual environment:
       source venv/bin/activate
    3. Install dependencies:
       pip install pandas numpy scikit-learn matplotlib seaborn
    4. Run the script:
       python3 4-ClassificationAI/tree_and_svm_classification.py

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ==========================================
# Configuration & Data Loading
# ==========================================

# Datasets object in which we store datasets urls and headers. These headers are empty as we're importing raw data.
DATASETS = {
    'Ionosphere': {
        'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data',
        'header': None
    },
    'Breast_Cancer': {
        'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',
        'header': None
    }
}

def load_data(name):
    """
    Loads and preprocesses the specific dataset.

    Args:
        name (str): The name of the dataset to load ('Ionosphere' or 'Breast_Cancer').

    Returns:
        tuple: A tuple containing:
            - X_train_scaled (numpy.ndarray): Scaled training features.
            - X_test_scaled (numpy.ndarray): Scaled testing features.
            - y_train (numpy.ndarray): Encoded training labels.
            - y_test (numpy.ndarray): Encoded testing labels.
            - le (LabelEncoder): Fitted label encoder.
            - feature_cols (pandas.Index): Feature names.

    Raises:
        ValueError: If an unknown dataset name is provided.
    """
    print(f"\n--- Loading {name} Dataset ---")
    url = DATASETS[name]['url']

    if name == 'Ionosphere':
        # Ionosphere: 34 numeric features, last column is class 'g' or 'b'
        df = pd.read_csv(url, header=None)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
    elif name == 'Breast_Cancer':
        # Breast Cancer WDbc: ID, Diagnosis (M/B), 30 features
        df = pd.read_csv(url, header=None)
        # Col 0 is ID (drop), Col 1 is Diagnosis (Target), Col 2-31 are features
        y = df.iloc[:, 1]
        X = df.iloc[:, 2:]
        
    else:
        raise ValueError("Unknown dataset")

    # Label encoding (e.g., M/B -> 1/0)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"Data loaded. Shape: {X.shape}")
    print(f"Classes: {le.classes_}")
    
    # Spliting data into training and testing sets (Training: 70%, Testing: 30%)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
    
    # Scaling features (Important for SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, le, X.columns

# ==========================================
# Classification & Analysis
# ==========================================

def train_decision_tree(X_train, X_test, y_train, y_test, class_names):
    """
    Trains a Decision Tree classifier and evaluates its performance.

    Args:
        X_train (numpy.ndarray): Training features.
        X_test (numpy.ndarray): Testing features.
        y_train (numpy.ndarray): Training labels.
        y_test (numpy.ndarray): Testing labels.
        class_names (list): List of class names for the classification report.

    Returns:
        DecisionTreeClassifier: The trained Decision Tree model.
    """
    print("\n[Decision Tree]")
    dt = DecisionTreeClassifier(random_state=42, max_depth=5) # Max depth to avoid overfitting on small data
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    return dt

def train_svm_with_kernels(X_train, X_test, y_train, y_test):
    """Trains SVM classifiers with different kernels and evaluates them.

    Iterates through 'linear', 'rbf', 'poly', and 'sigmoid' kernels, training
    an SVM for each and printing the accuracy. Returns the best performing model.

    Args:
        X_train (numpy.ndarray): Training features.
        X_test (numpy.ndarray): Testing features.
        y_train (numpy.ndarray): Training labels.
        y_test (numpy.ndarray): Testing labels.

    Returns:
        SVC: The best performing SVM model.
    """
    print("\n[SVM Analysis - Kernel Comparison]")
    
    kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    results = {}
    
    best_acc = 0
    best_model = None
    
    for k in kernels:
        print(f"\nTesting Kernel: {k.upper()}")
        svc = SVC(kernel=k, probability=True, random_state=42)
        svc.fit(X_train, y_train)
        y_pred = svc.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        results[k] = acc
        print(f"Accuracy: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_model = svc
            
    print("\n--- Kernel Summary ---")
    for k, acc in results.items():
        print(f"Kernel {k.ljust(10)}: {acc:.4f}")
        
    return best_model

def visualize_results(model, X_test, y_test, title, class_names):
    """
    Generates and saves a confusion matrix plot.

    Args:
        model (sklearn.base.BaseEstimator): The trained model to evaluate.
        X_test (numpy.ndarray): Testing features.
        y_test (numpy.ndarray): Testing labels.
        title (str): Title for the confusion matrix plot.
        class_names (list): List of class names for axis labels.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    print(f"Saved plot: {title.replace(' ', '_').lower()}.png")

# ==========================================
# Main Execution
# ==========================================

def run_analysis_for_dataset(name):
    """
    Runs the full analysis pipeline for a specified dataset.

    Loads data, trains a decision tree and SVMs, and generates visualizations.

    Args:
        name (str): The name of the dataset to analyze.
    """
    # 1. Load
    X_train, X_test, y_train, y_test, le, feature_cols = load_data(name)
    class_names = [str(c) for c in le.classes_]
    
    # 2. Decision Tree
    dt_model = train_decision_tree(X_train, X_test, y_train, y_test, class_names)
    visualize_results(dt_model, X_test, y_test, f"{name} - Decision Tree Confusion Matrix", class_names)
    
    # 3. SVM
    svm_model = train_svm_with_kernels(X_train, X_test, y_train, y_test)
    visualize_results(svm_model, X_test, y_test, f"{name} - Best SVM Confusion Matrix", class_names)
    
    # 4. Example Prediction
    print(f"\n[Example Prediction for {name}]")
    sample_input = X_test[0].reshape(1, -1) # Take first test sample
    dt_pred = le.inverse_transform(dt_model.predict(sample_input))
    svm_pred = le.inverse_transform(svm_model.predict(sample_input))
    true_label = le.inverse_transform([y_test[0]])
    
    print(f"Input Features (Subset): {sample_input[0][:5]}...")
    print(f"True Label: {true_label[0]}")
    print(f"DT Prediction: {dt_pred[0]}")
    print(f"SVM Prediction: {svm_pred[0]}")

if __name__ == "__main__":
    # Analysis for Dataset 1
    run_analysis_for_dataset('Ionosphere')
    
    print("\n" + "="*50 + "\n")
    
    # Analysis for Dataset 2
    run_analysis_for_dataset('Breast_Cancer')
