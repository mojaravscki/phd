import os
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
# Removed LightGBM import
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Additional imports for Bayesian, MLP, and Logistic Regression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

# For parallel processing
from joblib import Parallel, delayed

# =============================================================================
# Configuration
# =============================================================================
pathsname = "aug"  # or "no_aug"
folders = ["h1_asis", "h2_warp", "h3_ahe", "h7_combined"]  # Example for no_aug
names = ["train", "valid", "test"]
sizes = 32
color_mode = "LAB"  # Could also be "RGB", "H", etc.

# Base output folder where CSVs are stored
base_output = "/Users/macbookpro/Documents/Projects/phd/roboflow/output"

# -----------------------------------------------------------------------------
# Feature extraction methods
# -----------------------------------------------------------------------------
def extract_color_histogram(df):
    """
    Example: Flatten all columns in each row.
    Adjust logic as needed if your CSV is already storing certain feature columns.
    """
    return df.apply(lambda row: pd.Series(row.to_numpy().flatten()), axis=1)

def extract_color_moments(df):
    """
    Example: Store mean and variance across columns (this might be simplistic).
    """
    return df.apply(lambda row: pd.Series([row.mean(), row.var()]), axis=1)

def extract_bag_of_colors(df, n_colors=8):
    """
    Example: Take only the first n_colors columns.
    """
    return df.apply(lambda row: pd.Series(row.to_numpy().flatten()[:n_colors]), axis=1)


# -----------------------------------------------------------------------------
# Models to evaluate (8 models total)
# -----------------------------------------------------------------------------
models = {
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(n_jobs=1),  # Bagging
    "ExtraTrees": ExtraTreesClassifier(n_jobs=1),      # Bagging
    "SVM": SVC(),                                      # SVM (no n_jobs param)
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_jobs=1),
    "BayesianNB": GaussianNB(),                        # Naive Bayes
    "MLP": MLPClassifier(),                            # Multilayer Perceptron
    "LogisticRegression": LogisticRegression(n_jobs=1) # LR
}

# -----------------------------------------------------------------------------
# Feature selection methods
# -----------------------------------------------------------------------------
feature_methods = {
    "color_histogram": extract_color_histogram,
    "color_moments": extract_color_moments,
    "bag_of_colors": extract_bag_of_colors,
}

# =============================================================================
# Function to train and evaluate a single model (used in Parallel)
# =============================================================================
def fit_and_evaluate_model(model_name, model, X_train, y_train, X_test, y_test,
                           base_output, pathsname, folder, color_mode, sizes, method_name):
    """
    Trains a given model, evaluates on test data, and saves the classification
    report and confusion matrix to disk.
    """
    print(f"  -> Training {model_name}...")

    # Train the model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Generate classification report
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_text = classification_report(y_test, y_pred, output_dict=False)
    print(f"  -> {model_name} report:")
    print(report_text)

    # Save classification report
    classification_report_df = pd.DataFrame(report_dict).transpose()
    report_path = os.path.join(
        base_output,
        pathsname,
        folder,
        f"{color_mode}_{sizes}x{sizes}_{method_name}_{model_name}_classification_report.csv"
    )
    classification_report_df.to_csv(report_path, index=True)
    print(f"  -> Classification report saved to: {report_path}")

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Save confusion matrix as CSV
    conf_matrix_path = os.path.join(
        base_output,
        pathsname,
        folder,
        f"{color_mode}_{sizes}x{sizes}_{method_name}_{model_name}_confusion_matrix.csv"
    )
    np.savetxt(conf_matrix_path, conf_matrix, delimiter=",", fmt='%d')
    print(f"  -> Confusion matrix saved to: {conf_matrix_path}")

    # Save confusion matrix as a heatmap
    heatmap_path = os.path.join(
        base_output,
        pathsname,
        folder,
        f"{color_mode}_{sizes}x{sizes}_{method_name}_{model_name}_confusion_matrix_heatmap.png"
    )
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="YlGnBu",
                xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title(f"Confusion Matrix: {model_name} ({method_name})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(heatmap_path)
    plt.close()
    print(f"  -> Confusion matrix heatmap saved to: {heatmap_path}")
    print("-" * 70)


# =============================================================================
# Main Loop: For each folder, read 'train', 'valid', 'test' CSVs and evaluate
# =============================================================================
for folder in folders:
    print(f"\n=== Processing folder: {folder} ===")

    # Build CSV paths for train, valid, and test
    train_csv = os.path.join(
        base_output,
        pathsname,
        folder,
        f"{folder}_train_{pathsname}_{color_mode}_{sizes}x{sizes}.csv"
    )
    valid_csv = os.path.join(
        base_output,
        pathsname,
        folder,
        f"{folder}_valid_{pathsname}_{color_mode}_{sizes}x{sizes}.csv"
    )
    test_csv = os.path.join(
        base_output,
        pathsname,
        folder,
        f"{folder}_test_{pathsname}_{color_mode}_{sizes}x{sizes}.csv"
    )

    # Check if files exist
    if not all(os.path.exists(p) for p in [train_csv, valid_csv, test_csv]):
        print(f"One or more CSV files not found for folder '{folder}'. Skipping.")
        continue

    # Load datasets
    train_df = pd.read_csv(train_csv)
    valid_df = pd.read_csv(valid_csv)
    test_df  = pd.read_csv(test_csv)

    # Combine train + valid
    full_train_df = pd.concat([train_df, valid_df], ignore_index=True)

    # Separate features and labels (assuming these column names exist)
    X_train = full_train_df.drop(columns=['image_name', 'class_number']).values
    y_train = full_train_df['class_number'].values

    X_test = test_df.drop(columns=['image_name', 'class_number']).values
    y_test = test_df['class_number'].values

    # For each feature selection method, evaluate all models
    for method_name, method_func in feature_methods.items():
        print(f"\nEvaluating with '{method_name}' features...")

        # Extract features
        X_train_selected = method_func(pd.DataFrame(X_train))
        X_test_selected = method_func(pd.DataFrame(X_test))

        # ---------------------------------------------------------------------
        # Run each model in parallel so each model can utilize one CPU core
        # ---------------------------------------------------------------------
        Parallel(n_jobs=8, backend="loky")(
            delayed(fit_and_evaluate_model)(
                model_name, 
                model, 
                X_train_selected, 
                y_train, 
                X_test_selected, 
                y_test,
                base_output,
                pathsname,
                folder,
                color_mode,
                sizes,
                method_name
            )
            for model_name, model in models.items()
        )

    print(f"Finished processing folder: {folder}")

print("\nEvaluation complete for all folders, dataset partitions, and feature methods.")
