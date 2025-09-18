"""Kirollos Georgy - 40190279, Abdelaziz Mekkaoui - 40192247"""

import os
import time
from typing import Dict, Tuple

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import label_binarize

from report_generator import ReportGenerator


def get_data(filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Read the dataset from the given filepath
    and return the features (X) and target (y).
    Impute missing values using KNNImputer.

    Args:
        filepath (str): Path to the dataset file

    Returns:
        tuple: Features (X) and target (y)
    """
    df = pd.read_csv(filepath, usecols=range(32))

    # Columns with missing values
    columns_to_check = [
        "concavity_mean",
        "concave points_mean",
        "concavity_se",
        "concave points_se",
    ]

    # Use KNNImputer to fill missing values with the mean of the k-nearest neighbors
    imputer = KNNImputer(missing_values=0)
    df[columns_to_check] = imputer.fit_transform(df[columns_to_check])

    # Remove the first two columns (id and diagnosis) to keep only the features
    X = df.iloc[:, 2:]
    y = df["diagnosis"]

    return X, y


def get_best_parameters(
        model: RandomForestClassifier, X: pd.DataFrame, y: pd.Series
) -> Dict:
    """
    Perform grid search to find the best hyperparameters for the given model.

    Args:
        model (RandomForestClassifier): The machine learning model
        X (pd.DataFrame): Features
        y (pd.Series): Target

    Returns:
        Dict: Best hyperparameters
    """
    # Parameters to search
    param_grid = {
        "n_estimators": [50, 100],
        "max_features": ["sqrt", "log2"],
        "criterion": ["gini", "entropy"],
        "min_samples_split": [2, 5, 10],
        "max_depth": [10, 20, 50],
    }

    # Scoring criteria to use in the grid search to get the best parameters
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, pos_label="M"),
        "recall": make_scorer(recall_score, pos_label="M"),
        "f1": make_scorer(f1_score, pos_label="M"),
        "roc_auc": "roc_auc",
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=10,
        scoring=scoring,
        refit="recall",
        n_jobs=-1,
    )
    grid_search.fit(X, y)
    best_params = grid_search.best_params_

    return best_params


def evaluate_model(y_test: pd.Series, y_pred: pd.Series) -> Dict:
    """
    Evaluate the model using different metrics.

    Args:
        y_test (pd.Series): True labels
        y_pred (pd.Series): Predicted labels

    Returns:
        Dict: Evaluation metrics
    """
    accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
    precision = round(precision_score(y_test, y_pred, pos_label="M") * 100, 2)
    recall = round(recall_score(y_test, y_pred, pos_label="M") * 100, 2)
    f1 = round(f1_score(y_test, y_pred, pos_label="M") * 100, 2)
    roc_auc = round(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) * 100, 2)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }


def plot_confusion_matrix(y_test: pd.Series, y_pred: pd.Series, filename) -> None:
    """
    Plot the confusion matrix.

    Args:
        y_test (pd.Series): True labels
        y_pred (pd.Series): Predicted labels
    """
    cm = confusion_matrix(y_test, y_pred, labels=["M", "B"])
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["M", "B"],
        yticklabels=["M", "B"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(filename)
    plt.close()


def plot_roc_auc(y_test: pd.Series, y_proba: pd.Series, filename) -> None:
    """
    Plot the ROC-AUC curve.

    Args:
        y_test (pd.Series): True labels
        y_proba (pd.Series): Predicted probabilities for the positive class
    """
    # Convert labels to binary format for ROC curve computation
    y_test_binary = label_binarize(y_test, classes=["B", "M"])[:, 0]
    fpr, tpr, _ = roc_curve(y_test_binary, y_proba)
    plt.plot(
        fpr, tpr, label=f"ROC Curve (AUC = {roc_auc_score(y_test_binary, y_proba):.2f})"
    )
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-AUC Curve")
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    os.makedirs("report_data", exist_ok=True)
    X, y = get_data("data.csv")
    model = RandomForestClassifier(random_state=42)
    best_params = get_best_parameters(model, X, y)
    print("Best Parameters (refit recall):", best_params)

    report_generator = ReportGenerator("output.pdf")

    for N in [40, 140, 240, 340, 440]:
        T = N // 4
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=N, test_size=T, random_state=42
        )

        model = RandomForestClassifier(**best_params, random_state=42)
        model.fit(X_train, y_train)

        start_time = time.time()
        y_pred = model.predict(X_test)
        end_time = time.time()
        execution_time = round((end_time - start_time) * 1000, 4)
        metrics = evaluate_model(y_test, y_pred)
        print(f"N+T: {N+T}: (Execution time = {execution_time} milliseconds)", metrics)

        confusion_matrix_file = f"report_data/confusion_matrix_{N+T}.png"
        roc_auc_file = f"report_data/roc_auc_{N+T}.png"
        y_proba = model.predict_proba(X_test)[:, 1]
        plot_confusion_matrix(y_test, y_pred, confusion_matrix_file)
        plot_roc_auc(y_test, y_proba, roc_auc_file)
        section_heading = f"For train size = {N}, test size = {T}:"
        report_generator.write_section(
            metrics,
            confusion_matrix_file,
            roc_auc_file,
            section_heading=section_heading,
            execution_time=execution_time,
        )

    report_generator.generate_file()