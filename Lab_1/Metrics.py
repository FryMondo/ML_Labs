import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, log_loss,
                             roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix)

DATA_PATH = "bioresponse.csv"
TARGET_COL = "Activity"
EPS = 1e-6


def evaluate_thresholded(y_true, y_proba, threshold: float = 0.5, eps: float = EPS):
    y_proba = np.clip(np.asarray(y_proba, dtype=float), eps, 1.0 - eps)
    y_pred = (y_proba >= float(threshold)).astype(int)
    tn, fp, fn, tp = map(int, confusion_matrix(y_true, y_pred).ravel())

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "log_loss": float(log_loss(y_true, y_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def select_threshold_min_fn(y_true, y_proba, grid_size: int = 2001, eps: float = EPS):
    y_proba = np.clip(np.asarray(y_proba, dtype=float), eps, 1.0 - eps)
    thresholds = np.linspace(0.0, 1.0, int(grid_size))
    best_key = None
    best_metrics = None

    for t in thresholds:
        m = evaluate_thresholded(y_true, y_proba, threshold=float(t), eps=eps)
        key = (m["fn"], m["fp"], -m["recall"])
        if best_key is None or key < best_key:
            best_key = key
            best_metrics = m

    return float(best_metrics["threshold"]), best_metrics


def main():
    df = pd.read_csv(DATA_PATH)
    assert TARGET_COL in df.columns, f"Column not found '{TARGET_COL}'"
    y = df[TARGET_COL].astype(int).values
    X = df.drop(columns=[TARGET_COL]).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    models = [
        ("DecisionTree(max_depth=3)", DecisionTreeClassifier(max_depth=3, random_state=42)),
        ("DecisionTree(unlimited)", DecisionTreeClassifier(random_state=42)),
        ("RandomForest(max_depth=3)",
         RandomForestClassifier(n_estimators=300, max_depth=3, random_state=42, n_jobs=-1)),
        ("RandomForest(unlimited)", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
    ]

    results = []
    probes = {}
    for name, model in models:
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        probes[name] = y_proba
        res = evaluate_thresholded(y_test, y_proba, 0.5)
        res["model"] = name
        results.append(res)

    metrics_df = pd.DataFrame(results).set_index("model").sort_values("roc_auc", ascending=False)
    print("Model metrics (threshold 0.5)")
    print(metrics_df[["accuracy", "precision", "recall", "f1", "log_loss"]])
    print("==================================================================")
    print("\nModel metrics (threshold 0.5)")
    print(metrics_df[["roc_auc", "tp", "fp", "fn", "tn"]])

    plt.figure()
    for name, y_proba in probes.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curves")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure()
    for name, y_proba in probes.items():
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        plt.plot(recall, precision, label=name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curves")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    rf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1).fit(X_train, y_train)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]
    best_t, best_m = select_threshold_min_fn(y_test, y_proba_rf, grid_size=2001)
    base_m = evaluate_thresholded(y_test, y_proba_rf, threshold=0.5)

    print("\nSelected threshold to minimize FN")
    print(f"Best threshold: {best_t:.4f}")
    print("\nMetrics at threshold 0.5:")
    print({k: v for k, v in base_m.items() if k != 'threshold'})
    print("\nMetrics at the selected threshold (min FN):")
    print({k: v for k, v in best_m.items() if k != 'threshold'})


main()
