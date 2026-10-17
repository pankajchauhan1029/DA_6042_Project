import argparse
import json
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from src.features import load_labeled_folder


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained image-forensics model on a separate test set.")
    parser.add_argument("--test_dir", required=True, help="Folder containing real/ and fake/ subfolders")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output", default="metrics.json")
    args = parser.parse_args()

    saved = joblib.load(args.model_path)
    X, y, paths = load_labeled_folder(args.test_dir, mode=saved["feature_mode"])
    probabilities = saved["model"].predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    result = {
        "test_count": int(len(y)),
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "confusion_matrix_labels_real_fake": confusion_matrix(y, predictions, labels=[0, 1]).tolist(),
        "roc_auc": float(roc_auc_score(y, probabilities)) if len(set(y)) == 2 else None,
        "average_precision": float(average_precision_score(y, probabilities)) if len(set(y)) == 2 else None,
        "feature_mode": saved["feature_mode"],
        "model_path": str(Path(args.model_path)),
        "test_directory": str(Path(args.test_dir)),
        "note": "Interpret only for this dataset and split; this result does not establish generalization to unseen generators.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
