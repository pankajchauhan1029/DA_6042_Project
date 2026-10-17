import argparse
import json
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.features import load_labeled_folder


def main():
    parser = argparse.ArgumentParser(description="Train a lightweight real/fake image classifier.")
    parser.add_argument("--train_dir", required=True, help="Folder containing real/ and fake/ subfolders")
    parser.add_argument("--model_path", default="artifacts/freqcheck.joblib")
    parser.add_argument("--features", choices=["frequency", "all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X, y, paths = load_labeled_folder(args.train_dir, mode=args.features)
    indices = list(range(len(y)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=args.seed, stratify=y)
    model = RandomForestClassifier(n_estimators=250, min_samples_leaf=2, class_weight="balanced", random_state=args.seed, n_jobs=-1)
    model.fit(X[train_idx], y[train_idx])
    probabilities = model.predict_proba(X[val_idx])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    y_val = y[val_idx]
    metrics = {
        "validation_count": int(len(val_idx)),
        "accuracy": float(accuracy_score(y_val, predictions)),
        "f1": float(f1_score(y_val, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_val, probabilities)) if len(set(y_val)) == 2 else None,
        "average_precision": float(average_precision_score(y_val, probabilities)) if len(set(y_val)) == 2 else None,
    }
    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_mode": args.features, "feature_count": int(X.shape[1]), "seed": args.seed}, model_path)
    metrics_path = model_path.with_suffix(".validation.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"model_path": str(model_path), "feature_mode": args.features, "total_images": len(paths), "validation_metrics": metrics}, indent=2))
    print("Note: validation results are preliminary; use a separate test set for final reporting.")


if __name__ == "__main__":
    main()
