"""
Treinamento LOCAL do modelo de churn prediction.

Treina XGBoost + RandomForest, compara, e exporta o melhor modelo.
Gera relatório com métricas, feature importance e confusion matrix.

Uso:
    python scripts/train_local.py
    python scripts/train_local.py --data data/training_data_sample.csv --output models/
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

# Opcional: XGBoost se instalado
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def load_data(csv_path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Carrega CSV e retorna X, y, feature_names."""
    import csv

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV vazio: {csv_path}")

    # Todas as colunas exceto 'label' são features
    feature_names = [k for k in rows[0].keys() if k != "label"]

    X = np.array([[float(row[f]) for f in feature_names] for row in rows])
    y = np.array([int(float(row["label"])) for row in rows])

    return X, y, feature_names


def train_and_evaluate(
    X_train, X_test, y_train, y_test, model, model_name: str
) -> dict:
    """Treina modelo e retorna métricas."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    return metrics


def print_results(metrics: dict, y_test, y_pred, feature_names, importances):
    """Imprime resultados formatados."""
    print(f"\n{'='*50}")
    print(f"  {metrics['model']}")
    print(f"{'='*50}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")

    print(f"\n  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                 Predicted")
    print(f"                 Ativo  Churn")
    print(f"  Real Ativo  [{cm[0][0]:5d}  {cm[0][1]:5d}]")
    print(f"  Real Churn  [{cm[1][0]:5d}  {cm[1][1]:5d}]")

    print(f"\n  Top 10 Features (importância):")
    sorted_idx = np.argsort(importances)[::-1]
    for i in range(min(10, len(feature_names))):
        idx = sorted_idx[i]
        print(f"    {i+1:2d}. {feature_names[idx]:<30s} {importances[idx]:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Treinar modelo de churn localmente")
    parser.add_argument("--data", default="data/training_data_sample.csv", help="CSV de treinamento")
    parser.add_argument("--output", default="models", help="Diretório para salvar modelo")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proporção do test set")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("=" * 60)
    print("🧠 Treinamento Local - Churn Prediction Model")
    print("=" * 60)
    start = time.time()

    # 1. Carregar dados
    print(f"\n📄 Carregando: {args.data}")
    X, y, feature_names = load_data(args.data)
    print(f"  Registros: {len(y)}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Churn: {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"  Ativo: {len(y) - y.sum()} ({(1-y.mean())*100:.1f}%)")

    # 2. Split stratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    print(f"\n📊 Split: {len(X_train)} treino / {len(X_test)} teste")

    # 3. Definir modelos
    models = {}

    # GradientBoosting (funciona como XGBoost mas nativo sklearn)
    models["GradientBoosting"] = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=args.seed,
    )

    # RandomForest
    models["RandomForest"] = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=args.seed,
    )

    # XGBoost (se disponível)
    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=len(y[y==0]) / max(len(y[y==1]), 1),
            random_state=args.seed,
            eval_metric="logloss",
            use_label_encoder=False,
        )

    # 4. Treinar e avaliar cada modelo
    print("\n🏋️ Treinando modelos...")
    best_model = None
    best_metrics = None
    best_name = None
    all_results = []

    for name, model in models.items():
        print(f"\n  → {name}...", end=" ")
        metrics = train_and_evaluate(X_train, X_test, y_train, y_test, model, name)
        print(f"F1={metrics['f1']:.4f}, AUC={metrics['roc_auc']:.4f}")

        y_pred = model.predict(X_test)
        importances = model.feature_importances_
        print_results(metrics, y_test, y_pred, feature_names, importances)

        all_results.append(metrics)

        if best_metrics is None or metrics["f1"] > best_metrics["f1"]:
            best_model = model
            best_metrics = metrics
            best_name = name

    # 5. Cross-validation do melhor modelo
    print(f"\n{'='*60}")
    print(f"🏆 Melhor modelo: {best_name} (F1={best_metrics['f1']:.4f})")
    print(f"{'='*60}")

    print(f"\n🔄 Cross-validation (5-fold) do {best_name}...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    cv_scores = cross_val_score(best_model, X, y, cv=cv, scoring="f1")
    print(f"  F1 scores: {cv_scores}")
    print(f"  F1 médio: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    cv_auc = cross_val_score(best_model, X, y, cv=cv, scoring="roc_auc")
    print(f"  AUC scores: {cv_auc}")
    print(f"  AUC médio: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

    # 6. Retreinar com todos os dados
    print(f"\n🔁 Retreinando {best_name} com TODOS os dados...")
    best_model.fit(X, y)

    # 7. Salvar modelo
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "churn_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"  💾 Modelo salvo: {model_path}")

    # Salvar metadados
    metadata = {
        "model_name": best_name,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.data,
        "dataset_size": len(y),
        "churn_count": int(y.sum()),
        "active_count": int(len(y) - y.sum()),
        "features": feature_names,
        "n_features": len(feature_names),
        "test_metrics": best_metrics,
        "cv_f1_mean": float(cv_scores.mean()),
        "cv_f1_std": float(cv_scores.std()),
        "cv_auc_mean": float(cv_auc.mean()),
        "cv_auc_std": float(cv_auc.std()),
        "feature_importance": {
            feature_names[i]: float(best_model.feature_importances_[i])
            for i in range(len(feature_names))
        },
    }

    meta_path = output_dir / "model_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  📋 Metadados: {meta_path}")

    # 8. Upload para S3
    print(f"\n☁️  Upload para S3...")
    try:
        import boto3
        s3 = boto3.client("s3", region_name="us-east-1")
        bucket = "sky-brazil-churn-prediction"

        # Upload modelo
        s3_model_key = "models/approved/churn_model.pkl"
        s3.upload_file(str(model_path), bucket, s3_model_key)
        print(f"  ✅ s3://{bucket}/{s3_model_key}")

        # Upload metadados
        s3_meta_key = "models/approved/model_metadata.json"
        s3.put_object(
            Bucket=bucket, Key=s3_meta_key,
            Body=json.dumps(metadata, indent=2, ensure_ascii=False),
            ContentType="application/json",
        )
        print(f"  ✅ s3://{bucket}/{s3_meta_key}")
    except Exception as e:
        print(f"  ⚠️  Upload falhou: {e}")
        print(f"     Modelo disponível em: {model_path}")

    elapsed = time.time() - start
    print(f"\n🏁 Concluído em {elapsed:.1f}s")
    print(f"\n📊 Resumo Final:")
    print(f"   Modelo: {best_name}")
    print(f"   F1 (test): {best_metrics['f1']:.4f}")
    print(f"   ROC AUC (test): {best_metrics['roc_auc']:.4f}")
    print(f"   F1 (CV 5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


if __name__ == "__main__":
    main()
