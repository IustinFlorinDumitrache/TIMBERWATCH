"""ML PASUL 4: antreneaza Random Forest pe candidatii etichetati si il evalueaza.

    python ml_step4_train.py

Iesire: data/ml/model.joblib + raport (precizie / recall) + importanta caracteristicilor
"""
import joblib
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import config
from ml_features import FEATURES


def main():
    cand = gpd.read_file(config.ML_CANDIDATI)
    lab = pd.read_csv(config.ML_ETICHETE)
    df = cand.merge(lab, on="cand_id")
    X = df[FEATURES].fillna(df[FEATURES].median())
    y = df["label"].astype(int)
    print(f"Exemple: {len(df)}  (taieri: {y.sum()}, alarme false: {(y == 0).sum()})")
    if y.nunique() < 2 or min(y.sum(), (y == 0).sum()) < 5:
        raise SystemExit("Prea putine exemple dintr-o clasa - etichetati mai multe (minim 5 din fiecare).")

    rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                class_weight="balanced", random_state=42, n_jobs=-1)
    folds = min(5, int(min(y.sum(), (y == 0).sum())))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    proba = cross_val_predict(rf, X, y, cv=cv, method="predict_proba")[:, 1]
    pred = (proba >= config.ML_PRAG_PROBABILITATE).astype(int)

    print(f"\n=== Validare incrucisata ({folds} fold-uri) ===")
    print(classification_report(y, pred, target_names=["alarma falsa", "taiere"], digits=2))
    print("Matrice de confuzie [rand = real, coloana = prezis]:")
    print(confusion_matrix(y, pred))

    rf.fit(X, y)
    joblib.dump({"model": rf, "features": FEATURES, "medians": X.median().to_dict()}, config.ML_MODEL)
    print(f"\nModel salvat -> {config.ML_MODEL}")

    print("\nImportanta caracteristicilor:")
    for imp, name in sorted(zip(rf.feature_importances_, FEATURES), reverse=True):
        print(f"  {name:15s} {imp:.3f}  {'#' * int(imp * 100)}")


if __name__ == "__main__":
    main()