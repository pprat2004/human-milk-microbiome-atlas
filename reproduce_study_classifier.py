"""
Exploratory MAMI study-source classifier.
Target is Study source only (a batch/source diagnostic), NOT country, disease or clinical outcome.
Input files expected:
  mother_breastMilk_abundance.csv
  mother_breastMilk_accession.txt
Requires pandas, numpy, scikit-learn.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report

ab = pd.read_csv("mother_breastMilk_abundance.csv")
meta = pd.read_csv("mother_breastMilk_accession.txt")
assert ab["sample"].astype(str).tolist() == meta["Run"].astype(str).tolist()

cols = ab.columns[3:].tolist()
X = ab[cols].astype(float).to_numpy()
nz = X.sum(axis=0) > 0
X = X[:, nz]
names = np.array(cols)[nz]
keep = np.array([x not in {"Mammalia", "Chloroplast", "Mitochondria"} for x in names])
X = X[:, keep]
y = meta["Study"].astype(str).to_numpy()

class PrevalenceCLR(BaseEstimator, TransformerMixin):
    def __init__(self, min_prevalence=0.10, delta=1e-5):
        self.min_prevalence = min_prevalence
        self.delta = delta
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.mask_ = (X > 0).mean(axis=0) >= self.min_prevalence
        if self.mask_.sum() == 0:
            self.mask_[np.argmax((X > 0).mean(axis=0))] = True
        return self
    def transform(self, X):
        X = np.asarray(X, dtype=float)[:, self.mask_].copy()
        X = X / X.sum(axis=1, keepdims=True)
        X[X == 0] = self.delta
        X = X / X.sum(axis=1, keepdims=True)
        L = np.log(X)
        return L - L.mean(axis=1, keepdims=True)

pipe = Pipeline([
    ("clr", PrevalenceCLR(min_prevalence=0.10, delta=1e-5)),
    ("scale", StandardScaler()),
    ("classifier", LogisticRegression(C=0.1, max_iter=5000,
                                      class_weight="balanced"))
])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
pred = cross_val_predict(pipe, X, y, cv=cv)
print("Balanced accuracy:", balanced_accuracy_score(y, pred))
print("Macro F1:", f1_score(y, pred, average="macro"))
print(classification_report(y, pred, zero_division=0))
