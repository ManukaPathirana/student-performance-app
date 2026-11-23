"""Data loading, feature metadata and model training for the
Student Academic Performance Prediction System."""
from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier, RandomForestClassifier,
                              StackingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = Path(__file__).parent / "data" / "data.csv"
TARGET = "Target"
CLASSES = ["Not Qualified", "Qualified", "High Achiever"]
RANDOM_STATE = 42
# Single process: parallel joblib workers each copy the data and push peak memory
# above the ~1 GB limit of Streamlit Community Cloud.
N_JOBS = 1

# ---------------------------------------------------------------------------
# Human-readable labels for the coded columns of the Sri Lankan G.C.E. A/L 2020
# dataset prepared by scripts/prepare_al_data.py. Codes not listed are "Code N".
# ---------------------------------------------------------------------------
_SUBJECTS_PATH = Path(__file__).parent / "data" / "subjects.json"
_SUBJECTS = ({int(k): v.title().replace("&", "and")
              for k, v in json.loads(_SUBJECTS_PATH.read_text()).items()}
             if _SUBJECTS_PATH.exists() else {})

LABELS: dict[str, dict[int, str]] = {
    "Stream": {0: "Arts", 1: "Commerce", 2: "Biological Science", 3: "Physical Science",
               4: "Engineering Technology", 5: "Biosystems Technology"},
    "Subject 1": _SUBJECTS, "Subject 2": _SUBJECTS, "Subject 3": _SUBJECTS,
    "Gender": {1: "Male", 0: "Female"},
    "Syllabus": {1: "New syllabus", 0: "Old syllabus"},
    "General English": {-1: "Absent", 0: "F (fail)", 1: "S", 2: "C", 3: "B", 4: "A"},
}

# Feature groups used to lay out the prediction form.
GROUPS: dict[str, list[str]] = {
    "📚 Subjects": ["Stream", "Subject 1", "Subject 2", "Subject 3", "Syllabus"],
    "👤 Student": ["Gender", "Age at exam"],
    "📝 General tests": ["CGT score", "General English"],
}

# Numeric (continuous) columns -> number inputs; everything else -> select boxes.
CONTINUOUS = {"Age at exam", "CGT score"}


def label_for(col: str, code) -> str:
    try:
        code = int(code)
    except (TypeError, ValueError):
        return str(code)
    return LABELS.get(col, {}).get(code, f"Code {code}")


def read_table(source) -> pd.DataFrame:
    """Read the dataset from a path or uploaded file (CSV with ; or , separator, or Excel)."""
    name = str(getattr(source, "name", source)).lower()
    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(source)
    else:
        raw = source.read() if hasattr(source, "read") else Path(source).read_bytes()
        text = raw.decode("utf-8-sig")
        sep = ";" if text.split("\n", 1)[0].count(";") > text.split("\n", 1)[0].count(",") else ","
        df = pd.read_csv(io.StringIO(text), sep=sep)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_default() -> pd.DataFrame:
    return read_table(DATA_PATH)


@dataclass
class TrainResult:
    model: object
    model_name: str
    features: list[str]
    leaderboard: pd.DataFrame
    accuracy: float
    f1_macro: float
    confusion: np.ndarray
    report: pd.DataFrame
    importance: pd.Series
    defaults: dict = field(default_factory=dict)
    options: dict = field(default_factory=dict)
    ranges: dict = field(default_factory=dict)
    models: dict = field(default_factory=dict)  # name -> per-model details
    y_test: np.ndarray | None = None  # true labels of the 20% test split

    def use(self, name: str) -> "TrainResult":
        """Return a copy where the active model is `name` (one of the trained models)."""
        m = self.models[name]
        return replace(self, model=m["model"], model_name=name, accuracy=m["accuracy"],
                       f1_macro=m["f1"], confusion=m["confusion"], report=m["report"])


class HybridRFANN(ClassifierMixin, BaseEstimator):
    """Hybrid model: Random Forest + Artificial Neural Network.

    1. A Random Forest is trained; its *out-of-fold* class probabilities on the training
       set (5-fold cross_val_predict) become 3 extra features. Out-of-fold avoids leakage:
       the ANN never sees probabilities the forest produced for rows it trained on.
    2. An MLP (ANN) is trained on the scaled original features + the RF probabilities.
    The forest contributes robust non-linear splits on the raw attributes; the ANN learns
    a smooth combination of those forest "opinions" with the original signals.
    """

    def __init__(self, n_estimators=200, hidden_layer_sizes=(32,), alpha=1.0,
                 random_state=42):
        self.n_estimators = n_estimators
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha  # strong L2: without it the ANN over-trusts the RF features
        self.random_state = random_state

    def _rf(self):
        return RandomForestClassifier(n_estimators=self.n_estimators, min_samples_leaf=20,
                                      n_jobs=N_JOBS, random_state=self.random_state)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.rf_ = self._rf()
        oof = cross_val_predict(self._rf(), X, y, cv=5, method="predict_proba")
        self.rf_.fit(X, y)
        self.classes_ = self.rf_.classes_
        self.scaler_ = StandardScaler().fit(X)
        self.ann_ = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, alpha=self.alpha,
                                  max_iter=1000, random_state=self.random_state)
        self.ann_.fit(np.hstack([self.scaler_.transform(X), oof]), y)
        return self

    def _features(self, X):
        X = np.asarray(X, dtype=float)
        return np.hstack([self.scaler_.transform(X), self.rf_.predict_proba(X)])

    def predict_proba(self, X):
        return self.ann_.predict_proba(self._features(X))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


# Ensemble family of every model, used for grouping and interpretation in the dashboard.
FAMILY = {
    "Logistic Regression": "Single model (baseline)",
    "Decision Tree": "Single model (baseline)",
    "K-Nearest Neighbors": "Single model (baseline)",
    "Support Vector Machine": "Single model (baseline)",
    "Naive Bayes": "Single model (baseline)",
    "Neural Network (MLP)": "Single model (baseline)",
    "Bagging (Decision Trees)": "Bagging",
    "Random Forest": "Bagging",
    "Extra Trees": "Bagging",
    "Gradient Boosting": "Boosting",
    "AdaBoost": "Boosting",
    "Voting (soft)": "Stacking / Voting",
    "Stacking": "Stacking / Voting",
    "Hybrid RF + ANN": "Hybrid",
}
FAMILY_ORDER = ["Single model (baseline)", "Bagging", "Boosting", "Stacking / Voting", "Hybrid"]


def make_models() -> dict:
    """All classifiers that are trained and compared (14 models in 5 families)."""
    scaled = lambda est: make_pipeline(StandardScaler(), est)  # noqa: E731
    rs = RANDOM_STATE
    return {
        # --- single models (baselines) ---
        "Logistic Regression": scaled(LogisticRegression(max_iter=2000, C=0.5)),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=10,
                                                random_state=rs),
        "K-Nearest Neighbors": scaled(KNeighborsClassifier(n_neighbors=15, weights="distance")),
        "Support Vector Machine": scaled(CalibratedClassifierCV(SVC(C=1.0), ensemble=False,
                                                                cv=3)),
        "Naive Bayes": GaussianNB(),
        "Neural Network (MLP)": scaled(MLPClassifier(
            hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=500, early_stopping=True,
            random_state=rs)),
        # --- bagging: many trees on bootstrap samples, averaged (reduces variance) ---
        # min_samples_leaf regularises the trees: the A/L features are noisy (pre-exam
        # information only) and fully grown trees memorise noise.
        "Bagging (Decision Trees)": BaggingClassifier(
            DecisionTreeClassifier(min_samples_leaf=10, random_state=rs), n_estimators=100,
            n_jobs=N_JOBS, random_state=rs),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=20, n_jobs=N_JOBS, random_state=rs),
        "Extra Trees": ExtraTreesClassifier(n_estimators=200, min_samples_leaf=10, n_jobs=N_JOBS,
                                            random_state=rs),
        # --- boosting: trees built one after another, each fixing earlier errors (reduces bias) ---
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.08, random_state=rs),
        "AdaBoost": AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=rs),
        # --- stacking / voting: combine different model types ---
        "Voting (soft)": VotingClassifier(
            estimators=[("rf", RandomForestClassifier(n_estimators=200, min_samples_leaf=20,
                                                      n_jobs=N_JOBS, random_state=rs)),
                        ("gb", HistGradientBoostingClassifier(max_iter=200,
                                                              learning_rate=0.08,
                                                              random_state=rs)),
                        ("lr", scaled(LogisticRegression(max_iter=2000, C=0.5)))],
            voting="soft"),
        "Stacking": StackingClassifier(
            estimators=[("rf", RandomForestClassifier(n_estimators=150, min_samples_leaf=20,
                                                      n_jobs=N_JOBS, random_state=rs)),
                        ("et", ExtraTreesClassifier(n_estimators=150, min_samples_leaf=10,
                                                    n_jobs=N_JOBS, random_state=rs)),
                        ("gb", HistGradientBoostingClassifier(max_iter=150,
                                                              learning_rate=0.08,
                                                              random_state=rs)),
                        ("knn", scaled(KNeighborsClassifier(n_neighbors=15)))],
            final_estimator=LogisticRegression(max_iter=2000), cv=5,
            stack_method="predict_proba"),
        # --- hybrid ---
        "Hybrid RF + ANN": HybridRFANN(random_state=rs),
    }


def train(df: pd.DataFrame) -> TrainResult:
    if TARGET not in df.columns:
        raise ValueError(f"Dataset must contain a '{TARGET}' column.")
    features = [c for c in df.columns if c != TARGET]
    X = df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = df[TARGET].astype(str)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    rows, fitted, models = [], {}, {}
    for name, est in make_models().items():
        t0 = time.perf_counter()
        est.fit(X_tr, y_tr)
        secs = time.perf_counter() - t0
        pred = est.predict(X_te)
        acc = accuracy_score(y_te, pred)
        f1 = f1_score(y_te, pred, average="macro")
        rows.append({"Model": name, "Family": FAMILY.get(name, "Other"), "Accuracy": acc, "F1 (macro)": f1,
                     "Precision (macro)": precision_score(y_te, pred, average="macro",
                                                          zero_division=0),
                     "Recall (macro)": recall_score(y_te, pred, average="macro"),
                     "Train time (s)": secs})
        fitted[name] = est
        models[name] = {
            "model": est, "accuracy": float(acc), "f1": float(f1), "pred": np.asarray(pred),
            "train_time": secs,
            "confusion": confusion_matrix(y_te, pred, labels=list(est.classes_)),
            "report": pd.DataFrame(classification_report(
                y_te, pred, output_dict=True, zero_division=0)).T}
    board = pd.DataFrame(rows).sort_values(["Accuracy", "F1 (macro)"], ascending=False)         .reset_index(drop=True)
    board.insert(0, "Rank", range(1, len(board) + 1))
    best_name = board.loc[0, "Model"]
    best = fitted[best_name]
    pred = best.predict(X_te)
    labels = list(best.classes_)

    report = pd.DataFrame(classification_report(y_te, pred, output_dict=True)).T
    importance = pd.Series(fitted["Random Forest"].feature_importances_, index=features) \
        .sort_values(ascending=False)

    defaults, options, ranges = {}, {}, {}
    for c in features:
        s = X[c]
        if c in CONTINUOUS:
            defaults[c] = float(s.median())
            ranges[c] = (float(s.min()), float(s.max()))
        else:
            vc = s.astype(int).value_counts()
            defaults[c] = int(vc.index[0])
            options[c] = sorted(vc.index.tolist(), key=lambda v: (-vc[v], v))
    combo = [c for c in ("Stream", "Subject 1", "Subject 2", "Subject 3") if c in features]
    if len(combo) > 1:  # default to the most common real subject combination
        defaults.update(dict(zip(combo, (int(v) for v in
                                         X[combo].astype(int).value_counts().index[0]))))

    return TrainResult(
        model=best, model_name=best_name, features=features, leaderboard=board,
        accuracy=float(accuracy_score(y_te, pred)),
        f1_macro=float(f1_score(y_te, pred, average="macro")),
        confusion=confusion_matrix(y_te, pred, labels=labels), report=report,
        importance=importance, defaults=defaults, options=options, ranges=ranges,
        models=models, y_test=np.asarray(y_te))
