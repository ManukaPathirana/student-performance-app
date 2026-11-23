"""Data loading, feature metadata and model training for the
Student Academic Performance Prediction System."""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).parent / "data" / "data.csv"
TARGET = "Target"
CLASSES = ["Dropout", "Enrolled", "Graduate"]
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Human-readable labels for the coded columns (UCI "Predict Students' Dropout
# and Academic Success" dataset). Codes not listed are shown as "Code N".
# ---------------------------------------------------------------------------
LABELS: dict[str, dict[int, str]] = {
    "Marital status": {1: "Single", 2: "Married", 3: "Widower", 4: "Divorced",
                       5: "Facto union", 6: "Legally separated"},
    "Application mode": {
        1: "1st phase - general contingent", 2: "Ordinance No. 612/93",
        5: "1st phase - special contingent (Azores)", 7: "Holders of other higher courses",
        10: "Ordinance No. 854-B/99", 15: "International student (bachelor)",
        16: "1st phase - special contingent (Madeira)", 17: "2nd phase - general contingent",
        18: "3rd phase - general contingent", 26: "Ordinance No. 533-A/99, item b2",
        27: "Ordinance No. 533-A/99, item b3", 39: "Over 23 years old", 42: "Transfer",
        43: "Change of course", 44: "Technological specialization diploma holders",
        51: "Change of institution/course", 53: "Short cycle diploma holders",
        57: "Change of institution/course (International)"},
    "Course": {
        33: "Biofuel Production Technologies", 171: "Animation and Multimedia Design",
        8014: "Social Service (evening)", 9003: "Agronomy", 9070: "Communication Design",
        9085: "Veterinary Nursing", 9119: "Informatics Engineering", 9130: "Equinculture",
        9147: "Management", 9238: "Social Service", 9254: "Tourism", 9500: "Nursing",
        9556: "Oral Hygiene", 9670: "Advertising and Marketing Management",
        9773: "Journalism and Communication", 9853: "Basic Education",
        9991: "Management (evening)"},
    "Daytime/evening attendance": {1: "Daytime", 0: "Evening"},
    "Previous qualification": {
        1: "Secondary education", 2: "Bachelor's degree", 3: "Degree", 4: "Master's",
        5: "Doctorate", 6: "Frequency of higher education", 9: "12th year - not completed",
        10: "11th year - not completed", 12: "Other - 11th year", 14: "10th year",
        15: "10th year - not completed", 19: "Basic education 3rd cycle",
        38: "Basic education 2nd cycle", 39: "Technological specialization course",
        40: "Higher education - degree (1st cycle)", 42: "Professional higher technical course",
        43: "Higher education - master (2nd cycle)"},
    "Nacionality": {
        1: "Portuguese", 2: "German", 6: "Spanish", 11: "Italian", 13: "Dutch", 14: "English",
        17: "Lithuanian", 21: "Angolan", 22: "Cape Verdean", 24: "Guinean",
        25: "Mozambican", 26: "Santomean", 32: "Turkish", 41: "Brazilian", 62: "Romanian",
        100: "Moldova", 101: "Mexican", 103: "Ukrainian", 105: "Russian", 108: "Cuban",
        109: "Colombian"},
    "Gender": {1: "Male", 0: "Female"},
}
for _yn in ["Displaced", "Educational special needs", "Debtor", "Tuition fees up to date",
            "Scholarship holder", "International"]:
    LABELS[_yn] = {1: "Yes", 0: "No"}

# Feature groups used to lay out the prediction form.
GROUPS: dict[str, list[str]] = {
    "👤 Personal": ["Gender", "Age at enrollment", "Marital status", "Nacionality",
                   "International", "Displaced", "Educational special needs"],
    "🎓 Application & Background": ["Course", "Daytime/evening attendance", "Application mode",
                                   "Application order", "Previous qualification",
                                   "Previous qualification (grade)", "Admission grade"],
    "👪 Family": ["Mother's qualification", "Father's qualification",
                 "Mother's occupation", "Father's occupation"],
    "💰 Financial": ["Scholarship holder", "Tuition fees up to date", "Debtor"],
    "📘 1st Semester": [c for c in [
        "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)",
        "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (grade)", "Curricular units 1st sem (without evaluations)"]],
    "📗 2nd Semester": [
        "Curricular units 2nd sem (credited)", "Curricular units 2nd sem (enrolled)",
        "Curricular units 2nd sem (evaluations)", "Curricular units 2nd sem (approved)",
        "Curricular units 2nd sem (grade)", "Curricular units 2nd sem (without evaluations)"],
    "📉 Economy (at enrollment)": ["Unemployment rate", "Inflation rate", "GDP"],
}

# Numeric (continuous) columns -> number inputs; everything else -> select boxes.
CONTINUOUS = {"Previous qualification (grade)", "Admission grade", "Age at enrollment",
              "Application order", "Unemployment rate", "Inflation rate", "GDP"} | {
    c for g in ("📘 1st Semester", "📗 2nd Semester") for c in GROUPS[g]}


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


def train(df: pd.DataFrame) -> TrainResult:
    if TARGET not in df.columns:
        raise ValueError(f"Dataset must contain a '{TARGET}' column.")
    features = [c for c in df.columns if c != TARGET]
    X = df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = df[TARGET].astype(str)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    candidates = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced_subsample", n_jobs=-1,
            random_state=RANDOM_STATE),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.08, random_state=RANDOM_STATE),
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)),
    }
    rows, fitted = [], {}
    for name, est in candidates.items():
        est.fit(X_tr, y_tr)
        pred = est.predict(X_te)
        rows.append({"Model": name, "Accuracy": accuracy_score(y_te, pred),
                     "F1 (macro)": f1_score(y_te, pred, average="macro")})
        fitted[name] = est
    board = pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)
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

    return TrainResult(
        model=best, model_name=best_name, features=features, leaderboard=board,
        accuracy=float(accuracy_score(y_te, pred)),
        f1_macro=float(f1_score(y_te, pred, average="macro")),
        confusion=confusion_matrix(y_te, pred, labels=labels), report=report,
        importance=importance, defaults=defaults, options=options, ranges=ranges)
