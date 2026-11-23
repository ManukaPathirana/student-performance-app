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
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
AL_DATA_FILE = Path(__file__).parent / 'data' / 'data.csv'
OUTCOME_COL = 'Target'
RESULT_CLASSES = ['Not Qualified', 'Qualified', 'High Achiever']
SEED = 42
WORKER_COUNT = 1
_SUBJECT_FILE = Path(__file__).parent / 'data' / 'subjects.json'
_SUBJECT_NAMES = {int(key): val.title().replace('&', 'and') for key, val in json.loads(_SUBJECT_FILE.read_text()).items()} if _SUBJECT_FILE.exists() else {}
CODE_NAMES: dict[str, dict[int, str]] = {'Stream': {0: 'Arts', 1: 'Commerce', 2: 'Biological Science', 3: 'Physical Science', 4: 'Engineering Technology', 5: 'Biosystems Technology'}, 'Subject 1': _SUBJECT_NAMES, 'Subject 2': _SUBJECT_NAMES, 'Subject 3': _SUBJECT_NAMES, 'Gender': {1: 'Male', 0: 'Female'}, 'Syllabus': {1: 'New syllabus', 0: 'Old syllabus'}, 'General English': {-1: 'Absent', 0: 'F (fail)', 1: 'S', 2: 'C', 3: 'B', 4: 'A'}}
FORM_SECTIONS: dict[str, list[str]] = {'📚 Subjects': ['Stream', 'Subject 1', 'Subject 2', 'Subject 3', 'Syllabus'], '👤 Student': ['Gender', 'Age at exam'], '📝 General tests': ['CGT score', 'General English']}
NUMERIC_FIELDS = {'Age at exam', 'CGT score'}

def decode_value(field_name: str, raw_code) -> str:
    try:
        raw_code = int(raw_code)
    except (TypeError, ValueError):
        return str(raw_code)
    return CODE_NAMES.get(field_name, {}).get(raw_code, f'Code {raw_code}')

def load_table_file(src_obj) -> pd.DataFrame:
    model_label = str(getattr(src_obj, 'name', src_obj)).lower()
    if model_label.endswith(('.xlsx', '.xls')):
        frame = pd.read_excel(src_obj)
    else:
        raw_bytes = src_obj.read() if hasattr(src_obj, 'read') else Path(src_obj).read_bytes()
        decoded = raw_bytes.decode('utf-8-sig')
        delimiter = ';' if decoded.split('\n', 1)[0].count(';') > decoded.split('\n', 1)[0].count(',') else ','
        frame = pd.read_csv(io.StringIO(decoded), sep=delimiter)
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame

def load_al_sample() -> pd.DataFrame:
    return load_table_file(AL_DATA_FILE)

@dataclass
class ModelBundle:
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
    models: dict = field(default_factory=dict)
    y_test: np.ndarray | None = None

    def use(self, model_label: str) -> 'ModelBundle':
        info = self.models[model_label]
        return replace(self, model=info['model'], model_name=model_label, accuracy=info['accuracy'], f1_macro=info['f1'], confusion=info['confusion'], report=info['report'])

class ForestNeuralHybrid(ClassifierMixin, BaseEstimator):

    def __init__(self, n_estimators=200, hidden_layer_sizes=(32,), alpha=1.0, random_state=42):
        self.n_estimators = n_estimators
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.random_state = random_state

    def _rf(self):
        return RandomForestClassifier(n_estimators=self.n_estimators, min_samples_leaf=20, n_jobs=WORKER_COUNT, random_state=self.random_state)

    def fit(self, feat_matrix, outcome_vec):
        feat_matrix = np.asarray(feat_matrix, dtype=float)
        self.rf_ = self._rf()
        oof_proba = cross_val_predict(self._rf(), feat_matrix, outcome_vec, cv=5, method='predict_proba')
        self.rf_.fit(feat_matrix, outcome_vec)
        self.classes_ = self.rf_.classes_
        self.scaler_ = StandardScaler().fit(feat_matrix)
        self.ann_ = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, alpha=self.alpha, max_iter=1000, random_state=self.random_state)
        self.ann_.fit(np.hstack([self.scaler_.transform(feat_matrix), oof_proba]), outcome_vec)
        return self

    def _features(self, feat_matrix):
        feat_matrix = np.asarray(feat_matrix, dtype=float)
        return np.hstack([self.scaler_.transform(feat_matrix), self.rf_.predict_proba(feat_matrix)])

    def predict_proba(self, feat_matrix):
        return self.ann_.predict_proba(self._features(feat_matrix))

    def predict(self, feat_matrix):
        return self.classes_[self.predict_proba(feat_matrix).argmax(axis=1)]
MODEL_FAMILY = {'Logistic Regression': 'Single model (baseline)', 'Decision Tree': 'Single model (baseline)', 'K-Nearest Neighbors': 'Single model (baseline)', 'Support Vector Machine': 'Single model (baseline)', 'Naive Bayes': 'Single model (baseline)', 'Neural Network (MLP)': 'Single model (baseline)', 'Bagging (Decision Trees)': 'Bagging', 'Random Forest': 'Bagging', 'Extra Trees': 'Bagging', 'Gradient Boosting': 'Boosting', 'AdaBoost': 'Boosting', 'Voting (soft)': 'Stacking / Voting', 'Stacking': 'Stacking / Voting', 'Hybrid RF + ANN': 'Hybrid'}
FAMILY_SEQUENCE = ['Single model (baseline)', 'Bagging', 'Boosting', 'Stacking / Voting', 'Hybrid']

def build_model_zoo() -> dict:
    with_scaling = lambda estimator: make_pipeline(StandardScaler(), estimator)
    seed = SEED
    return {'Logistic Regression': with_scaling(LogisticRegression(max_iter=2000, C=0.5)), 'Decision Tree': DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, random_state=seed), 'K-Nearest Neighbors': with_scaling(KNeighborsClassifier(n_neighbors=15, weights='distance')), 'Support Vector Machine': with_scaling(CalibratedClassifierCV(SVC(C=1.0), ensemble=False, cv=3)), 'Naive Bayes': GaussianNB(), 'Neural Network (MLP)': with_scaling(MLPClassifier(hidden_layer_sizes=(64, 32), alpha=0.001, max_iter=500, early_stopping=True, random_state=seed)), 'Bagging (Decision Trees)': BaggingClassifier(DecisionTreeClassifier(min_samples_leaf=10, random_state=seed), n_estimators=100, n_jobs=WORKER_COUNT, random_state=seed), 'Random Forest': RandomForestClassifier(n_estimators=200, min_samples_leaf=20, n_jobs=WORKER_COUNT, random_state=seed), 'Extra Trees': ExtraTreesClassifier(n_estimators=200, min_samples_leaf=10, n_jobs=WORKER_COUNT, random_state=seed), 'Gradient Boosting': HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=seed), 'AdaBoost': AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=seed), 'Voting (soft)': VotingClassifier(estimators=[('rf', RandomForestClassifier(n_estimators=200, min_samples_leaf=20, n_jobs=WORKER_COUNT, random_state=seed)), ('gb', HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=seed)), ('lr', with_scaling(LogisticRegression(max_iter=2000, C=0.5)))], voting='soft'), 'Stacking': StackingClassifier(estimators=[('rf', RandomForestClassifier(n_estimators=150, min_samples_leaf=20, n_jobs=WORKER_COUNT, random_state=seed)), ('et', ExtraTreesClassifier(n_estimators=150, min_samples_leaf=10, n_jobs=WORKER_COUNT, random_state=seed)), ('gb', HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, random_state=seed)), ('knn', with_scaling(KNeighborsClassifier(n_neighbors=15)))], final_estimator=LogisticRegression(max_iter=2000), cv=5, stack_method='predict_proba'), 'Hybrid RF + ANN': ForestNeuralHybrid(random_state=seed)}

def fit_all_models(frame: pd.DataFrame) -> ModelBundle:
    if OUTCOME_COL not in frame.columns:
        raise ValueError(f"Dataset must contain a '{OUTCOME_COL}' column.")
    feature_cols = [column for column in frame.columns if column != OUTCOME_COL]
    feat_matrix = frame[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    outcome_vec = frame[OUTCOME_COL].astype(str)
    train_x, test_x, train_y, test_y = train_test_split(feat_matrix, outcome_vec, test_size=0.2, stratify=outcome_vec, random_state=SEED)
    score_rows, trained, model_store = ([], {}, {})
    for model_label, estimator in build_model_zoo().items():
        start_time = time.perf_counter()
        estimator.fit(train_x, train_y)
        fit_seconds = time.perf_counter() - start_time
        preds = estimator.predict(test_x)
        acc_score = accuracy_score(test_y, preds)
        f1_val = f1_score(test_y, preds, average='macro')
        score_rows.append({'Model': model_label, 'Family': MODEL_FAMILY.get(model_label, 'Other'), 'Accuracy': acc_score, 'F1 (macro)': f1_val, 'Precision (macro)': precision_score(test_y, preds, average='macro', zero_division=0), 'Recall (macro)': recall_score(test_y, preds, average='macro'), 'Train time (s)': fit_seconds})
        trained[model_label] = estimator
        model_store[model_label] = {'model': estimator, 'accuracy': float(acc_score), 'f1': float(f1_val), 'pred': np.asarray(preds), 'train_time': fit_seconds, 'confusion': confusion_matrix(test_y, preds, labels=list(estimator.classes_)), 'report': pd.DataFrame(classification_report(test_y, preds, output_dict=True, zero_division=0)).T}
    ranking = pd.DataFrame(score_rows).sort_values(['Accuracy', 'F1 (macro)'], ascending=False).reset_index(drop=True)
    ranking.insert(0, 'Rank', range(1, len(ranking) + 1))
    top_name = ranking.loc[0, 'Model']
    top_model = trained[top_name]
    preds = top_model.predict(test_x)
    class_order = list(top_model.classes_)
    class_report = pd.DataFrame(classification_report(test_y, preds, output_dict=True)).T
    imp_series = pd.Series(trained['Random Forest'].feature_importances_, index=feature_cols).sort_values(ascending=False)
    default_vals, option_lists, value_ranges = ({}, {}, {})
    for column in feature_cols:
        column_vals = feat_matrix[column]
        if column in NUMERIC_FIELDS:
            default_vals[column] = float(column_vals.median())
            value_ranges[column] = (float(column_vals.min()), float(column_vals.max()))
        else:
            freq = column_vals.astype(int).value_counts()
            default_vals[column] = int(freq.index[0])
            option_lists[column] = sorted(freq.index.tolist(), key=lambda val: (-freq[val], val))
    combo_cols = [column for column in ('Stream', 'Subject 1', 'Subject 2', 'Subject 3') if column in feature_cols]
    if len(combo_cols) > 1:
        default_vals.update(dict(zip(combo_cols, (int(val) for val in feat_matrix[combo_cols].astype(int).value_counts().index[0]))))
    return ModelBundle(model=top_model, model_name=top_name, features=feature_cols, leaderboard=ranking, accuracy=float(accuracy_score(test_y, preds)), f1_macro=float(f1_score(test_y, preds, average='macro')), confusion=confusion_matrix(test_y, preds, labels=class_order), report=class_report, importance=imp_series, defaults=default_vals, options=option_lists, ranges=value_ranges, models=model_store, y_test=np.asarray(test_y))
