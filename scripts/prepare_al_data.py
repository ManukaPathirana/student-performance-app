import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
PROJECT_DIR = Path(__file__).resolve().parent.parent
VALID_GRADES = list('ABCSF')
ENGLISH_GRADE_CODE = {'Absent': -1, 'F': 0, 'S': 1, 'C': 2, 'B': 3, 'A': 4}
MONTH_NUM = {m: i for i, m in enumerate(['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'], start=1)}
STREAM_IDS = {'Arts': 0, 'Commerce': 1, 'Biological Science': 2, 'Physical Science': 3, 'Engineering Technology': 4, 'Biosystems Technology': 5}
OUTCOME_ORDER = ['Not Qualified', 'Qualified', 'High Achiever']

def stream_from_subjects(subject_sets: set) -> str:
    if 'ENGINEERING TECHNOLOGY' in subject_sets:
        return 'Engineering Technology'
    if 'BIO SYSTEMS TECHNOLOGY' in subject_sets:
        return 'Biosystems Technology'
    if 'BIOLOGY' in subject_sets and 'CHEMISTRY' in subject_sets:
        return 'Biological Science'
    if 'COMBINED MATHEMATICS' in subject_sets and ('PHYSICS' in subject_sets or 'CHEMISTRY' in subject_sets):
        return 'Physical Science'
    if len(subject_sets & {'ECONOMICS', 'BUSINESS STUDIES', 'ACCOUNTING'}) >= 2:
        return 'Commerce'
    return 'Arts'

def build_sample(csv_path: str, n_rows: int=12000):
    raw_df = pd.read_csv(csv_path, dtype=str)
    print('raw rows:', len(raw_df))
    grade_cols = ['sub1_r', 'sub2_r', 'sub3_r']
    cand = raw_df[raw_df[grade_cols].isin(VALID_GRADES).all(axis=1)].copy()
    cand = cand[cand['gender'].isin(['male', 'female'])]
    cand = cand[cand['birth_year'].str.fullmatch('\\d{4}', na=False) & cand['birth_month'].isin(MONTH_NUM)]
    cand = cand[cand['cgt_r'].str.fullmatch('\\d+', na=False)]
    cand = cand[cand['ge_r'].isin(ENGLISH_GRADE_CODE)]
    print('usable rows:', len(cand))
    zscore = pd.to_numeric(cand['Zscore'], errors='coerce')
    passed_all = (cand[grade_cols] != 'F').all(axis=1)
    cand['Target'] = np.where(~passed_all, 'Not Qualified', np.where(zscore >= 1.0, 'High Achiever', 'Qualified'))
    subject_list = pd.concat([cand['sub1'], cand['sub2'], cand['sub3']]).value_counts().index.tolist()
    subject_ids = {s: i for i, s in enumerate(subject_list)}
    subject_sets = cand[['sub1', 'sub2', 'sub3']].apply(set, axis=1)
    birth_decimal = cand['birth_year'].astype(int) + (cand['birth_month'].map(MONTH_NUM) - 0.5) / 12
    prepared = pd.DataFrame({'Stream': subject_sets.map(stream_from_subjects).map(STREAM_IDS), 'Subject 1': cand['sub1'].map(subject_ids), 'Subject 2': cand['sub2'].map(subject_ids), 'Subject 3': cand['sub3'].map(subject_ids), 'Gender': (cand['gender'] == 'male').astype(int), 'Age at exam': (2020 + 10 / 12 - birth_decimal).round(1), 'Syllabus': (cand['syllabus'] == 'new').astype(int), 'CGT score': cand['cgt_r'].astype(int), 'General English': cand['ge_r'].map(ENGLISH_GRADE_CODE), 'Target': cand['Target']})
    print(prepared['Target'].value_counts(normalize=True).round(3).to_dict())
    sampled, _ = train_test_split(prepared, train_size=n_rows, stratify=prepared['Target'], random_state=42)
    sampled = sampled.reset_index(drop=True)
    (PROJECT_DIR / 'data').mkdir(exist_ok=True)
    sampled.to_csv(PROJECT_DIR / 'data' / 'data.csv', index=False)
    json.dump({str(v): k for k, v in subject_ids.items()}, open(PROJECT_DIR / 'data' / 'subjects.json', 'w'), indent=1)
    print('sample:', sampled.shape, sampled['Target'].value_counts().to_dict())
if __name__ == '__main__':
    build_sample(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 12000)
