"""Prepare the Sri Lankan G.C.E. Advanced Level 2020 results for modelling.

Input : the Kaggle file `2020_al_data_kaggle_upload_new_old_syllabi.csv`
        (337,553 candidates, 2020 A/L examination, Department of Examinations Sri Lanka).
Output: data/data.csv      – a stratified sample used by the dashboard and the notebook
        data/subjects.json – subject code -> subject name

Target (3 classes, from the 3 main-subject grades and the Z-score):
    Not Qualified  – failed (F) at least one of the 3 main subjects
    Qualified      – passed all 3 subjects (S or better) with Z-score < 1.0
    High Achiever  – passed all 3 subjects with Z-score >= 1.0
Features use only information that is *not* the subject grades themselves (no leakage):
    derived stream, the 3 subjects taken, gender, age at the exam, syllabus,
    Common General Test (CGT) score and General English grade.

Usage: python scripts/prepare_al_data.py <path-to-raw-csv> [sample_size]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
GRADES = list("ABCSF")
GE_CODE = {"Absent": -1, "F": 0, "S": 1, "C": 2, "B": 3, "A": 4}
MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August",
     "September", "October", "November", "December"], start=1)}
STREAM_CODE = {"Arts": 0, "Commerce": 1, "Biological Science": 2, "Physical Science": 3,
               "Engineering Technology": 4, "Biosystems Technology": 5}
TARGET_ORDER = ["Not Qualified", "Qualified", "High Achiever"]


def derive_stream(subs: set) -> str:
    """Stream from the subject combination (the official stream column is blank for
    candidates who failed, so it cannot be used without leaking the result)."""
    if "ENGINEERING TECHNOLOGY" in subs:
        return "Engineering Technology"
    if "BIO SYSTEMS TECHNOLOGY" in subs:
        return "Biosystems Technology"
    if "BIOLOGY" in subs and "CHEMISTRY" in subs:
        return "Biological Science"
    if "COMBINED MATHEMATICS" in subs and ("PHYSICS" in subs or "CHEMISTRY" in subs):
        return "Physical Science"
    if len(subs & {"ECONOMICS", "BUSINESS STUDIES", "ACCOUNTING"}) >= 2:
        return "Commerce"
    return "Arts"


def main(raw_path: str, sample_size: int = 12000):
    raw = pd.read_csv(raw_path, dtype=str)
    print("raw rows:", len(raw))
    g = ["sub1_r", "sub2_r", "sub3_r"]
    df = raw[raw[g].isin(GRADES).all(axis=1)].copy()          # sat all 3 subjects
    df = df[df["gender"].isin(["male", "female"])]
    df = df[df["birth_year"].str.fullmatch(r"\d{4}", na=False) & df["birth_month"].isin(MONTHS)]
    df = df[df["cgt_r"].str.fullmatch(r"\d+", na=False)]      # sat the CGT
    df = df[df["ge_r"].isin(GE_CODE)]
    print("usable rows:", len(df))

    z = pd.to_numeric(df["Zscore"], errors="coerce")
    all_pass = (df[g] != "F").all(axis=1)
    df["Target"] = np.where(~all_pass, "Not Qualified",
                            np.where(z >= 1.0, "High Achiever", "Qualified"))

    subjects = pd.concat([df["sub1"], df["sub2"], df["sub3"]]).value_counts().index.tolist()
    sub_code = {s: i for i, s in enumerate(subjects)}
    subs = df[["sub1", "sub2", "sub3"]].apply(set, axis=1)
    birth = df["birth_year"].astype(int) + (df["birth_month"].map(MONTHS) - 0.5) / 12
    out = pd.DataFrame({
        "Stream": subs.map(derive_stream).map(STREAM_CODE),
        "Subject 1": df["sub1"].map(sub_code),
        "Subject 2": df["sub2"].map(sub_code),
        "Subject 3": df["sub3"].map(sub_code),
        "Gender": (df["gender"] == "male").astype(int),
        "Age at exam": (2020 + 10 / 12 - birth).round(1),   # exam held Oct–Nov 2020
        "Syllabus": (df["syllabus"] == "new").astype(int),
        "CGT score": df["cgt_r"].astype(int),
        "General English": df["ge_r"].map(GE_CODE),
        "Target": df["Target"],
    })
    print(out["Target"].value_counts(normalize=True).round(3).to_dict())

    sample, _ = train_test_split(out, train_size=sample_size, stratify=out["Target"],
                                 random_state=42)
    sample = sample.reset_index(drop=True)
    (ROOT / "data").mkdir(exist_ok=True)
    sample.to_csv(ROOT / "data" / "data.csv", index=False)
    json.dump({str(v): k for k, v in sub_code.items()},
              open(ROOT / "data" / "subjects.json", "w"), indent=1)
    print("sample:", sample.shape, sample["Target"].value_counts().to_dict())


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 12000)
