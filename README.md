# 🎓 Student Academic Performance Prediction System — Sri Lankan A/L Ensemble Learning Dashboard

ML & Pattern Recognition — *Ensemble Learning Dashboard* assignment.

Predicts a Sri Lankan student's **G.C.E. Advanced Level (2020)** result class (**Not Qualified**,
**Qualified** or **High Achiever**) and compares **bagging, boosting, stacking/voting and a hybrid
Random Forest + ANN** against single-model baselines.

- 🌐 **Live dashboard:** https://student-performance-app-tvqzyrlzyfe9ahvma6mtrj.streamlit.app
- 📓 **Notebook:** [`Ensemble_Learning_Student_Performance.ipynb`](Ensemble_Learning_Student_Performance.ipynb)
- 🧹 **Data preparation:** [`scripts/prepare_al_data.py`](scripts/prepare_al_data.py)
- 💾 **Saved model (pickle):** `artifacts/best_model.pkl` (best model from the notebook)

> ## 🤖 AI-use statement
> The code of this project (notebook, data-preparation script, `model.py`, `app.py`) and the dashboard
> front end were **generated with the help of Claude (Anthropic), a generative-AI tool**. The student
> directed the work, reviewed the code, ran and tested it, and checked the results and interpretation.

## Data — Sri Lankan G.C.E. A/L 2020
Public Kaggle upload of the 2020 A/L results (Department of Examinations, Sri Lanka), 337,553 candidates.
After cleaning there are 256,606 students who sat all 3 subjects and the CGT. `data/data.csv` is a
**stratified sample of 12,000** of them, so all 14 models can be trained live in the dashboard.

| Target class | Definition | Share |
|---|---|---|
| Not Qualified | failed (F) at least one of the 3 subjects | 36.4% |
| Qualified | passed all 3 subjects, Z-score < 1.0 | 47.9% |
| High Achiever | passed all 3 subjects, Z-score ≥ 1.0 | 15.7% |

**Features (9):** stream (derived from the subject combination), subject 1–3, gender, age at the
exam, syllabus (new/old), CGT score and General English grade. The subject grades, Z-score, ranks and
the official stream column (blank for failed candidates) are **not** used as features, because they
would leak the answer.

To rebuild the sample from the raw Kaggle file:
```
python scripts/prepare_al_data.py path/to/2020_al_data_kaggle_upload_new_old_syllabi.csv 12000
```

## Models (14, in 5 families)

| Family | Models |
|---|---|
| Single-model baselines | Logistic Regression, Decision Tree, KNN, SVM, Naive Bayes, Neural Network (MLP) |
| **Bagging** | BaggingClassifier (decision trees), Random Forest, Extra Trees (all regularised with a minimum leaf size) |
| **Boosting** | Gradient Boosting (histogram GBM), AdaBoost |
| **Stacking / Voting** | Soft Voting (RF + GBM + LR), StackingClassifier (RF, ET, GBM, KNN → Logistic Regression meta-learner, 5-fold), GridSearchCV on the meta-learner (notebook) |
| **Hybrid** | Random Forest + ANN (RF out-of-fold probabilities fed into an MLP) |

## Key findings
See the notebook and the dashboard's **Ensemble Insights** page for the full numbers and interpretation.
- This is a **hard, noisy problem**: only pre-result information is used, so models reach about
  55–60% accuracy, compared with about 48% for always guessing the largest class.
- **Bagging:** fully grown trees overfit (about 52%). A regularised Random Forest (min. 20 students
  per leaf) reduces variance and gives the best accuracy.
- **Boosting** reduces bias and finds noticeably more of the rare *High Achiever* students.
- **Stacking** and the **hybrid RF + ANN** combine models with different errors (they disagree on
  17–33% of students) for balanced, top-ranked results.
- **What drives the result:** CGT score, gender, General English, age
  and the subject combination. Age has a non-linear effect: second-attempt candidates fail less often but rarely reach Z ≥ 1.0.

## Dashboard pages
Home · Train Model (upload your own CSV/Excel and retrain all 14 models) · Data Explorer ·
Predict Student · Batch Prediction · Model Performance · Ensemble Insights.
Sample files for upload are in `sample_data/`.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```
