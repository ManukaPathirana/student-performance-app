# 🎓 Student Academic Performance Prediction System — Ensemble Learning Dashboard

ML & Pattern Recognition — *Ensemble Learning Dashboard* assignment.

Predicts whether a student will **Drop out**, stay **Enrolled**, or **Graduate**, and compares
**bagging, boosting, stacking/voting and a hybrid Random Forest + ANN** against single-model baselines.

- 🌐 **Live dashboard:** https://student-performance-app-tvqzyrlzyfe9ahvma6mtrj.streamlit.app
- 📓 **Notebook:** [`Ensemble_Learning_Student_Performance.ipynb`](Ensemble_Learning_Student_Performance.ipynb)
- 💾 **Saved model (pickle):** `artifacts/best_model.pkl` (best model from the notebook)

> ## 🤖 AI-use statement
> The code of this project (notebook, `model.py`, `app.py`) and the dashboard front end were
> **generated with the help of Claude (Anthropic), a generative-AI tool**. The student directed the
> work, reviewed the code, ran and tested it, and checked the results and interpretation.

## Models (14, in 5 families)

| Family | Models |
|---|---|
| Single-model baselines | Logistic Regression, Decision Tree, KNN, SVM, Naive Bayes, Neural Network (MLP) |
| **Bagging** | BaggingClassifier (decision trees), Random Forest, Extra Trees |
| **Boosting** | Gradient Boosting (histogram GBM), AdaBoost |
| **Stacking / Voting** | Soft Voting (RF + GBM + LR), StackingClassifier (RF, ET, GBM, KNN → Logistic Regression meta-learner, 5-fold), GridSearchCV on the meta-learner (notebook) |
| **Hybrid** | Random Forest + ANN (RF out-of-fold probabilities fed into an MLP) |

## Key results (20% test set)

| Model | Accuracy | Macro-F1 |
|---|---|---|
| ⭐ Stacking | 77.4% | 0.713 |
| Voting (soft) | 76.9% | 0.694 |
| Gradient Boosting | 76.7% | 0.698 |
| Random Forest | 76.7% | 0.680 |
| Hybrid RF + ANN | 75.8% | 0.689 |
| Neural Network (MLP) | 74.0% | 0.643 |
| Naive Bayes | 66.0% | 0.558 |

Interpretation (full version in the notebook and on the **Ensemble Insights** page):
- **Boosting vs bagging:** same accuracy, but boosting has a better macro-F1 and finds more of the hard
  *Enrolled* students. Boosting reduces bias by fitting the errors one tree after another; bagging
  reduces variance by averaging independent trees.
- **Stacking** wins because its base models make different mistakes and the meta-learner learns which model to trust.
- **Hybrid RF + ANN** makes the ANN much better and has a slightly higher macro-F1 than the forest, but
  a slightly lower accuracy. With weak regularisation it overfits (68.7%).
- **Semester results** (approved units, grades) are the strongest predictors.

## Dashboard pages
Home · Train Model (upload your own CSV/Excel and retrain all 14 models) · Data Explorer ·
Predict Student · Batch Prediction · Model Performance · Ensemble Insights.
Sample files for upload are in `sample_data/`.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Data
*Predict Students' Dropout and Academic Success* — Realinho, V., Vieira Martins, M., Machado, J.,
& Baptista, L. (2021). UCI Machine Learning Repository. CC BY 4.0. (Higher-education students from
Portugal.)
