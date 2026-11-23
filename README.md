# 🎓 Student Academic Performance Prediction System

A Streamlit web app that predicts whether a student will **Drop out**, stay **Enrolled**, or **Graduate**,
using the UCI *Predict Students' Dropout and Academic Success* dataset (4,424 students, 36 attributes).

## Features
- **Home** – dataset summary, outcome distribution, top factors
- **Data Explorer** – outcome breakdown by any attribute, correlations, raw data
- **Predict Student** – form with all attributes → predicted outcome, probabilities, recommendation
- **Batch Prediction** – upload CSV/Excel of many students, download predictions
- **Model Performance** – compares 10 models (Logistic Regression, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, AdaBoost, KNN, SVM, Naive Bayes, Neural Network), confusion matrix, feature importance; pick any model for predictions from the sidebar
- **Train Model** – upload your own dataset (same columns + `Target`) and retrain the models; sample files are in `sample_data/`

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this folder to a public GitHub repository.
2. Go to https://share.streamlit.io → **Create app** → pick the repo, branch `main`, main file `app.py`.
3. Choose your app URL (e.g. `student-performance-prediction.streamlit.app`) and click **Deploy**.

Dataset: Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). UCI Machine Learning Repository. CC BY 4.0.
