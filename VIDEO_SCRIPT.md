# 3-minute video script (voice-over)

Total ≈ 2:50. Record your screen with the live dashboard open; read the voice-over.
Show your face only if your lecturer asks for it.

| Time | Show on screen | Voice-over |
|---|---|---|
| 0:00–0:20 | Home page | "Hi! This is my Ensemble Learning Dashboard, the Student Academic Performance Prediction System. It predicts whether a university student will drop out, stay enrolled or graduate, so that at-risk students can get help early." |
| 0:20–0:40 | Data Explorer (2nd-sem approved units box plot) | "The dataset has 4,424 students and 36 attributes: background, finances, and 1st and 2nd semester results. Semester results are clearly the strongest signal: dropouts pass far fewer units." |
| 0:40–1:10 | Model Performance (bar chart of 14 models) | "I trained 14 models in five families: single-model baselines, bagging (Random Forest, Extra Trees and a Bagging classifier), boosting (Gradient Boosting and AdaBoost), stacking and soft voting, and a hybrid Random Forest plus neural network." |
| 1:10–1:40 | Ensemble Insights, sections 1–2 | "Random Forest and Gradient Boosting have the same accuracy, but boosting finds more of the hard 'Enrolled' students. Bagging averages independent trees, which reduces variance. Boosting fixes the previous trees' errors one after another, which reduces bias and helps on borderline cases." |
| 1:40–2:05 | Ensemble Insights, section 3 (stacking and disagreement matrix) | "Stacking is the best model at 77.4% accuracy. The base models disagree on 10 to 20 percent of students, and a logistic-regression meta-learner learns which model to trust, so combining them adds accuracy." |
| 2:05–2:20 | Ensemble Insights, section 4 (hybrid) | "For hybridization I fed the Random Forest's probabilities into a neural network. It makes the network much better, but on a small tabular dataset it can't clearly beat the forest itself." |
| 2:20–2:40 | Predict Student: fill the form, press Predict; then the Train Model page | "Anyone can predict one student, upload a file for batch predictions, or upload a new dataset and retrain all 14 models live." |
| 2:40–2:55 | README on GitHub showing the AI statement | "The code and the dashboard were generated with the help of Claude, an AI tool, and I reviewed and tested everything. The notebook, code and pickle model are on GitHub. Thank you!" |

Tips: you can use OBS Studio or the Windows Snipping Tool (screen recording) or the Xbox Game Bar (Win+G). Keep the video under 3:00.
