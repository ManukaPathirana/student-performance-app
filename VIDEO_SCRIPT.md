# 3-minute video script (voice-over)

Total ≈ 2:50. Record your screen with the live dashboard open and read the voice-over.
Show your face only if your lecturer asks for it.

| Time | Show on screen | Voice-over |
|---|---|---|
| 0:00–0:20 | Home page | "Hi! This is my Ensemble Learning Dashboard, the Student Academic Performance Prediction System. It uses real Sri Lankan G.C.E. A/L 2020 results to predict whether a student will be Not Qualified, Qualified, or a High Achiever with a Z-score of 1 or more." |
| 0:20–0:45 | Data Explorer (CGT score box plot, then result by stream) | "The data covers over 250,000 A/L candidates; I use a stratified sample of 12,000. To avoid leaking the answer, I only use the stream, the three subjects, gender, age, syllabus, CGT score and General English. Higher CGT scores clearly go with better results, and pass rates differ a lot between streams." |
| 0:45–1:10 | Model Performance (bar chart of 14 models) | "I trained 14 models in five families: single-model baselines, bagging (Random Forest, Extra Trees and a Bagging classifier), boosting (Gradient Boosting and AdaBoost), stacking and soft voting, and a hybrid Random Forest plus neural network." |
| 1:10–1:40 | Ensemble Insights, sections 1–2 | "This is a noisy problem, so bias and variance decide the ranking. Bagging averages many trees, which reduces variance. Once I limited the leaf size, Random Forest became the most accurate model. Boosting fixes the previous trees' errors, which reduces bias, so it finds more of the rare High Achievers." |
| 1:40–2:05 | Ensemble Insights, section 3 (stacking and disagreement matrix) | "The models disagree on 17 to 33 percent of students. A stacking meta-learner learns which model to trust, which gives one of the best and most balanced results." |
| 2:05–2:20 | Ensemble Insights, sections 4–5 | "The hybrid feeds Random Forest probabilities into a neural network, which beats the network alone. The CGT score, gender and General English drive the predictions most." |
| 2:20–2:40 | Predict Student: choose subjects and a CGT score, press Predict; then the Train Model page | "Anyone can predict one student, upload a file for batch predictions, or upload a new dataset and retrain all 14 models live." |
| 2:40–2:55 | README on GitHub showing the AI statement | "The code and the dashboard were generated with the help of Claude, an AI tool, and I reviewed and tested everything. The notebook, code and pickle model are on GitHub. Thank you!" |

Tips: record with the Windows Snipping Tool (screen recording), the Xbox Game Bar (Win+G) or OBS Studio. Keep the video under 3:00.
