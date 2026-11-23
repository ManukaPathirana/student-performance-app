from __future__ import annotations
import hashlib
import importlib
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import model as _model_module
MODEL_SOURCE_HASH = hashlib.md5(Path(_model_module.__file__).read_bytes()).hexdigest()
if getattr(_model_module, 'CODE_VERSION', None) != MODEL_SOURCE_HASH:
    _model_module = importlib.reload(_model_module)
    _model_module.CODE_VERSION = MODEL_SOURCE_HASH
from model import RESULT_CLASSES, NUMERIC_FIELDS, FAMILY_SEQUENCE, FORM_SECTIONS, OUTCOME_COL, decode_value, load_al_sample, load_table_file, fit_all_models, AL_DATA_FILE as SAMPLE_CSV
SAMPLES_FOLDER = SAMPLE_CSV.parent.parent / 'sample_data'
st.set_page_config(page_title='Student Academic Performance Prediction System', page_icon='🎓', layout='wide')
OUTCOME_COLOURS = {'Not Qualified': '#E4572E', 'Qualified': '#F3A712', 'High Achiever': '#29A36A'}
OUTCOME_GUIDANCE = {'Not Qualified': ('At risk of failing at least one A/L subject', 'Give extra support in the weakest subject (past papers, revision classes), check attendance, and plan for a repeat attempt if needed.'), 'Qualified': ('Likely to pass all three subjects (university-eligible range)', 'Push for higher grades to improve the Z-score — selection to popular university courses depends on it.'), 'High Achiever': ('Likely to pass all three subjects with a high Z-score (≥ 1.0)', 'Strong candidate for competitive university courses. Keep up the momentum and consider mentoring peers.')}
st.markdown('\n<style>\n.block-container {padding-top: 2rem;}\n.hero {padding: 1.4rem 1.6rem; border-radius: 14px; color: white;\n       background: linear-gradient(120deg, #1F3B73 0%, #3A6FD8 100%); margin-bottom: 1.2rem;}\n.hero h1 {color: white; margin: 0; font-size: 2rem;}\n.hero p {margin: .3rem 0 0; opacity: .9;}\n.result {padding: 1.2rem 1.4rem; border-radius: 14px; color: white; margin-bottom: .8rem;}\n.result h2 {color: white; margin: 0;}\n</style>', unsafe_allow_html=True)

@st.cache_data(show_spinner=False, max_entries=3)
def cached_dataset(payload: bytes | None, payload_name: str | None) -> pd.DataFrame:
    if payload is None:
        return load_al_sample()
    import io
    byte_stream = io.BytesIO(payload)
    byte_stream.name = payload_name
    return load_table_file(byte_stream)

@st.cache_resource(max_entries=2, show_spinner='Training 14 models (bagging, boosting, stacking, hybrid)… about 1–2 minutes, first run only')
def cached_models(dataset_key: str, _frame: pd.DataFrame, code_hash: str=''):
    return fit_all_models(_frame)
with st.sidebar:
    st.markdown('## 🎓 SAPPS')
    st.caption('Student Academic Performance Prediction System')
    nav_choice = st.radio('Navigate', ['🏠 Home', '🧠 Train Model', '📊 Data Explorer', '🔮 Predict Student', '📁 Batch Prediction', '📈 Model Performance', '🧩 Ensemble Insights'], label_visibility='collapsed')
    st.divider()
upload_src = st.session_state.get('train_src')
if upload_src:
    al_df = cached_dataset(*upload_src)
    bundle = cached_models(hashlib.md5(upload_src[0]).hexdigest(), al_df, MODEL_SOURCE_HASH)
    source_label = upload_src[1]
else:
    al_df = cached_dataset(None, None)
    bundle = cached_models('default', al_df, MODEL_SOURCE_HASH)
    source_label = 'Built-in dataset'
if not hasattr(bundle, 'use'):
    cached_models.clear()
    st.rerun()
top_model_name = bundle.model_name
ranked_names = bundle.leaderboard['Model'].tolist()
with st.sidebar:
    chosen_model = st.selectbox('Model used for predictions', ranked_names, index=0, key='chosen_model', format_func=lambda mname: f"{mname} — {bundle.models[mname]['accuracy']:.1%}" + (' ⭐ best' if mname == top_model_name else ''))
    bundle = bundle.use(chosen_model)
    st.success(f'**Model:** {bundle.model_name}\n\n**Test accuracy:** {bundle.accuracy:.1%}\n\n**Trained on:** {source_label} ({len(al_df):,} rows, {len(ranked_names)} models compared)')
    st.caption('🤖 **AI disclosure:** the code and the dashboard UI of this project were generated with the help of Claude (Anthropic), a generative-AI tool, and then reviewed, run and tested by the student.')

def render_leaderboard(ranking: pd.DataFrame):
    formats = {'Accuracy': '{:.2%}', 'F1 (macro)': '{:.3f}', 'Precision (macro)': '{:.3f}', 'Recall (macro)': '{:.3f}', 'Train time (s)': '{:.2f}'}
    styled = ranking.style.format(formats).apply(lambda right_col: ['background-color: #cde8d6' if right_col['Model'] == bundle.model_name else '' for _ in right_col], axis=1)
    st.dataframe(styled, width='stretch', hide_index=True)

def banner(heading: str, subtitle: str):
    st.markdown(f'<div class="hero"><h1>{heading}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)

def readable_values(attr: str, ser: pd.Series) -> pd.Series:
    return ser if attr in NUMERIC_FIELDS else ser.map(lambda v: decode_value(attr, v))
if nav_choice == '🏠 Home':
    banner('🎓 Student Academic Performance Prediction System', "Predict a Sri Lankan student's G.C.E. Advanced Level result — Not Qualified, Qualified or High Achiever — from subjects, CGT, General English and profile.")
    class_counts = al_df[OUTCOME_COL].value_counts()
    kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)
    kpi_1.metric('Total students', f'{len(al_df):,}')
    for attr, item_name in zip((kpi_2, kpi_3, kpi_4), ('High Achiever', 'Qualified', 'Not Qualified')):
        count_n = class_counts.get(item_name, 0)
        attr.metric(f'{item_name} ({count_n / len(al_df):.0%})', f'{count_n:,}')
    left_pane, right_pane = st.columns([1, 1])
    with left_pane:
        chart = px.pie(names=class_counts.index, values=class_counts.values, hole=0.55, color=class_counts.index, color_discrete_map=OUTCOME_COLOURS, title='A/L result distribution')
        st.plotly_chart(chart, width='stretch')
    with right_pane:
        top_feats = bundle.importance.head(10)[::-1]
        chart = px.bar(x=top_feats.values, y=top_feats.index, orientation='h', title='Factors influencing the A/L result', labels={'x': 'Importance', 'y': ''})
        chart.update_traces(marker_color='#3A6FD8')
        st.plotly_chart(chart, width='stretch')
    st.subheader('How it works')
    first, second, fld = st.columns(3)
    first.info(f'**1. Data** — Sri Lankan G.C.E. A/L 2020 results (Department of Examinations): {len(al_df):,} students sampled from 256,606 candidates. 9 attributes per student.')
    second.info(f"**2. Model** — {len(ranked_names)} models (bagging, boosting, stacking, voting, hybrid RF+ANN and single-model baselines) are compared; **{top_model_name}** performed best ({bundle.models[top_model_name]['accuracy']:.1%} accuracy).")
    fld.info("**3. Predict** — retrain on your own data in *Train Model*, then enter a student's details on the *Predict Student* page, or upload a file for batch prediction.")
    st.caption('Dataset: G.C.E. Advanced Level Examination 2020 results, Sri Lanka (public Kaggle upload of Department of Examinations data). Not Qualified = failed ≥1 subject · Qualified = passed all 3, Z < 1.0 · High Achiever = passed all 3, Z ≥ 1.0.')
elif nav_choice == '🧠 Train Model':
    banner('🧠 Train the Model', 'Upload a student dataset (CSV or Excel) and train the prediction models on it.')
    st.markdown('#### 1. Get a dataset')
    st.write('The file needs the 9 student attribute columns plus a **Target** column (Not Qualified / Qualified / High Achiever). Download the sample dataset to see the format.')
    dl_c, dl_d = st.columns(2)
    dl_c.download_button('⬇️ Sample dataset (Excel)', SAMPLES_FOLDER.joinpath('student_training_dataset.xlsx').read_bytes(), 'student_training_dataset.xlsx', width='stretch')
    dl_d.download_button('⬇️ Sample dataset (CSV)', SAMPLE_CSV.read_bytes(), 'student_training_dataset.csv', 'text/csv', width='stretch')
    st.markdown('#### 2. Upload and train')
    uploaded = st.file_uploader('Upload training dataset', type=['csv', 'xlsx', 'xls'], key='train_up')
    if uploaded is not None:
        try:
            preview_df = cached_dataset(uploaded.getvalue(), uploaded.name)
            missing_cols = [fld for fld in bundle.features if fld not in preview_df.columns]
            if OUTCOME_COL not in preview_df.columns:
                st.error(f"The file has no '{OUTCOME_COL}' column — the model needs it to learn.")
            elif missing_cols:
                st.error(f"Missing columns: {', '.join(missing_cols)}")
            else:
                class_freq = preview_df[OUTCOME_COL].value_counts()
                st.success(f'✅ {uploaded.name}: {len(preview_df):,} students, {preview_df.shape[1]} columns — ' + ', '.join((f'{k}: {v:,}' for k, v in class_freq.items())))
                st.dataframe(preview_df.head(10), width='stretch')
                if class_freq.min() < 5:
                    st.error('Each Target class needs at least 5 students to train.')
                elif st.button('🚀 Train models on this dataset', type='primary', width='stretch'):
                    with st.spinner('Training 14 models (bagging, boosting, stacking, voting, hybrid RF+ANN, baselines). This takes 1–2 minutes.'):
                        cached_models(hashlib.md5(uploaded.getvalue()).hexdigest(), preview_df, MODEL_SOURCE_HASH)
                    st.session_state['train_src'] = (uploaded.getvalue(), uploaded.name)
                    st.session_state['just_trained'] = True
                    st.rerun()
        except Exception as exc:
            st.error(f'Could not read file: {exc}')
    if upload_src:
        if st.button('↩️ Reset to built-in dataset'):
            st.session_state.pop('train_src', None)
            st.rerun()
    st.markdown('#### 3. Current model')
    if st.session_state.pop('just_trained', False):
        st.balloons()
        st.success(f'Training finished on **{source_label}**. All pages now use the new model.')
    kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)
    kpi_1.metric('Training rows', f'{len(al_df):,}')
    kpi_2.metric('Models trained', len(ranked_names))
    kpi_3.metric('Best accuracy', f"{bundle.models[top_model_name]['accuracy']:.1%}")
    kpi_4.metric('Best F1 (macro)', f"{bundle.models[top_model_name]['f1']:.3f}")
    render_leaderboard(bundle.leaderboard)
    st.caption(f'⭐ Best model: **{top_model_name}**. 80% of rows are used for training and 20% held out for testing. Pick a different model in the sidebar to use it for predictions. See *Model Performance* for details.')
elif nav_choice == '📊 Data Explorer':
    banner('📊 Data Explorer', 'Explore how student attributes relate to academic outcomes.')
    attribute_list = [fld for fld in al_df.columns if fld != OUTCOME_COL]
    attr = st.selectbox('Choose an attribute', attribute_list, index=attribute_list.index('CGT score') if 'CGT score' in attribute_list else 0)
    subset = al_df[[attr, OUTCOME_COL]].copy()
    if attr in NUMERIC_FIELDS:
        chart = px.box(subset, x=OUTCOME_COL, y=attr, color=OUTCOME_COL, color_discrete_map=OUTCOME_COLOURS, category_orders={OUTCOME_COL: RESULT_CLASSES}, points=False, title=f'{attr} by outcome')
        st.plotly_chart(chart, width='stretch')
        st.dataframe(subset.groupby(OUTCOME_COL)[attr].describe().round(2), width='stretch')
    else:
        subset[attr] = readable_values(attr, subset[attr])
        top_feats = subset[attr].value_counts().head(12).index
        subset = subset[subset[attr].isin(top_feats)]
        share_table = pd.crosstab(subset[attr], subset[OUTCOME_COL], normalize='index').mul(100).round(1)
        share_table = share_table.reindex(columns=[fld for fld in RESULT_CLASSES if fld in share_table.columns])
        chart = px.bar(share_table, barmode='stack', orientation='h', color_discrete_map=OUTCOME_COLOURS, title=f'Outcome share (%) by {attr}', labels={'value': '%', attr: ''})
        st.plotly_chart(chart, width='stretch')
        st.dataframe(pd.crosstab(subset[attr], subset[OUTCOME_COL], margins=True), width='stretch')
    st.subheader('Result by stream')
    share_table = pd.crosstab(readable_values('Stream', al_df['Stream']), al_df[OUTCOME_COL], normalize='index').mul(100)
    share_table = share_table.reindex(columns=[fld for fld in RESULT_CLASSES if fld in share_table.columns]).round(1)
    st.plotly_chart(px.bar(share_table, barmode='stack', orientation='h', color_discrete_map=OUTCOME_COLOURS, labels={'value': '%', 'index': ''}), width='stretch')
    with st.expander('View raw data'):
        st.dataframe(al_df, width='stretch')
elif nav_choice == '🔮 Predict Student':
    banner("🔮 Predict a Student's Outcome", "Fill in the student's details. Fields are pre-filled with the most typical values.")
    form_values = {}
    with st.form('predict'):
        form_tabs = st.tabs(list(FORM_SECTIONS))
        for form_tab, (section_name, section_cols) in zip(form_tabs, FORM_SECTIONS.items()):
            with form_tab:
                form_grid = st.columns(3)
                for i, fld in enumerate([fld for fld in section_cols if fld in bundle.features]):
                    with form_grid[i % 3]:
                        if fld in NUMERIC_FIELDS:
                            low, high = bundle.ranges[fld]
                            whole_numbers = float(al_df[fld].dropna().mod(1).abs().sum()) == 0
                            if whole_numbers:
                                form_values[fld] = st.number_input(fld, int(low), int(high) + 10, int(bundle.defaults[fld]), step=1)
                            else:
                                form_values[fld] = st.number_input(fld, float(min(low, 0.0)), float(high) * 1.2 + 1, float(round(bundle.defaults[fld], 2)), step=0.1, format='%.2f')
                        else:
                            choices = bundle.options[fld]
                            form_values[fld] = st.selectbox(fld, choices, index=choices.index(bundle.defaults[fld]), format_func=lambda v, fld=fld: decode_value(fld, v))
        form_sent = st.form_submit_button('🔮 Predict outcome', type='primary', width='stretch')
    if form_sent:
        input_row = pd.DataFrame([{fld: form_values.get(fld, bundle.defaults[fld]) for fld in bundle.features}])
        probs = bundle.model.predict_proba(input_row)[0]
        class_names = list(bundle.model.classes_)
        predicted = class_names[probs.argmax()]
        heading, guidance = OUTCOME_GUIDANCE.get(predicted, (predicted, ''))
        st.markdown(f"""<div class="result" style="background:{OUTCOME_COLOURS.get(predicted, '#3A6FD8')}"><p style="margin:0;opacity:.9">Predicted outcome</p><h2>{predicted} — {probs.max():.0%} confidence</h2><p style="margin:.3rem 0 0">{heading}</p></div>""", unsafe_allow_html=True)
        left_col, right_col = st.columns([1, 1])
        with left_col:
            chart = go.Figure(go.Bar(x=[pv * 100 for pv in probs], y=class_names, orientation='h', marker_color=[OUTCOME_COLOURS.get(fld) for fld in class_names], text=[f'{pv:.1%}' for pv in probs], textposition='auto'))
            chart.update_layout(title='Probability per outcome', xaxis_title='%', xaxis_range=[0, 100], height=280, margin=dict(t=40, b=10))
            st.plotly_chart(chart, width='stretch')
        with right_col:
            st.markdown('#### 💡 Recommendation')
            st.write(guidance)
            cgt_val = form_values.get('CGT score')
            metric_a, metric_b = st.columns(2)
            metric_a.metric('CGT score', f'{cgt_val:.0f}' if cgt_val is not None else '—', help='Common General Test; 30 or more is required for university entry.')
            metric_b.metric('General English', decode_value('General English', form_values.get('General English')))
            if cgt_val is not None and cgt_val < 30:
                st.warning('CGT score is below 30 — the minimum needed for university entrance.')
            if form_values.get('General English') in (-1, 0):
                st.warning('General English is failed/absent — a strong signal of weak results.')
            chosen_subjects = {form_values.get(f'Subject {i}') for i in (1, 2, 3)}
            if len(chosen_subjects) < 3:
                st.warning('The same subject is selected more than once.')
elif nav_choice == '📁 Batch Prediction':
    banner('📁 Batch Prediction', 'Upload a CSV / Excel file of students to predict all at once.')
    st.write('The file needs the same 9 attribute columns as the training data (no Target column needed).')
    dl_a, dl_b = st.columns(2)
    dl_a.download_button('⬇️ Sample students (Excel)', SAMPLES_FOLDER.joinpath('students_to_predict.xlsx').read_bytes(), 'students_to_predict.xlsx', width='stretch')
    dl_b.download_button('⬇️ Empty template (CSV)', pd.DataFrame(columns=bundle.features).to_csv(index=False).encode(), 'students_template.csv', 'text/csv', width='stretch')
    batch_file = st.file_uploader('Upload students file', type=['csv', 'xlsx', 'xls'], key='batch')
    if batch_file:
        try:
            batch_df = load_table_file(batch_file)
            missing_cols = [fld for fld in bundle.features if fld not in batch_df.columns]
            if missing_cols:
                st.error(f"Missing columns: {', '.join(missing_cols)}")
            else:
                batch_x = batch_df[bundle.features].apply(pd.to_numeric, errors='coerce').fillna(0)
                probs = bundle.model.predict_proba(batch_x)
                scored = batch_df.copy()
                scored['Predicted'] = bundle.model.predict(batch_x)
                scored['Confidence'] = probs.max(axis=1).round(3)
                for i, fld in enumerate(bundle.model.classes_):
                    scored[f'P({fld})'] = probs[:, i].round(3)
                class_freq = scored['Predicted'].value_counts()
                section_cols = st.columns(len(class_freq) + 1)
                section_cols[0].metric('Students', len(scored))
                for attr, (k, v) in zip(section_cols[1:], class_freq.items()):
                    attr.metric(k, v)
                lead_cols = ['Predicted', 'Confidence'] + [f'P({fld})' for fld in bundle.model.classes_]
                st.dataframe(scored[lead_cols + [fld for fld in scored.columns if fld not in lead_cols]], width='stretch')
                st.download_button('⬇️ Download predictions', scored.to_csv(index=False).encode(), 'predictions.csv', 'text/csv', type='primary')
        except Exception as exc:
            st.error(f'Could not read file: {exc}')
elif nav_choice == '📈 Model Performance':
    banner('📈 Model Performance', 'How well the models predict on unseen students (20% test set).')
    kpi_1, kpi_2, kpi_3 = st.columns(3)
    kpi_1.markdown(f'**Best model**\n\n### ⭐ {top_model_name}')
    kpi_2.metric('Best accuracy', f"{bundle.models[top_model_name]['accuracy']:.1%}")
    kpi_3.metric('Models compared', len(ranked_names))
    st.subheader(f'Comparison of {len(ranked_names)} models')
    ranking = bundle.leaderboard
    long_form = ranking.melt(id_vars='Model', value_vars=['Accuracy', 'F1 (macro)'], var_name='Metric', value_name='Score')
    chart = px.bar(long_form, x='Score', y='Model', color='Metric', barmode='group', orientation='h', text=long_form['Score'].map(lambda v: f'{v:.1%}'), category_orders={'Model': ranked_names}, color_discrete_map={'Accuracy': '#3A6FD8', 'F1 (macro)': '#9DB8EE'})
    chart.update_layout(height=520, xaxis_range=[0, 1], xaxis_tickformat='.0%', yaxis_title='', legend_title='', margin=dict(t=20))
    st.plotly_chart(chart, width='stretch')
    render_leaderboard(ranking)
    st.caption('Accuracy = share of test students predicted correctly. F1 (macro) averages the 3 classes equally, so it also rewards getting the minority *High Achiever* class right.')
    st.subheader('Inspect a model')
    inspect_choice = st.selectbox('Model', ranked_names, index=ranked_names.index(bundle.model_name))
    bundle = bundle.use(inspect_choice)
    left_col, right_col = st.columns(2)
    with left_col:
        label_list = list(bundle.model.classes_)
        chart = px.imshow(bundle.confusion, x=label_list, y=label_list, text_auto=True, color_continuous_scale='Blues', title='Confusion matrix', labels={'x': 'Predicted', 'y': 'Actual', 'color': 'Students'}, aspect='auto')
        chart.update_layout(height=400)
        st.plotly_chart(chart, width='stretch')
    with right_col:
        class_rep = bundle.report.loc[[fld for fld in label_list if fld in bundle.report.index], ['precision', 'recall', 'f1-score', 'support']]
        st.markdown('#### Per-class metrics')
        st.dataframe(class_rep.style.format({'precision': '{:.2f}', 'recall': '{:.2f}', 'f1-score': '{:.2f}', 'support': '{:.0f}'}), width='stretch')
        st.caption('*Qualified* sits between the other two profiles, and *High Achiever* is the smallest class, so these are the hardest to separate.')
    top_feats = bundle.importance.head(15)[::-1]
    chart = px.bar(x=top_feats.values, y=top_feats.index, orientation='h', title='Feature importance (Random Forest)', labels={'x': 'Importance', 'y': ''})
    chart.update_traces(marker_color='#3A6FD8')
    st.plotly_chart(chart, width='stretch')
else:
    banner('🧩 Ensemble Learning Insights', 'Bagging vs boosting vs stacking vs hybrid — what the results say and why the models behave this way.')
    ranking = bundle.leaderboard.copy()
    all_models = bundle.models
    acc_of = lambda count_n: all_models[count_n]['accuracy']
    f1_of = lambda count_n: all_models[count_n]['f1']
    y_te = bundle.y_test

    def class_recall(item_name: str, cls_name: str) -> float:
        class_mask = y_te == cls_name
        return float((all_models[item_name]['pred'][class_mask] == cls_name).mean()) if class_mask.any() else float('nan')
    st.subheader('1. Which ensemble family works best?')
    family_table = ranking.groupby('Family').agg(**{'Best accuracy': ('Accuracy', 'max'), 'Mean accuracy': ('Accuracy', 'mean'), 'Best F1 (macro)': ('F1 (macro)', 'max'), 'Models': ('Model', 'count')}).reindex([batch_file for batch_file in FAMILY_SEQUENCE if batch_file in set(ranking['Family'])])
    family_table['Best model'] = [ranking[ranking['Family'] == batch_file].iloc[0]['Model'] for batch_file in family_table.index]
    left_col, right_col = st.columns([3, 2])
    with left_col:
        long_form = family_table.reset_index().melt(id_vars='Family', value_vars=['Best accuracy', 'Best F1 (macro)'])
        chart = px.bar(long_form, x='Family', y='value', color='variable', barmode='group', text=long_form['value'].map(lambda v: f'{v:.1%}'), color_discrete_sequence=['#3A6FD8', '#9DB8EE'], labels={'value': '', 'variable': '', 'Family': ''})
        low = max(0.0, float(long_form['value'].min()) - 0.1)
        chart.update_layout(yaxis_range=[low, 1], yaxis_tickformat='.0%', height=380, margin=dict(t=10))
        st.plotly_chart(chart, width='stretch')
    with right_col:
        st.dataframe(family_table.style.format({'Best accuracy': '{:.2%}', 'Mean accuracy': '{:.2%}', 'Best F1 (macro)': '{:.3f}'}), width='stretch')
        lead_family = family_table['Best F1 (macro)'].idxmax()
        st.success(f"Best family by F1: **{lead_family}** ({family_table.loc[lead_family, 'Best model']}). Overall best model: **{top_model_name}** ({acc_of(top_model_name):.1%} accuracy).")
    st.subheader('2. Bagging vs Boosting — Random Forest vs Gradient Boosting')
    rf_key, gb_key = ('Random Forest', 'Gradient Boosting')
    present_classes = [fld for fld in RESULT_CLASSES if fld in set(y_te)]
    rf_gb_table = pd.DataFrame({'Metric': ['Accuracy', 'F1 (macro)'] + [f'Recall – {fld}' for fld in present_classes] + ['Train time (s)'], rf_key: [acc_of(rf_key), f1_of(rf_key)] + [class_recall(rf_key, fld) for fld in present_classes] + [all_models[rf_key]['train_time']], gb_key: [acc_of(gb_key), f1_of(gb_key)] + [class_recall(gb_key, fld) for fld in present_classes] + [all_models[gb_key]['train_time']]}).set_index('Metric')
    left_col, right_col = st.columns([2, 3])
    with left_col:
        st.dataframe(rf_gb_table.style.format('{:.3f}'), width='stretch')
    with right_col:
        minority_cls = 'High Achiever' if 'High Achiever' in present_classes else present_classes[0]
        majority_cls = 'Qualified' if 'Qualified' in present_classes else present_classes[-1]
        st.markdown(f'**What we see:** Random Forest {acc_of(rf_key):.1%} vs Gradient Boosting {acc_of(gb_key):.1%} accuracy (macro-F1 {f1_of(rf_key):.3f} vs {f1_of(gb_key):.3f}). Recall on the small *{minority_cls}* class: RF {class_recall(rf_key, minority_cls):.0%} vs GB {class_recall(gb_key, minority_cls):.0%}; on the big *{majority_cls}* class: RF {class_recall(rf_key, majority_cls):.0%} vs GB {class_recall(gb_key, majority_cls):.0%}.\n\n**Why:**\n- **Bagging (Random Forest)** trains many trees *in parallel* on bootstrap samples and averages them. Averaging **reduces variance**, which matters here because the pre-exam features are noisy. The average is pulled towards the majority class, so RF is very good on *{majority_cls}* but finds fewer *{minority_cls}* students. Fully grown trees overfitted (≈52% accuracy); limiting leaf size (min 20 students) fixed this.\n- **Boosting (Gradient Boosting)** builds shallow trees *one after another*, each fitting the errors of the previous ones. This **reduces bias** and focuses on the hard students, so it recovers more of the rare *{minority_cls}* class, at the cost of more mistakes on the majority class and higher sensitivity to tuning.')
    st.subheader('3. Stacking & Voting — combining different models')
    base_models = ['Random Forest', 'Extra Trees', 'Gradient Boosting', 'K-Nearest Neighbors']
    ensemble_models = ['Stacking', 'Voting (soft)']
    stack_table = pd.DataFrame({'Model': ensemble_models + base_models})
    stack_table['Accuracy'] = stack_table['Model'].map(acc_of)
    stack_table['F1 (macro)'] = stack_table['Model'].map(f1_of)
    stack_table['Type'] = stack_table['Model'].isin(ensemble_models).map({True: 'Ensemble of models', False: 'Base model'})
    left_col, right_col = st.columns([3, 2])
    with left_col:
        chart = px.bar(stack_table, x='Model', y='F1 (macro)', color='Type', text=stack_table['F1 (macro)'].map(lambda v: f'{v:.3f}'), color_discrete_map={'Ensemble of models': '#29A36A', 'Base model': '#9DB8EE'}, labels={'Type': '', 'Model': ''}, title='Stacking vs its base models')
        chart.update_layout(yaxis_range=[max(0.0, stack_table['F1 (macro)'].min() - 0.1), None], height=380, margin=dict(t=40))
        st.plotly_chart(chart, width='stretch')
    with right_col:
        compare_names = ['Random Forest', 'Gradient Boosting', 'K-Nearest Neighbors', 'Logistic Regression']
        disagree = pd.DataFrame([[float((all_models[first]['pred'] != all_models[second]['pred']).mean()) for second in compare_names] for first in compare_names], index=compare_names, columns=compare_names)
        chart = px.imshow(disagree, text_auto='.0%', color_continuous_scale='Greens', title='How often two models disagree', aspect='auto')
        chart.update_layout(height=380, margin=dict(t=40), coloraxis_showscale=False)
        st.plotly_chart(chart, width='stretch')
    best_base_model = max(base_models, key=f1_of)
    f1_gain = f1_of('Stacking') - f1_of(best_base_model)
    st.markdown(f"**What we see:** Stacking reaches **{acc_of('Stacking'):.1%}** accuracy and **{f1_of('Stacking'):.3f}** macro-F1 ({f1_gain:+.3f} F1 compared with its best base model, {best_base_model}). Soft voting gets {acc_of('Voting (soft)'):.1%}.\n\n**Why:** the base models make *different* mistakes (see the disagreement matrix). **Soft voting** simply averages their probabilities. **Stacking** goes further: a Logistic-Regression *meta-learner* is trained on the base models' out-of-fold probabilities (5-fold CV), so it learns *which model to trust for which kind of student*. Gains are modest because the strongest base models are already tree ensembles that agree on most students — combining models helps most when their errors are uncorrelated.")
    st.subheader('4. Hybridization — Random Forest + ANN')
    hybrid_key = 'Hybrid RF + ANN'
    hybrid_table = pd.DataFrame({'Model': ['Random Forest', 'Neural Network (MLP)', hybrid_key]})
    hybrid_table['Accuracy'] = hybrid_table['Model'].map(acc_of)
    hybrid_table['F1 (macro)'] = hybrid_table['Model'].map(f1_of)
    left_col, right_col = st.columns([2, 3])
    with left_col:
        st.dataframe(hybrid_table.style.format({'Accuracy': '{:.2%}', 'F1 (macro)': '{:.3f}'}), width='stretch', hide_index=True)
    with right_col:
        delta_ann = acc_of(hybrid_key) - acc_of('Neural Network (MLP)')
        delta_rf = acc_of(hybrid_key) - acc_of('Random Forest')
        st.markdown(f"""**How it works:** the Random Forest's out-of-fold class probabilities are added as 3 extra inputs to a neural network, which is trained on the scaled original features + these "forest opinions".\n\n**What we see:** the hybrid is {delta_ann * 100:+.1f} points vs the ANN alone and {delta_rf * 100:+.1f} points vs the Random Forest alone. The forest's probabilities give the ANN a compact, non-linear summary of the subject combinations, so the ANN improves. It cannot go far beyond the forest because both models see the same 9 pre-exam features — the remaining error is information that simply is not in the data (study effort, school, tuition, exam-day performance). The ANN also needs strong regularisation (L2 alpha = 1.0) so it does not over-trust the RF inputs.""")
    st.subheader('5. Why the models behave this way')
    left_col, right_col = st.columns(2)
    with left_col:
        top_feats = bundle.importance.head(8)[::-1]
        chart = px.bar(x=top_feats.values, y=top_feats.index, orientation='h', title='What drives the prediction (RF importance)', labels={'x': 'Importance', 'y': ''})
        chart.update_traces(marker_color='#3A6FD8')
        chart.update_layout(height=380)
        st.plotly_chart(chart, width='stretch')
    with right_col:
        chart = px.scatter(ranking, x='Train time (s)', y='F1 (macro)', color='Family', hover_name='Model', log_x=True, title='Quality vs training cost')
        chart.update_traces(marker_size=12)
        chart.update_layout(height=380)
        st.plotly_chart(chart, width='stretch')
    class_share = al_df[OUTCOME_COL].value_counts(normalize=True)
    top_three = ', '.join(bundle.importance.index[:3])
    st.markdown(f"- **What drives the result:** {top_three}. The **CGT** (Common General Test) measures general reasoning and is taken in the same exam, so it is the best single proxy for overall ability; **General English** works the same way. **Gender** matters too: in this data female candidates fail a subject far less often than male candidates. **Age** acts non-linearly: 20–21-year-olds (typically a second attempt) fail less often than first-time 19-year-olds but rarely become High Achievers, while candidates over 23 do much worse. Its importance is also inflated because impurity-based importance favours continuous features. The **subject combination** captures how hard each stream is to pass.\n- **Why accuracy is only ~60%:** the models only see 9 pre-exam attributes — not marks, school, district, tuition or effort. A large part of the A/L result is not in these features, so all models land close together (median accuracy {ranking['Accuracy'].median():.0%}, vs {class_share.max():.0%} for always guessing the largest class). This is a realistic, noisy problem — exactly where ensembles help.\n- **Class imbalance:** Qualified {class_share.get('Qualified', 0):.0%}, Not Qualified {class_share.get('Not Qualified', 0):.0%}, High Achiever {class_share.get('High Achiever', 0):.0%}. *High Achiever* is the smallest class and the hardest to find; macro-F1 is reported next to accuracy so it is not ignored.\n- **Single models have clear weak spots:** Naive Bayes assumes independent features (stream and subjects are strongly linked); KNN treats subject codes as distances; Logistic Regression can only draw linear boundaries over coded subjects; a single Decision Tree overfits. **Ensembles fix exactly these weaknesses** — bagging lowers variance, boosting lowers bias, stacking mixes different strengths.\n- **Cost vs gain:** Stacking and SVM are the most expensive to train; a regularised Random Forest gives top accuracy in about a second.")
    st.subheader('6. Conclusion')
    best_f1_model = ranking.sort_values('F1 (macro)', ascending=False).iloc[0]['Model']
    st.info(f'**{top_model_name}** has the best accuracy ({acc_of(top_model_name):.1%}) and **{best_f1_model}** the best macro-F1 ({f1_of(best_f1_model):.3f}) for predicting Sri Lankan A/L results from pre-exam information. Regularised bagging is the strongest on accuracy, boosting finds more of the rare *High Achiever* students, and stacking / the hybrid combine models that make different errors for a more balanced result. Every ensemble family beats the weakest single models (Naive Bayes, KNN, Logistic Regression).')
    st.caption('🤖 AI disclosure: code and dashboard generated with the help of Claude (Anthropic) and reviewed/tested by the student.')
