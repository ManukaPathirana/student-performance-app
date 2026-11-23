"""Student Academic Performance Prediction System — Streamlit app."""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import model as _model_mod

# When model.py changes (e.g. a new git push on Streamlit Cloud) the running server may
# still hold the old module; reload it and key the model cache on its hash.
MODEL_CODE_VERSION = hashlib.md5(Path(_model_mod.__file__).read_bytes()).hexdigest()
if getattr(_model_mod, "CODE_VERSION", None) != MODEL_CODE_VERSION:
    _model_mod = importlib.reload(_model_mod)
    _model_mod.CODE_VERSION = MODEL_CODE_VERSION

from model import (CLASSES, CONTINUOUS, GROUPS, TARGET, label_for, load_default,  # noqa: E402
                   read_table, train, DATA_PATH as DATA_CSV)

SAMPLE_DIR = DATA_CSV.parent.parent / "sample_data"

st.set_page_config(page_title="Student Academic Performance Prediction System",
                   page_icon="🎓", layout="wide")

COLORS = {"Dropout": "#E4572E", "Enrolled": "#F3A712", "Graduate": "#29A36A"}
ADVICE = {
    "Dropout": ("High risk of dropping out",
                "Schedule an academic advisor meeting, check financial situation "
                "(tuition / debts / scholarship) and monitor semester progress closely."),
    "Enrolled": ("Still enrolled — progress is slower than expected",
                 "Offer tutoring for units not yet approved and set clear semester goals."),
    "Graduate": ("On track to graduate",
                 "Keep up the good work. Consider mentoring or advanced opportunities."),
}

st.markdown("""
<style>
.block-container {padding-top: 2rem;}
.hero {padding: 1.4rem 1.6rem; border-radius: 14px; color: white;
       background: linear-gradient(120deg, #1F3B73 0%, #3A6FD8 100%); margin-bottom: 1.2rem;}
.hero h1 {color: white; margin: 0; font-size: 2rem;}
.hero p {margin: .3rem 0 0; opacity: .9;}
.result {padding: 1.2rem 1.4rem; border-radius: 14px; color: white; margin-bottom: .8rem;}
.result h2 {color: white; margin: 0;}
</style>""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- data/model
@st.cache_data(show_spinner=False)
def get_data(file_bytes: bytes | None, file_name: str | None) -> pd.DataFrame:
    if file_bytes is None:
        return load_default()
    import io
    buf = io.BytesIO(file_bytes)
    buf.name = file_name
    return read_table(buf)


@st.cache_resource(show_spinner="Training 10 models… (about a minute, first run only)")
def get_model(data_key: str, _df: pd.DataFrame, code_version: str = ""):
    # code_version = hash of model.py, so a redeploy with new model code retrains
    # instead of serving a stale cached result.
    return train(_df)


with st.sidebar:
    st.markdown("## 🎓 SAPPS")
    st.caption("Student Academic Performance Prediction System")
    page = st.radio("Navigate", ["🏠 Home", "🧠 Train Model", "📊 Data Explorer",
                                 "🔮 Predict Student", "📁 Batch Prediction",
                                 "📈 Model Performance"],
                    label_visibility="collapsed")
    st.divider()

# Training source: None = built-in dataset, else (bytes, filename) uploaded on Train page.
src = st.session_state.get("train_src")
if src:
    df = get_data(*src)
    res = get_model(hashlib.md5(src[0]).hexdigest(), df, MODEL_CODE_VERSION)
    src_label = src[1]
else:
    df = get_data(None, None)
    res = get_model("default", df, MODEL_CODE_VERSION)
    src_label = "Built-in dataset"

if not hasattr(res, "use"):  # result cached by an older version of model.py
    get_model.clear()
    st.rerun()

best_name = res.model_name
ranked = res.leaderboard["Model"].tolist()
with st.sidebar:
    chosen = st.selectbox(
        "Model used for predictions", ranked, index=0, key="chosen_model",
        format_func=lambda m: f"{m} — {res.models[m]['accuracy']:.1%}"
                              + (" ⭐ best" if m == best_name else ""))
    res = res.use(chosen)
    st.success(f"**Model:** {res.model_name}\n\n**Test accuracy:** {res.accuracy:.1%}\n\n"
               f"**Trained on:** {src_label} ({len(df):,} rows, {len(ranked)} models compared)")


def show_board(board: pd.DataFrame):
    """Leaderboard of all trained models; the active model's row is highlighted."""
    fmt = {"Accuracy": "{:.2%}", "F1 (macro)": "{:.3f}", "Precision (macro)": "{:.3f}",
           "Recall (macro)": "{:.3f}", "Train time (s)": "{:.2f}"}
    sty = board.style.format(fmt).apply(
        lambda r: ["background-color: #cde8d6" if r["Model"] == res.model_name else ""
                   for _ in r], axis=1)
    st.dataframe(sty, width="stretch", hide_index=True)


def hero(title: str, sub: str):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{sub}</p></div>',
                unsafe_allow_html=True)


def pretty(col: str, s: pd.Series) -> pd.Series:
    return s if col in CONTINUOUS else s.map(lambda v: label_for(col, v))


# --------------------------------------------------------------------------- pages
if page == "🏠 Home":
    hero("🎓 Student Academic Performance Prediction System",
         "Predict whether a student will Drop out, stay Enrolled or Graduate — "
         "using demographic, socio-economic and academic data.")
    counts = df[TARGET].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total students", f"{len(df):,}")
    for col, name in zip((c2, c3, c4), ("Graduate", "Enrolled", "Dropout")):
        n = counts.get(name, 0)
        col.metric(f"{name} ({n / len(df):.0%})", f"{n:,}")

    left, right = st.columns([1, 1])
    with left:
        fig = px.pie(names=counts.index, values=counts.values, hole=.55,
                     color=counts.index, color_discrete_map=COLORS,
                     title="Student outcome distribution")
        st.plotly_chart(fig, width="stretch")
    with right:
        top = res.importance.head(10)[::-1]
        fig = px.bar(x=top.values, y=top.index, orientation="h",
                     title="Top 10 factors influencing outcome",
                     labels={"x": "Importance", "y": ""})
        fig.update_traces(marker_color="#3A6FD8")
        st.plotly_chart(fig, width="stretch")

    st.subheader("How it works")
    a, b, c = st.columns(3)
    a.info("**1. Data** — 36 attributes per student: background, finances, "
           "and 1st/2nd semester results.")
    b.info(f"**2. Model** — 10 algorithms are compared; **{best_name}** "
           f"performed best ({res.models[best_name]['accuracy']:.1%} accuracy).")
    c.info("**3. Predict** — retrain on your own data in *Train Model*, then enter a student's details on the *Predict Student* "
           "page, or upload a file for batch prediction.")
    st.caption("Dataset: *Predict Students' Dropout and Academic Success* — "
               "UCI Machine Learning Repository (Realinho et al., 2021).")

elif page == "🧠 Train Model":
    hero("🧠 Train the Model",
         "Upload a student dataset (CSV or Excel) and train the prediction models on it.")
    st.markdown("#### 1. Get a dataset")
    st.write("The file needs the 36 student attribute columns plus a **Target** column "
             "(Dropout / Enrolled / Graduate). Download the sample dataset to see the format.")
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Sample dataset (Excel)",
                       SAMPLE_DIR.joinpath("student_training_dataset.xlsx").read_bytes(),
                       "student_training_dataset.xlsx", width="stretch")
    d2.download_button("⬇️ Sample dataset (CSV)", DATA_CSV.read_bytes(),
                       "student_training_dataset.csv", "text/csv", width="stretch")

    st.markdown("#### 2. Upload and train")
    up = st.file_uploader("Upload training dataset", type=["csv", "xlsx", "xls"], key="train_up")
    if up is not None:
        try:
            preview = get_data(up.getvalue(), up.name)
            missing = [c for c in res.features if c not in preview.columns]
            if TARGET not in preview.columns:
                st.error(f"The file has no '{TARGET}' column — the model needs it to learn.")
            elif missing:
                st.error(f"Missing columns: {', '.join(missing)}")
            else:
                vc = preview[TARGET].value_counts()
                st.success(f"✅ {up.name}: {len(preview):,} students, {preview.shape[1]} columns — "
                           + ", ".join(f"{k}: {v:,}" for k, v in vc.items()))
                st.dataframe(preview.head(10), width="stretch")
                if vc.min() < 5:
                    st.error("Each Target class needs at least 5 students to train.")
                elif st.button("🚀 Train models on this dataset", type="primary", width="stretch"):
                    with st.spinner("Training 10 models (Logistic Regression, Random Forest, SVM, "
                                    "Neural Network…). This takes about a minute."):
                        get_model(hashlib.md5(up.getvalue()).hexdigest(), preview, MODEL_CODE_VERSION)
                    st.session_state["train_src"] = (up.getvalue(), up.name)
                    st.session_state["just_trained"] = True
                    st.rerun()
        except Exception as exc:
            st.error(f"Could not read file: {exc}")

    if src:
        if st.button("↩️ Reset to built-in dataset"):
            st.session_state.pop("train_src", None)
            st.rerun()

    st.markdown("#### 3. Current model")
    if st.session_state.pop("just_trained", False):
        st.balloons()
        st.success(f"Training finished on **{src_label}**. All pages now use the new model.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training rows", f"{len(df):,}")
    c2.metric("Models trained", len(ranked))
    c3.metric("Best accuracy", f"{res.models[best_name]['accuracy']:.1%}")
    c4.metric("Best F1 (macro)", f"{res.models[best_name]['f1']:.3f}")
    show_board(res.leaderboard)
    st.caption(f"⭐ Best model: **{best_name}**. 80% of rows are used for training and 20% "
               "held out for testing. Pick a different model in the sidebar to use it for "
               "predictions. See *Model Performance* for details.")

elif page == "📊 Data Explorer":
    hero("📊 Data Explorer", "Explore how student attributes relate to academic outcomes.")
    feats = [c for c in df.columns if c != TARGET]
    col = st.selectbox("Choose an attribute", feats,
                       index=feats.index("Curricular units 2nd sem (approved)")
                       if "Curricular units 2nd sem (approved)" in feats else 0)
    tmp = df[[col, TARGET]].copy()
    if col in CONTINUOUS:
        fig = px.box(tmp, x=TARGET, y=col, color=TARGET, color_discrete_map=COLORS,
                     category_orders={TARGET: CLASSES}, points=False,
                     title=f"{col} by outcome")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(tmp.groupby(TARGET)[col].describe().round(2), width="stretch")
    else:
        tmp[col] = pretty(col, tmp[col])
        top = tmp[col].value_counts().head(12).index
        tmp = tmp[tmp[col].isin(top)]
        ct = pd.crosstab(tmp[col], tmp[TARGET], normalize="index").mul(100).round(1)
        ct = ct.reindex(columns=[c for c in CLASSES if c in ct.columns])
        fig = px.bar(ct, barmode="stack", orientation="h", color_discrete_map=COLORS,
                     title=f"Outcome share (%) by {col}", labels={"value": "%", col: ""})
        st.plotly_chart(fig, width="stretch")
        st.dataframe(pd.crosstab(tmp[col], tmp[TARGET], margins=True), width="stretch")

    st.subheader("Correlation of semester performance")
    sem = [c for c in df.columns if c.startswith("Curricular")]
    corr = df[sem].corr().round(2)
    corr.index = corr.columns = [c.replace("Curricular units ", "") for c in sem]
    st.plotly_chart(px.imshow(corr, color_continuous_scale="Blues", text_auto=True,
                              aspect="auto"), width="stretch")
    with st.expander("View raw data"):
        st.dataframe(df, width="stretch")

elif page == "🔮 Predict Student":
    hero("🔮 Predict a Student's Outcome",
         "Fill in the student's details. Fields are pre-filled with the most typical values.")
    values = {}
    with st.form("predict"):
        tabs = st.tabs(list(GROUPS))
        for tab, (group, cols) in zip(tabs, GROUPS.items()):
            with tab:
                grid = st.columns(3)
                for i, c in enumerate([c for c in cols if c in res.features]):
                    with grid[i % 3]:
                        if c in CONTINUOUS:
                            lo, hi = res.ranges[c]
                            is_int = float(df[c].dropna().mod(1).abs().sum()) == 0
                            if is_int:
                                values[c] = st.number_input(
                                    c, int(lo), int(hi) + 10, int(res.defaults[c]), step=1)
                            else:
                                values[c] = st.number_input(
                                    c, float(min(lo, 0.0)), float(hi) * 1.2 + 1,
                                    float(round(res.defaults[c], 2)), step=0.1, format="%.2f")
                        else:
                            opts = res.options[c]
                            values[c] = st.selectbox(
                                c, opts, index=opts.index(res.defaults[c]),
                                format_func=lambda v, c=c: label_for(c, v))
        submitted = st.form_submit_button("🔮 Predict outcome", type="primary",
                                          width="stretch")

    if submitted:
        row = pd.DataFrame([{c: values.get(c, res.defaults[c]) for c in res.features}])
        proba = res.model.predict_proba(row)[0]
        classes = list(res.model.classes_)
        pred = classes[proba.argmax()]
        title, advice = ADVICE.get(pred, (pred, ""))
        st.markdown(
            f'<div class="result" style="background:{COLORS.get(pred, "#3A6FD8")}">'
            f'<p style="margin:0;opacity:.9">Predicted outcome</p>'
            f'<h2>{pred} — {proba.max():.0%} confidence</h2>'
            f'<p style="margin:.3rem 0 0">{title}</p></div>', unsafe_allow_html=True)
        l, r = st.columns([1, 1])
        with l:
            fig = go.Figure(go.Bar(x=[p * 100 for p in proba], y=classes, orientation="h",
                                   marker_color=[COLORS.get(c) for c in classes],
                                   text=[f"{p:.1%}" for p in proba], textposition="auto"))
            fig.update_layout(title="Probability per outcome", xaxis_title="%",
                              xaxis_range=[0, 100], height=280, margin=dict(t=40, b=10))
            st.plotly_chart(fig, width="stretch")
        with r:
            st.markdown("#### 💡 Recommendation")
            st.write(advice)
            e1 = values.get("Curricular units 1st sem (enrolled)", 0) or 0
            a1 = values.get("Curricular units 1st sem (approved)", 0) or 0
            e2 = values.get("Curricular units 2nd sem (enrolled)", 0) or 0
            a2 = values.get("Curricular units 2nd sem (approved)", 0) or 0
            m1, m2 = st.columns(2)
            m1.metric("1st sem pass rate", f"{a1 / e1:.0%}" if e1 else "—")
            m2.metric("2nd sem pass rate", f"{a2 / e2:.0%}" if e2 else "—")
            if values.get("Tuition fees up to date") == 0:
                st.warning("Tuition fees are not up to date — a strong dropout signal.")
            if values.get("Debtor") == 1:
                st.warning("Student is a debtor.")

elif page == "📁 Batch Prediction":
    hero("📁 Batch Prediction", "Upload a CSV / Excel file of students to predict all at once.")
    st.write("The file needs the same 36 attribute columns as the training data "
             "(no Target column needed).")
    b1, b2 = st.columns(2)
    b1.download_button("⬇️ Sample students (Excel)",
                       SAMPLE_DIR.joinpath("students_to_predict.xlsx").read_bytes(),
                       "students_to_predict.xlsx", width="stretch")
    b2.download_button("⬇️ Empty template (CSV)",
                       pd.DataFrame(columns=res.features).to_csv(index=False).encode(),
                       "students_template.csv", "text/csv", width="stretch")
    f = st.file_uploader("Upload students file", type=["csv", "xlsx", "xls"], key="batch")
    if f:
        try:
            data = read_table(f)
            missing = [c for c in res.features if c not in data.columns]
            if missing:
                st.error(f"Missing columns: {', '.join(missing)}")
            else:
                X = data[res.features].apply(pd.to_numeric, errors="coerce").fillna(0)
                proba = res.model.predict_proba(X)
                out = data.copy()
                out["Predicted"] = res.model.predict(X)
                out["Confidence"] = proba.max(axis=1).round(3)
                for i, c in enumerate(res.model.classes_):
                    out[f"P({c})"] = proba[:, i].round(3)
                vc = out["Predicted"].value_counts()
                cols = st.columns(len(vc) + 1)
                cols[0].metric("Students", len(out))
                for col, (k, v) in zip(cols[1:], vc.items()):
                    col.metric(k, v)
                lead = ["Predicted", "Confidence"] + [f"P({c})" for c in res.model.classes_]
                st.dataframe(out[lead + [c for c in out.columns if c not in lead]],
                             width="stretch")
                st.download_button("⬇️ Download predictions", out.to_csv(index=False).encode(),
                                   "predictions.csv", "text/csv", type="primary")
        except Exception as exc:
            st.error(f"Could not read file: {exc}")

else:  # Model Performance
    hero("📈 Model Performance", "How well the models predict on unseen students (20% test set).")
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"**Best model**\n\n### ⭐ {best_name}")
    c2.metric("Best accuracy", f"{res.models[best_name]['accuracy']:.1%}")
    c3.metric("Models compared", len(ranked))

    st.subheader(f"Comparison of {len(ranked)} models")
    board = res.leaderboard
    long = board.melt(id_vars="Model", value_vars=["Accuracy", "F1 (macro)"],
                      var_name="Metric", value_name="Score")
    fig = px.bar(long, x="Score", y="Model", color="Metric", barmode="group",
                 orientation="h", text=long["Score"].map(lambda v: f"{v:.1%}"),
                 category_orders={"Model": ranked},
                 color_discrete_map={"Accuracy": "#3A6FD8", "F1 (macro)": "#9DB8EE"})
    fig.update_layout(height=520, xaxis_range=[0, 1], xaxis_tickformat=".0%",
                      yaxis_title="", legend_title="", margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")
    show_board(board)
    st.caption("Accuracy = share of test students predicted correctly. F1 (macro) averages "
               "the 3 classes equally, so it also rewards getting *Enrolled* right.")

    st.subheader("Inspect a model")
    inspect = st.selectbox("Model", ranked, index=ranked.index(res.model_name))
    res = res.use(inspect)
    l, r = st.columns(2)
    with l:
        labels = list(res.model.classes_)
        fig = px.imshow(res.confusion, x=labels, y=labels, text_auto=True,
                        color_continuous_scale="Blues", title="Confusion matrix",
                        labels={"x": "Predicted", "y": "Actual", "color": "Students"},
                        aspect="auto")
        fig.update_layout(height=400)
        st.plotly_chart(fig, width="stretch")
    with r:
        rep = res.report.loc[[c for c in labels if c in res.report.index],
                             ["precision", "recall", "f1-score", "support"]]
        st.markdown("#### Per-class metrics")
        st.dataframe(rep.style.format({"precision": "{:.2f}", "recall": "{:.2f}",
                                       "f1-score": "{:.2f}", "support": "{:.0f}"}),
                     width="stretch")
        st.caption("*Enrolled* is the hardest class — those students sit between "
                   "the dropout and graduate profiles.")
    top = res.importance.head(15)[::-1]
    fig = px.bar(x=top.values, y=top.index, orientation="h", title="Feature importance (Random Forest)",
                 labels={"x": "Importance", "y": ""})
    fig.update_traces(marker_color="#3A6FD8")
    st.plotly_chart(fig, width="stretch")
