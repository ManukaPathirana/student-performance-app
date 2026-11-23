"""Student Academic Performance Prediction System — Streamlit app."""
from __future__ import annotations

import hashlib

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from model import (CLASSES, CONTINUOUS, GROUPS, TARGET, label_for, load_default,
                   read_table, train)

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


@st.cache_resource(show_spinner="Training models… (first run only)")
def get_model(data_key: str, _df: pd.DataFrame):
    return train(_df)


with st.sidebar:
    st.markdown("## 🎓 SAPPS")
    st.caption("Student Academic Performance Prediction System")
    page = st.radio("Navigate", ["🏠 Home", "📊 Data Explorer", "🔮 Predict Student",
                                 "📁 Batch Prediction", "📈 Model Performance"],
                    label_visibility="collapsed")
    st.divider()
    with st.expander("⚙️ Use your own dataset"):
        up = st.file_uploader("CSV or Excel with the same columns + 'Target'",
                              type=["csv", "xlsx", "xls"])
        st.caption("Leave empty to use the built-in dataset (4,424 students).")

file_bytes = up.getvalue() if up else None
try:
    df = get_data(file_bytes, up.name if up else None)
    key = hashlib.md5(file_bytes).hexdigest() if file_bytes else "default"
    res = get_model(key, df)
except Exception as exc:  # bad upload -> fall back to default
    st.sidebar.error(f"Could not use uploaded file: {exc}")
    df = get_data(None, None)
    res = get_model("default", df)

with st.sidebar:
    st.success(f"**Model:** {res.model_name}\n\n**Test accuracy:** {res.accuracy:.1%}")


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
    b.info(f"**2. Model** — 3 algorithms are compared; **{res.model_name}** "
           f"performed best ({res.accuracy:.1%} accuracy).")
    c.info("**3. Predict** — enter a student's details on the *Predict Student* "
           "page, or upload a file for batch prediction.")
    st.caption("Dataset: *Predict Students' Dropout and Academic Success* — "
               "UCI Machine Learning Repository (Realinho et al., 2021).")

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
    tmpl = df[res.features].head(5)
    st.download_button("⬇️ Download template (5 sample rows)",
                       tmpl.to_csv(index=False).encode(), "students_template.csv", "text/csv")
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
    c1.markdown(f"**Selected model**\n\n### {res.model_name}")
    c2.metric("Accuracy", f"{res.accuracy:.1%}")
    c3.metric("F1 score (macro)", f"{res.f1_macro:.3f}")

    st.subheader("Model comparison")
    st.dataframe(res.leaderboard.style.format({"Accuracy": "{:.2%}", "F1 (macro)": "{:.3f}"})
                 .highlight_max(subset=["Accuracy"], color="#cde8d6"),
                 width="stretch", hide_index=True)
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
