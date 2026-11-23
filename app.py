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

from model import (CLASSES, CONTINUOUS, FAMILY_ORDER, GROUPS, TARGET, label_for,  # noqa: E402
                   load_default,
                   read_table, train, DATA_PATH as DATA_CSV)

SAMPLE_DIR = DATA_CSV.parent.parent / "sample_data"

st.set_page_config(page_title="Student Academic Performance Prediction System",
                   page_icon="🎓", layout="wide")

COLORS = {"Not Qualified": "#E4572E", "Qualified": "#F3A712", "High Achiever": "#29A36A"}
ADVICE = {
    "Not Qualified": ("At risk of failing at least one A/L subject",
                      "Give extra support in the weakest subject (past papers, revision "
                      "classes), check attendance, and plan for a repeat attempt if needed."),
    "Qualified": ("Likely to pass all three subjects (university-eligible range)",
                  "Push for higher grades to improve the Z-score — selection to popular "
                  "university courses depends on it."),
    "High Achiever": ("Likely to pass all three subjects with a high Z-score (≥ 1.0)",
                      "Strong candidate for competitive university courses. Keep up the "
                      "momentum and consider mentoring peers."),
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


@st.cache_resource(show_spinner="Training 14 models (bagging, boosting, stacking, hybrid)… about 1–2 minutes, first run only")
def get_model(data_key: str, _df: pd.DataFrame, code_version: str = ""):
    # code_version = hash of model.py, so a redeploy with new model code retrains
    # instead of serving a stale cached result.
    return train(_df)


with st.sidebar:
    st.markdown("## 🎓 SAPPS")
    st.caption("Student Academic Performance Prediction System")
    page = st.radio("Navigate", ["🏠 Home", "🧠 Train Model", "📊 Data Explorer",
                                 "🔮 Predict Student", "📁 Batch Prediction",
                                 "📈 Model Performance", "🧩 Ensemble Insights"],
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
    st.caption("🤖 **AI disclosure:** the code and the dashboard UI of this project were "
               "generated with the help of Claude (Anthropic), a generative-AI tool, and "
               "then reviewed, run and tested by the student.")


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
         "Predict a Sri Lankan student's G.C.E. Advanced Level result — Not Qualified, "
         "Qualified or High Achiever — from subjects, CGT, General English and profile.")
    counts = df[TARGET].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total students", f"{len(df):,}")
    for col, name in zip((c2, c3, c4), ("High Achiever", "Qualified", "Not Qualified")):
        n = counts.get(name, 0)
        col.metric(f"{name} ({n / len(df):.0%})", f"{n:,}")

    left, right = st.columns([1, 1])
    with left:
        fig = px.pie(names=counts.index, values=counts.values, hole=.55,
                     color=counts.index, color_discrete_map=COLORS,
                     title="A/L result distribution")
        st.plotly_chart(fig, width="stretch")
    with right:
        top = res.importance.head(10)[::-1]
        fig = px.bar(x=top.values, y=top.index, orientation="h",
                     title="Factors influencing the A/L result",
                     labels={"x": "Importance", "y": ""})
        fig.update_traces(marker_color="#3A6FD8")
        st.plotly_chart(fig, width="stretch")

    st.subheader("How it works")
    a, b, c = st.columns(3)
    a.info("**1. Data** — Sri Lankan G.C.E. A/L 2020 results (Department of Examinations): "
           f"{len(df):,} students sampled from 256,606 candidates. 9 attributes per student.")
    b.info(f"**2. Model** — {len(ranked)} models (bagging, boosting, stacking, voting, "
           f"hybrid RF+ANN and single-model baselines) are compared; **{best_name}** "
           f"performed best ({res.models[best_name]['accuracy']:.1%} accuracy).")
    c.info("**3. Predict** — retrain on your own data in *Train Model*, then enter a student's details on the *Predict Student* "
           "page, or upload a file for batch prediction.")
    st.caption("Dataset: G.C.E. Advanced Level Examination 2020 results, Sri Lanka "
               "(public Kaggle upload of Department of Examinations data). "
               "Not Qualified = failed ≥1 subject · Qualified = passed all 3, Z < 1.0 · "
               "High Achiever = passed all 3, Z ≥ 1.0.")

elif page == "🧠 Train Model":
    hero("🧠 Train the Model",
         "Upload a student dataset (CSV or Excel) and train the prediction models on it.")
    st.markdown("#### 1. Get a dataset")
    st.write("The file needs the 9 student attribute columns plus a **Target** column "
             "(Not Qualified / Qualified / High Achiever). Download the sample dataset to see "
             "the format.")
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
                    with st.spinner("Training 14 models (bagging, boosting, stacking, voting, hybrid RF+ANN, "
                                    "baselines). This takes 1–2 minutes."):
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
                       index=feats.index("CGT score") if "CGT score" in feats else 0)
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

    st.subheader("Result by stream")
    ct = pd.crosstab(pretty("Stream", df["Stream"]), df[TARGET], normalize="index").mul(100)
    ct = ct.reindex(columns=[c for c in CLASSES if c in ct.columns]).round(1)
    st.plotly_chart(px.bar(ct, barmode="stack", orientation="h", color_discrete_map=COLORS,
                           labels={"value": "%", "index": ""}), width="stretch")
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
            cgt = values.get("CGT score")
            m1, m2 = st.columns(2)
            m1.metric("CGT score", f"{cgt:.0f}" if cgt is not None else "—",
                      help="Common General Test; 30 or more is required for university entry.")
            m2.metric("General English", label_for("General English",
                                                   values.get("General English")))
            if cgt is not None and cgt < 30:
                st.warning("CGT score is below 30 — the minimum needed for university entrance.")
            if values.get("General English") in (-1, 0):
                st.warning("General English is failed/absent — a strong signal of weak results.")
            subs = {values.get(f"Subject {i}") for i in (1, 2, 3)}
            if len(subs) < 3:
                st.warning("The same subject is selected more than once.")

elif page == "📁 Batch Prediction":
    hero("📁 Batch Prediction", "Upload a CSV / Excel file of students to predict all at once.")
    st.write("The file needs the same 9 attribute columns as the training data "
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

elif page == "📈 Model Performance":
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
               "the 3 classes equally, so it also rewards getting the minority *High Achiever* class right.")

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
        st.caption("*Qualified* sits between the other two profiles, and *High Achiever* is "
                   "the smallest class, so these are the hardest to separate.")
    top = res.importance.head(15)[::-1]
    fig = px.bar(x=top.values, y=top.index, orientation="h", title="Feature importance (Random Forest)",
                 labels={"x": "Importance", "y": ""})
    fig.update_traces(marker_color="#3A6FD8")
    st.plotly_chart(fig, width="stretch")

else:  # Ensemble Insights
    hero("🧩 Ensemble Learning Insights",
         "Bagging vs boosting vs stacking vs hybrid — what the results say and why "
         "the models behave this way.")
    board = res.leaderboard.copy()
    M = res.models
    acc = lambda n: M[n]["accuracy"]  # noqa: E731
    f1 = lambda n: M[n]["f1"]  # noqa: E731
    y_te = res.y_test

    def recall_of(name: str, cls: str) -> float:
        mask = y_te == cls
        return float((M[name]["pred"][mask] == cls).mean()) if mask.any() else float("nan")

    # ---------------------------------------------------------------- family overview
    st.subheader("1. Which ensemble family works best?")
    fam = (board.groupby("Family")
           .agg(**{"Best accuracy": ("Accuracy", "max"), "Mean accuracy": ("Accuracy", "mean"),
                   "Best F1 (macro)": ("F1 (macro)", "max"), "Models": ("Model", "count")})
           .reindex([f for f in FAMILY_ORDER if f in set(board["Family"])]))
    fam["Best model"] = [board[board["Family"] == f].iloc[0]["Model"] for f in fam.index]
    l, r = st.columns([3, 2])
    with l:
        long = fam.reset_index().melt(id_vars="Family",
                                      value_vars=["Best accuracy", "Best F1 (macro)"])
        fig = px.bar(long, x="Family", y="value", color="variable", barmode="group",
                     text=long["value"].map(lambda v: f"{v:.1%}"),
                     color_discrete_sequence=["#3A6FD8", "#9DB8EE"],
                     labels={"value": "", "variable": "", "Family": ""})
        lo = max(0.0, float(long["value"].min()) - 0.1)
        fig.update_layout(yaxis_range=[lo, 1], yaxis_tickformat=".0%", height=380,
                          margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")
    with r:
        st.dataframe(fam.style.format({"Best accuracy": "{:.2%}", "Mean accuracy": "{:.2%}",
                                       "Best F1 (macro)": "{:.3f}"}), width="stretch")
        top_fam = fam["Best F1 (macro)"].idxmax()
        st.success(f"Best family by F1: **{top_fam}** ({fam.loc[top_fam, 'Best model']}). "
                   f"Overall best model: **{best_name}** ({acc(best_name):.1%} accuracy).")

    # ---------------------------------------------------------------- bagging vs boosting
    st.subheader("2. Bagging vs Boosting — Random Forest vs Gradient Boosting")
    rf, gb = "Random Forest", "Gradient Boosting"
    cls_list = [c for c in CLASSES if c in set(y_te)]
    comp = pd.DataFrame({
        "Metric": ["Accuracy", "F1 (macro)"] + [f"Recall – {c}" for c in cls_list]
                  + ["Train time (s)"],
        rf: [acc(rf), f1(rf)] + [recall_of(rf, c) for c in cls_list] + [M[rf]["train_time"]],
        gb: [acc(gb), f1(gb)] + [recall_of(gb, c) for c in cls_list] + [M[gb]["train_time"]],
    }).set_index("Metric")
    l, r = st.columns([2, 3])
    with l:
        st.dataframe(comp.style.format("{:.3f}"), width="stretch")
    with r:
        enr = "High Achiever" if "High Achiever" in cls_list else cls_list[0]
        maj = "Qualified" if "Qualified" in cls_list else cls_list[-1]
        st.markdown(
            f"**What we see:** Random Forest {acc(rf):.1%} vs Gradient Boosting "
            f"{acc(gb):.1%} accuracy (macro-F1 {f1(rf):.3f} vs {f1(gb):.3f}). Recall on the "
            f"small *{enr}* class: RF {recall_of(rf, enr):.0%} vs GB {recall_of(gb, enr):.0%}; "
            f"on the big *{maj}* class: RF {recall_of(rf, maj):.0%} vs GB "
            f"{recall_of(gb, maj):.0%}.\n\n"
            "**Why:**\n"
            "- **Bagging (Random Forest)** trains many trees *in parallel* on bootstrap samples "
            "and averages them. Averaging **reduces variance**, which matters here because the "
            "pre-exam features are noisy. The average is pulled towards the majority class, so "
            f"RF is very good on *{maj}* but finds fewer *{enr}* students. Fully grown trees "
            "overfitted (≈52% accuracy); limiting leaf size (min 20 students) fixed this.\n"
            "- **Boosting (Gradient Boosting)** builds shallow trees *one after another*, each "
            "fitting the errors of the previous ones. This **reduces bias** and focuses on the "
            f"hard students, so it recovers more of the rare *{enr}* class, at the cost of "
            "more mistakes on the majority class and higher sensitivity to tuning.")

    # ---------------------------------------------------------------- stacking
    st.subheader("3. Stacking & Voting — combining different models")
    bases = ["Random Forest", "Extra Trees", "Gradient Boosting", "K-Nearest Neighbors"]
    ens = ["Stacking", "Voting (soft)"]
    stack_cmp = pd.DataFrame({"Model": ens + bases})
    stack_cmp["Accuracy"] = stack_cmp["Model"].map(acc)
    stack_cmp["F1 (macro)"] = stack_cmp["Model"].map(f1)
    stack_cmp["Type"] = stack_cmp["Model"].isin(ens).map(
        {True: "Ensemble of models", False: "Base model"})
    l, r = st.columns([3, 2])
    with l:
        fig = px.bar(stack_cmp, x="Model", y="F1 (macro)", color="Type",
                     text=stack_cmp["F1 (macro)"].map(lambda v: f"{v:.3f}"),
                     color_discrete_map={"Ensemble of models": "#29A36A",
                                         "Base model": "#9DB8EE"},
                     labels={"Type": "", "Model": ""}, title="Stacking vs its base models")
        fig.update_layout(yaxis_range=[max(0.0, stack_cmp["F1 (macro)"].min() - 0.1), None],
                          height=380, margin=dict(t=40))
        st.plotly_chart(fig, width="stretch")
    with r:
        names = ["Random Forest", "Gradient Boosting", "K-Nearest Neighbors",
                 "Logistic Regression"]
        dis = pd.DataFrame(
            [[float((M[a]["pred"] != M[b]["pred"]).mean()) for b in names] for a in names],
            index=names, columns=names)
        fig = px.imshow(dis, text_auto=".0%", color_continuous_scale="Greens",
                        title="How often two models disagree", aspect="auto")
        fig.update_layout(height=380, margin=dict(t=40), coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")
    best_base = max(bases, key=f1)
    gain = f1("Stacking") - f1(best_base)
    st.markdown(
        f"**What we see:** Stacking reaches **{acc('Stacking'):.1%}** accuracy and "
        f"**{f1('Stacking'):.3f}** macro-F1 ({gain:+.3f} F1 compared with its best base model, "
        f"{best_base}). Soft voting gets {acc('Voting (soft)'):.1%}.\n\n"
        "**Why:** the base models make *different* mistakes (see the disagreement matrix). "
        "**Soft voting** simply averages their probabilities. **Stacking** goes further: a "
        "Logistic-Regression *meta-learner* is trained on the base models' out-of-fold "
        "probabilities (5-fold CV), so it learns *which model to trust for which kind of "
        "student*. Gains are modest because the strongest base models are already tree "
        "ensembles that agree on most students — combining models helps most when their "
        "errors are uncorrelated.")

    # ---------------------------------------------------------------- hybrid
    st.subheader("4. Hybridization — Random Forest + ANN")
    hy = "Hybrid RF + ANN"
    hcmp = pd.DataFrame({"Model": ["Random Forest", "Neural Network (MLP)", hy]})
    hcmp["Accuracy"] = hcmp["Model"].map(acc)
    hcmp["F1 (macro)"] = hcmp["Model"].map(f1)
    l, r = st.columns([2, 3])
    with l:
        st.dataframe(hcmp.style.format({"Accuracy": "{:.2%}", "F1 (macro)": "{:.3f}"}),
                     width="stretch", hide_index=True)
    with r:
        vs_ann = acc(hy) - acc("Neural Network (MLP)")
        vs_rf = acc(hy) - acc("Random Forest")
        st.markdown(
            "**How it works:** the Random Forest's out-of-fold class probabilities are added "
            "as 3 extra inputs to a neural network, which is trained on the scaled original "
            "features + these \"forest opinions\".\n\n"
            f"**What we see:** the hybrid is {vs_ann * 100:+.1f} points vs the ANN alone and "
            f"{vs_rf * 100:+.1f} points vs the Random Forest alone. The forest's probabilities "
            "give the ANN a compact, non-linear summary of the subject combinations, so the "
            "ANN improves. It cannot go far beyond the forest because both models see the same "
            "9 pre-exam features — the remaining error is information that simply is not in "
            "the data (study effort, school, tuition, exam-day performance). The ANN also needs "
            "strong regularisation (L2 alpha = 1.0) so it does not over-trust the RF inputs.")

    # ---------------------------------------------------------------- behaviour
    st.subheader("5. Why the models behave this way")
    l, r = st.columns(2)
    with l:
        top = res.importance.head(8)[::-1]
        fig = px.bar(x=top.values, y=top.index, orientation="h",
                     title="What drives the prediction (RF importance)",
                     labels={"x": "Importance", "y": ""})
        fig.update_traces(marker_color="#3A6FD8")
        fig.update_layout(height=380)
        st.plotly_chart(fig, width="stretch")
    with r:
        fig = px.scatter(board, x="Train time (s)", y="F1 (macro)", color="Family",
                         hover_name="Model", log_x=True, title="Quality vs training cost")
        fig.update_traces(marker_size=12)
        fig.update_layout(height=380)
        st.plotly_chart(fig, width="stretch")
    share = df[TARGET].value_counts(normalize=True)
    top3 = ", ".join(res.importance.index[:3])
    st.markdown(
        f"- **What drives the result:** {top3}. The **CGT** (Common General Test) measures "
        "general reasoning and is taken in the same exam, so it is the best single proxy for "
        "overall ability; **General English** works the same way. **Gender** matters too: in "
        "this data female candidates fail a subject far less often than male candidates. "
        "**Age** acts non-linearly: 20–21-year-olds (typically a second attempt) fail less "
        "often than first-time 19-year-olds but rarely become High Achievers, while "
        "candidates over 23 do much worse. Its importance is also inflated because "
        "impurity-based importance favours continuous features. The **subject combination** "
        "captures how hard each stream is to pass.\n"
        "- **Why accuracy is only ~60%:** the models only see 9 pre-exam attributes — not "
        "marks, school, district, tuition or effort. A large part of the A/L result is "
        f"not in these features, so all models land close together (median accuracy "
        f"{board['Accuracy'].median():.0%}, vs {share.max():.0%} for always guessing the "
        "largest class). This is a realistic, noisy problem — exactly where ensembles help.\n"
        f"- **Class imbalance:** Qualified {share.get('Qualified', 0):.0%}, Not Qualified "
        f"{share.get('Not Qualified', 0):.0%}, High Achiever "
        f"{share.get('High Achiever', 0):.0%}. *High Achiever* is the smallest class and the "
        "hardest to find; macro-F1 is reported next to accuracy so it is not ignored.\n"
        "- **Single models have clear weak spots:** Naive Bayes assumes independent features "
        "(stream and subjects are strongly linked); KNN treats subject codes as distances; "
        "Logistic Regression can only draw linear boundaries over coded subjects; a single "
        "Decision Tree overfits. **Ensembles fix exactly these weaknesses** — bagging lowers "
        "variance, boosting lowers bias, stacking mixes different strengths.\n"
        "- **Cost vs gain:** Stacking and SVM are the most expensive to train; a regularised "
        "Random Forest gives top accuracy in about a second.")

    st.subheader("6. Conclusion")
    best_f1_name = board.sort_values("F1 (macro)", ascending=False).iloc[0]["Model"]
    st.info(
        f"**{best_name}** has the best accuracy ({acc(best_name):.1%}) and **{best_f1_name}** "
        f"the best macro-F1 ({f1(best_f1_name):.3f}) for predicting Sri Lankan A/L results "
        "from pre-exam information. Regularised bagging is the strongest on accuracy, boosting "
        "finds more of the rare *High Achiever* students, and stacking / the hybrid combine "
        "models that make different errors for a more balanced result. Every ensemble family "
        "beats the weakest single models (Naive Bayes, KNN, Logistic Regression).")
    st.caption("🤖 AI disclosure: code and dashboard generated with the help of Claude "
               "(Anthropic) and reviewed/tested by the student.")
