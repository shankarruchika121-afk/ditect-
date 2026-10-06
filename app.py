import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# Page config + styling
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="GlucoGuard | Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).parent
DATA_PATH = BASE / "data" / "diabetes.csv"
ART = BASE / "artifacts"

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
TEAL, RED, AMBER, GREEN = "#0ea5a4", "#ef4444", "#f59e0b", "#22c55e"

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem; max-width: 1200px;}
    .hero {
        background: linear-gradient(135deg, #0f766e 0%, #0ea5a4 55%, #38bdf8 100%);
        padding: 2rem 2.2rem; border-radius: 18px; color: white; margin-bottom: 1.4rem;
        box-shadow: 0 8px 24px rgba(14,165,164,.25);
    }
    .hero h1 {margin: 0; font-size: 2.2rem; color: white;}
    .hero p {margin: .4rem 0 0 0; font-size: 1.05rem; opacity: .92; color: white;}
    .card {
        border: 1px solid rgba(128,128,128,.25); border-radius: 14px;
        padding: 1.1rem 1.3rem; background: rgba(128,128,128,.06);
    }
    .risk-badge {
        display:inline-block; padding:.35rem 1rem; border-radius:999px;
        font-weight:700; color:white; font-size:1.05rem;
    }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.25); border-radius: 12px;
        padding: .7rem 1rem; background: rgba(128,128,128,.06);
    }
    .stButton>button, .stFormSubmitButton>button {
        border-radius: 10px; font-weight: 600;
    }
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Loading helpers (cached)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training models for the first time…")
def load_artifacts():
    """Load trained models. If they don't exist yet, train them automatically."""
    if not (ART / "models.joblib").exists():
        subprocess.run([sys.executable, str(BASE / "train_model.py")], check=True)
    models = joblib.load(ART / "models.joblib")
    with open(ART / "metrics.json") as f:
        metrics = json.load(f)
    return models, metrics


@st.cache_data
def load_data():
    raw = pd.read_csv(DATA_PATH)
    clean = raw.copy()
    clean[ZERO_AS_MISSING] = clean[ZERO_AS_MISSING].replace(0, np.nan)
    return raw, clean


models, metrics = load_artifacts()
raw_df, clean_df = load_data()
best_model = metrics["best_model"]

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🩺 GlucoGuard")
    page = st.radio(
        "Navigate",
        ["🔮 Predict", "📊 Data Explorer", "🏆 Model Performance", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Dataset")
    st.write(f"**{len(raw_df)}** patients · **{raw_df['Outcome'].sum()}** diabetic")
    st.caption("Best model (CV ROC-AUC)")
    st.write(f"**{best_model}**")
    st.divider()
    st.warning(
        "Educational project only. Not a medical diagnosis. "
        "Please consult a doctor for health decisions.",
        icon="⚠️",
    )

st.markdown(
    """
    <div class="hero">
        <h1>GlucoGuard · Diabetes Risk Predictor</h1>
        <p>Enter a few health measurements and let machine learning estimate diabetes risk instantly.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------------
def gauge(prob: float, color: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 44}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color, "thickness": 0.3},
            "steps": [
                {"range": [0, 35], "color": "rgba(34,197,94,.25)"},
                {"range": [35, 60], "color": "rgba(245,158,11,.25)"},
                {"range": [60, 100], "color": "rgba(239,68,68,.25)"},
            ],
        },
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=10),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def radar(patient: pd.Series):
    """Patient vs average non-diabetic vs average diabetic, min-max scaled."""
    mins, maxs = clean_df[FEATURES].min(), clean_df[FEATURES].max()
    scale = lambda s: ((s - mins) / (maxs - mins)).clip(0, 1)
    groups = {
        "Patient": scale(patient),
        "Avg non-diabetic": scale(clean_df[clean_df.Outcome == 0][FEATURES].mean()),
        "Avg diabetic": scale(clean_df[clean_df.Outcome == 1][FEATURES].mean()),
    }
    colors = {"Patient": TEAL, "Avg non-diabetic": GREEN, "Avg diabetic": RED}
    fig = go.Figure()
    for name, vals in groups.items():
        fig.add_trace(go.Scatterpolar(
            r=vals.tolist() + [vals.iloc[0]],
            theta=FEATURES + [FEATURES[0]],
            name=name,
            line=dict(color=colors[name], width=3 if name == "Patient" else 2,
                      dash="solid" if name == "Patient" else "dot"),
            fill="toself" if name == "Patient" else None,
            opacity=0.9,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], showticklabels=False)),
        height=420, margin=dict(l=40, r=40, t=30, b=30),
        legend=dict(orientation="h", y=-0.1),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ----------------------------------------------------------------------------
# PAGE: Predict
# ----------------------------------------------------------------------------
if page == "🔮 Predict":
    med = clean_df[FEATURES].median()

    left, right = st.columns([1, 1.15], gap="large")

    with left:
        st.subheader("Patient details")
        model_name = st.selectbox(
            "Model", list(models.keys()),
            index=list(models.keys()).index(best_model),
            help="Best model is selected by default.",
        )
        with st.form("patient_form"):
            c1, c2 = st.columns(2)
            preg = c1.number_input("Pregnancies", 0, 20, 1)
            age = c2.number_input("Age (years)", 18, 100, 33)
            glucose = st.slider("Glucose (mg/dL)", 40, 250, int(med.Glucose))
            bp = st.slider("Blood pressure – diastolic (mm Hg)", 30, 130, int(med.BloodPressure))
            bmi = st.slider("BMI", 15.0, 70.0, float(round(med.BMI, 1)), 0.1)
            dpf = st.slider("Diabetes pedigree function", 0.05, 2.5,
                            float(round(med.DiabetesPedigreeFunction, 2)), 0.01,
                            help="Score of diabetes family history. Higher = stronger family history.")
            c3, c4 = st.columns(2)
            skin = c3.number_input("Skin thickness (mm)", 0, 100, 0,
                                   help="Enter 0 if unknown. The model fills it in automatically.")
            insulin = c4.number_input("Insulin (µU/mL)", 0, 900, 0,
                                      help="Enter 0 if unknown. The model fills it in automatically.")
            submitted = st.form_submit_button("🔍 Predict risk", use_container_width=True,
                                              type="primary")

    patient = pd.DataFrame([{
        "Pregnancies": preg, "Glucose": glucose, "BloodPressure": bp,
        "SkinThickness": skin, "Insulin": insulin, "BMI": bmi,
        "DiabetesPedigreeFunction": dpf, "Age": age,
    }])
    # same rule as training: 0 means "missing" for these columns
    patient[ZERO_AS_MISSING] = patient[ZERO_AS_MISSING].replace(0, np.nan)

    with right:
        st.subheader("Result")
        if not submitted:
            st.info("Fill in the form and press **Predict risk** to see the result.", icon="👈")
        else:
            pipe = models[model_name]
            prob = float(pipe.predict_proba(patient)[0, 1])
            if prob < 0.35:
                label, color = "Low risk", GREEN
            elif prob < 0.60:
                label, color = "Moderate risk", AMBER
            else:
                label, color = "High risk", RED

            st.plotly_chart(gauge(prob, color), use_container_width=True)
            st.markdown(
                f"<div style='text-align:center'><span class='risk-badge' "
                f"style='background:{color}'>{label}</span></div>",
                unsafe_allow_html=True,
            )
            st.write("")

            # agreement across all models
            votes = {n: float(m.predict_proba(patient)[0, 1]) for n, m in models.items()}
            vote_df = pd.DataFrame({"Model": votes.keys(),
                                    "Probability": [v * 100 for v in votes.values()]})
            fig = px.bar(vote_df, x="Probability", y="Model", orientation="h",
                         range_x=[0, 100], text=vote_df["Probability"].round(1).astype(str) + "%")
            fig.update_traces(marker_color=TEAL, textposition="outside")
            fig.update_layout(height=230, margin=dict(l=0, r=30, t=10, b=0),
                              xaxis_title="Diabetes probability (%)", yaxis_title=None,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.caption("What each model thinks")
            st.plotly_chart(fig, use_container_width=True)

    if submitted:
        st.divider()
        a, b = st.columns([1.1, 1], gap="large")
        with a:
            st.subheader("How does this patient compare?")
            st.plotly_chart(radar(patient.iloc[0]), use_container_width=True)
        with b:
            st.subheader("Key readings")
            nd = clean_df[clean_df.Outcome == 0][FEATURES].mean()
            dm = clean_df[clean_df.Outcome == 1][FEATURES].mean()
            comp = pd.DataFrame({
                "Patient": patient.iloc[0],
                "Avg non-diabetic": nd,
                "Avg diabetic": dm,
            }).round(2)
            st.dataframe(comp, use_container_width=True, height=330)
            tips = []
            if glucose >= 126:
                tips.append("Glucose is in the diabetic range (≥126 mg/dL fasting).")
            elif glucose >= 100:
                tips.append("Glucose is in the pre-diabetic range (100–125 mg/dL).")
            if bmi >= 30:
                tips.append("BMI is in the obese range (≥30).")
            elif bmi >= 25:
                tips.append("BMI is in the overweight range (25–29.9).")
            if bp >= 90:
                tips.append("Diastolic blood pressure is high (≥90 mm Hg).")
            if tips:
                st.markdown("**Things worth noting**")
                for t in tips:
                    st.markdown(f"- {t}")
            else:
                st.success("Glucose, BMI and blood pressure are all in typical ranges.")

# ----------------------------------------------------------------------------
# PAGE: Data Explorer
# ----------------------------------------------------------------------------
elif page == "📊 Data Explorer":
    st.subheader("Explore the dataset")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients", len(raw_df))
    c2.metric("Diabetic", int(raw_df.Outcome.sum()))
    c3.metric("Non-diabetic", int((raw_df.Outcome == 0).sum()))
    c4.metric("Diabetic %", f"{raw_df.Outcome.mean() * 100:.1f}%")

    tab1, tab2, tab3, tab4 = st.tabs(["Distributions", "Correlations", "Relationships", "Raw data"])

    with tab1:
        feat = st.selectbox("Feature", FEATURES, index=1)
        plot_df = clean_df.copy()
        plot_df["Outcome"] = plot_df["Outcome"].map({0: "Non-diabetic", 1: "Diabetic"})
        fig = px.histogram(plot_df, x=feat, color="Outcome", barmode="overlay",
                           nbins=35, opacity=0.65, marginal="box",
                           color_discrete_map={"Non-diabetic": GREEN, "Diabetic": RED})
        fig.update_layout(height=450, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Zeros that really mean 'not measured' are excluded from these plots.")

    with tab2:
        corr = clean_df.corr(numeric_only=True).round(2)
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1, aspect="auto")
        fig.update_layout(height=520, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        x_col, y_col = st.columns(2)
        fx = x_col.selectbox("X axis", FEATURES, index=1)
        fy = y_col.selectbox("Y axis", FEATURES, index=5)
        plot_df = clean_df.copy()
        plot_df["Outcome"] = plot_df["Outcome"].map({0: "Non-diabetic", 1: "Diabetic"})
        fig = px.scatter(plot_df, x=fx, y=fy, color="Outcome", opacity=0.7,
                         color_discrete_map={"Non-diabetic": GREEN, "Diabetic": RED})
        fig.update_layout(height=480, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.dataframe(raw_df, use_container_width=True, height=420)
        st.download_button("⬇️ Download CSV", raw_df.to_csv(index=False).encode(),
                           "diabetes.csv", "text/csv")

# ----------------------------------------------------------------------------
# PAGE: Model Performance
# ----------------------------------------------------------------------------
elif page == "🏆 Model Performance":
    st.subheader("How well do the models perform?")
    st.caption(f"Evaluated on {metrics['n_test']} held-out patients "
               f"(trained on {metrics['n_train']}). CV ROC-AUC uses 5-fold cross-validation on all data.")

    rows = []
    for name, m in metrics["models"].items():
        rows.append({"Model": name, "Accuracy": m["accuracy"], "Precision": m["precision"],
                     "Recall": m["recall"], "F1": m["f1"], "ROC-AUC": m["roc_auc"],
                     "CV ROC-AUC": m["cv_roc_auc"]})
    perf = pd.DataFrame(rows).sort_values("CV ROC-AUC", ascending=False).reset_index(drop=True)
    st.dataframe(
        perf.style.format({c: "{:.3f}" for c in perf.columns if c != "Model"})
            .highlight_max(subset=[c for c in perf.columns if c != "Model"], color="rgba(14,165,164,.35)"),
        use_container_width=True, hide_index=True,
    )

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Metric comparison**")
        long = perf.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1"],
                         var_name="Metric", value_name="Score")
        fig = px.bar(long, x="Model", y="Score", color="Metric", barmode="group",
                     range_y=[0, 1])
        fig.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**ROC curves**")
        fig = go.Figure()
        for name, m in metrics["models"].items():
            fig.add_trace(go.Scatter(x=m["roc_curve"]["fpr"], y=m["roc_curve"]["tpr"],
                                     mode="lines", name=f"{name} (AUC {m['roc_auc']:.2f})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
                                 line=dict(dash="dash", color="gray")))
        fig.update_layout(height=400, xaxis_title="False positive rate",
                          yaxis_title="True positive rate", paper_bgcolor="rgba(0,0,0,0)",
                          legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2, gap="large")
    with c3:
        st.markdown("**Confusion matrix**")
        pick = st.selectbox("Model", list(models.keys()),
                            index=list(models.keys()).index(best_model), key="cm_pick")
        cm = np.array(metrics["models"][pick]["confusion_matrix"])
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Teal",
                        x=["Pred: No", "Pred: Yes"], y=["Actual: No", "Actual: Yes"])
        fig.update_layout(height=360, coloraxis_showscale=False, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        st.markdown("**What drives predictions? (Random Forest)**")
        imp = pd.Series(metrics["rf_importance"]).sort_values()
        fig = px.bar(imp, orientation="h", labels={"value": "Importance", "index": ""})
        fig.update_traces(marker_color=TEAL)
        fig.update_layout(height=360, showlegend=False, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# PAGE: About
# ----------------------------------------------------------------------------
else:
    st.subheader("About this project")
    st.markdown(
        """
        **GlucoGuard** predicts the likelihood of diabetes from 8 medical measurements using the
        Pima Indians Diabetes dataset (768 female patients, ≥21 years old).

        #### How it works
        1. **Cleaning** – zeros in Glucose, Blood Pressure, Skin Thickness, Insulin and BMI are
           biologically impossible, so they are treated as *missing* and filled with the median.
        2. **Scaling** – features are standardised so models like SVM and Logistic Regression behave well.
        3. **Training** – Logistic Regression, Decision Tree, Random Forest and SVM are trained in a
           single scikit-learn `Pipeline` (imputer → scaler → model).
        4. **Selection** – the best model is picked by 5-fold cross-validated ROC-AUC.

        #### Feature guide
        | Feature | Meaning |
        |---|---|
        | Pregnancies | Number of times pregnant |
        | Glucose | Plasma glucose concentration (2 h oral glucose tolerance test) |
        | BloodPressure | Diastolic blood pressure (mm Hg) |
        | SkinThickness | Triceps skin-fold thickness (mm) |
        | Insulin | 2-hour serum insulin (µU/mL) |
        | BMI | Body mass index |
        | DiabetesPedigreeFunction | Likelihood of diabetes based on family history |
        | Age | Age in years |

        #### Limitations
        The dataset is small and drawn from one population, so predictions may not generalise.
        This tool is for **learning and demonstration only**, not clinical use.
        """
    )
