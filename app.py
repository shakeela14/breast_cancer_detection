# app.py
import streamlit as st
from src.data import load_csv_dataset, prepare_data
from src.model import train_default_model, save_model, evaluate_model
from src.ui import set_page_theme, css_style, confidence_gauge
from utils.pdf_report import create_prediction_pdf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib
from pathlib import Path
import plotly.express as px
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# -----------------------------
# 🌈 GRADIENT UI CSS
# -----------------------------
st.markdown(
    """
    <style>
    /* Background gradient */
    .stApp {
        background: linear-gradient(135deg, #ccffcc, #99ffcc, #ccffe6);
        background-attachment: fixed;
        font-family: 'Poppins', sans-serif;
        font-style: italic;
        color: #003300;
        transition: all 0.4s ease-in-out;
    }

    /* Headers and titles */
    h1, h2, h3, h4 {
        color: #004d26 !important;
        text-shadow: 1px 1px 2px #99ffcc;
        font-style: italic;
    }

    /* Buttons */
    div.stButton > button {
        background: linear-gradient(90deg, #004d00, #00b36b);
        color: white;
        border-radius: 12px;
        padding: 10px 25px;
        border: none;
        font-weight: bold;
        font-style: italic;
        box-shadow: 0px 4px 10px rgba(0, 102, 51, 0.4);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    div.stButton > button:hover {
        transform: scale(1.07);
        box-shadow: 0px 6px 15px rgba(0, 153, 76, 0.6);
    }

    /* Select boxes */
    div[data-baseweb="select"] > div {
        background: linear-gradient(90deg, #006633, #00b36b, #33ff99) !important;
        color: white !important;
        border-radius: 10px !important;
        border: 2px solid #00b36b !important;
        font-weight: 600;
        box-shadow: 0px 2px 10px rgba(0, 102, 51, 0.3);
        animation: selectGlow 4s infinite alternate;
    }

    @keyframes selectGlow {
        from { box-shadow: 0px 0px 5px #00b36b; }
        to { box-shadow: 0px 0px 15px #00ff99; }
    }

    /* Tabs */
    div[data-baseweb="tab-list"] button {
        background: linear-gradient(90deg, #004d00, #009966) !important;
        color: white !important;
        border-radius: 10px !important;
        margin-right: 10px !important;
        font-style: italic;
        font-weight: 500;
        transition: all 0.3s ease-in-out;
    }
    div[data-baseweb="tab-list"] button[aria-selected="true"] {
        background: linear-gradient(90deg, #009966, #00cc99) !important;
        border-bottom: 3px solid #33ff99 !important;
        transform: scale(1.05);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ccffcc, #99ffcc, #e6fff2);
        color: #004d00 !important;
        font-style: italic;
        border-right: 3px solid #00b36b;
    }

    /* Tables */
    .stDataFrame {
        border: 2px solid #00b36b !important;
        border-radius: 12px !important;
        box-shadow: 0px 2px 10px rgba(0, 153, 76, 0.3);
    }

    /* Metrics */
    div[data-testid="stMetricValue"] {
        color: #004d00 !important;
        font-weight: bold;
    }

    /* Header Container with Gradient */
    .header-container {
        background: linear-gradient(90deg, #00cc66, #00994d, #00e699);
        border-radius: 20px;
        padding: 25px 30px;
        text-align: center;
        box-shadow: 0px 4px 15px rgba(0, 102, 51, 0.4);
        animation: gradientShift 6s ease infinite;
        background-size: 300% 300%;
        color: white;
        margin-bottom: 20px;
        transition: all 0.5s ease;
    }

    .header-container h1 {
        color: #ffffff;
        font-style: italic;
        font-weight: 700;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
    }

    .header-container .subtext {
        font-size: 16px;
        color: #e6ffe6;
        margin-top: -8px;
        font-style: italic;
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Download button */
    div.stDownloadButton > button {
        background: linear-gradient(90deg, #008000, #00b36b, #33ff99);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 10px 25px;
        font-weight: bold;
        font-style: italic;
        box-shadow: 0px 4px 12px rgba(0, 128, 0, 0.4);
        transition: all 0.3s ease-in-out;
        animation: pulseGreen 3s infinite alternate;
    }

    div.stDownloadButton > button:hover {
        transform: scale(1.08);
        box-shadow: 0px 8px 20px rgba(0, 204, 102, 0.6);
        background: linear-gradient(90deg, #00b36b, #00e699);
    }

    @keyframes pulseGreen {
        0% { box-shadow: 0px 0px 5px #00cc66; }
        100% { box-shadow: 0px 0px 20px #00ff99; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Page Theme
# -----------------------------
set_page_theme()
css_style(theme="light")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_and_prepare():
    df = load_csv_dataset()
    X_train, X_test, y_train, y_test, scaler, columns = prepare_data(df)
    return df, X_train, X_test, y_train, y_test, scaler, columns

df, X_train, X_test, y_train, y_test, scaler, columns = load_and_prepare()

# -----------------------------
# Sidebar — Phase Switcher
# -----------------------------
st.sidebar.title(" Project Phases")
phase = st.sidebar.radio("Select a Phase", (
    "Phase 1: ML Foundation",
    "Phase 2: UX & Visualization",
    "Phase 3: Interactivity",
    "Phase 4: Explainability",
))

# -----------------------------
# Train Model
# -----------------------------
@st.cache_data
def train_and_return():
    model = train_default_model(X_train, y_train)
    acc, cm = evaluate_model(model, X_test, y_test)
    save_model(model, scaler, name="logistic_v1")
    return model, acc, cm

with st.spinner("Training the logistic regression model..."):
    model, accuracy, confusion = train_and_return()

# -----------------------------
# Header (Container with Gradient)
# -----------------------------
st.markdown("""
<div class="header-container">
    <h1>Breast Cancer Detection</h1>
    <p class="subtext">Leveraging machine learning to empower early diagnosis, enhance awareness, and support better healthcare outcomes</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# PHASE 1 — ML Foundation
# -----------------------------
if phase.startswith("Phase 1"):
    st.header(" Phase 1 — Machine Learning Foundation")
    st.metric("Test Accuracy", f"{accuracy*100:.2f}%")

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    ax.imshow(confusion, cmap="Greens")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.write(" Sample of Dataset")
    st.dataframe(df.head())

# -----------------------------
# PHASE 2 — Visualization
# -----------------------------
elif phase.startswith("Phase 2"):
    st.header(" Phase 2 — UX & Visualization")
    importance = np.abs(model.coef_[0])
    imp_df = pd.DataFrame({"Feature": columns, "Importance": importance}).sort_values("Importance", ascending=False)
    st.subheader("🌟 Top Features")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(imp_df['Feature'][:10], imp_df['Importance'][:10], color="#00b36b")
    ax.invert_yaxis()
    st.pyplot(fig)

    st.subheader(" Download Demo Report")
    sample_inputs = {c: float(df[c].mean()) for c in columns[:10]}
    buf = create_prediction_pdf("Benign (Demo)", 0.87, accuracy, sample_inputs)
    st.download_button(" Download PDF", data=buf, file_name="demo_report.pdf", mime="application/pdf")

# -----------------------------
# PHASE 3 — Interactivity
# -----------------------------
elif phase.startswith("Phase 3"):
    st.header(" Phase 3 — Interactive Predictions")
    tabs = st.tabs(["Single Prediction", "Bulk Upload", "History"])

    if "history" not in st.session_state:
        st.session_state.history = []

    with tabs[0]:
        st.subheader("Single Prediction")
        user_vals = {feat: st.number_input(f"{feat}", float(df[feat].min()), float(df[feat].max()), float(df[feat].mean())) for feat in columns[:10]}
        if st.button("🔍 Predict"):
            x = np.array([list(user_vals.values()) + [0]*(len(columns)-10)])
            x_scaled = scaler.transform(x)
            pred = model.predict(x_scaled)[0]
            proba = model.predict_proba(x_scaled)[0].max()
            label = "Benign" if pred == 1 else "Malignant"
            st.success(f"*Prediction: {label} — Confidence: {proba:.2%}*")
            st.pyplot(confidence_gauge(proba))
            st.session_state.history.append({"Label": label, "Confidence": proba})

    with tabs[1]:
        st.subheader("Bulk CSV Upload")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            user_df = pd.read_csv(uploaded)

            # 🔹 Drop unwanted columns
            user_df = user_df.drop(columns=["id", "diagnosis", "Unnamed: 32"], errors="ignore")

            # 🔹 Keep only columns used for training
            user_df = user_df[[c for c in columns if c in user_df.columns]]

            # 🔹 Ensure same column order
            user_df = user_df.reindex(columns=columns, fill_value=0)

            # 🔹 Scale and predict
            scaled = scaler.transform(user_df)
            preds = model.predict(scaled)
            user_df["Prediction"] = ["Benign" if p == 1 else "Malignant" for p in preds]

            st.dataframe(user_df.head())
            csv_bytes = user_df.to_csv(index=False).encode("utf-8")
            st.download_button(" Download Results", data=csv_bytes, file_name="bulk_predictions.csv")

    with tabs[2]:
        st.subheader("Prediction History")
        st.dataframe(pd.DataFrame(st.session_state.history))

# -----------------------------
# PHASE 4 — Explainability
# -----------------------------
elif phase.startswith("Phase 4"):
    st.header(" Phase 4 — Model Explainability")
    model_choice = st.selectbox("Choose a Model", ["Logistic Regression", "Random Forest", "KNN"])
    if st.button(" Train Model"):
        if model_choice == "Logistic Regression":
            eval_model = LogisticRegression(max_iter=300)
        elif model_choice == "Random Forest":
            eval_model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            eval_model = KNeighborsClassifier(n_neighbors=5)

        eval_model.fit(X_train, y_train)
        y_pred = eval_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.metric("Accuracy", f"{acc*100:.2f}%")

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap="Greens", ax=ax)
        st.pyplot(fig)

        if hasattr(eval_model, "feature_importances_"):
            imp_df = pd.DataFrame({"Feature": columns, "Importance": eval_model.feature_importances_})
            fig = px.bar(imp_df.sort_values("Importance"), x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="Greens")
            st.plotly_chart(fig)

# -----------------------------
# PHASE 5 — Deployment
# -----------------------------
elif phase.startswith("Phase 5"):
    st.header("🚀 Phase 5 — Deployment & API")
    st.write("_FastAPI integration available in src/api.py_")
    st.code("uvicorn src.api:app --reload --port 8000", language="bash")

st.markdown("---")
st.caption("*✨ Built with ❤️ by Shakeela Shaik —  Streamlit ML App ✨*")
