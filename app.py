import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Page configuration
st.set_page_config(page_title="Student Well-being Risk Prediction", layout="wide")
st.title("📊 Student Well-being Risk Prediction")
st.markdown("""
This dashboard demonstrates classification of students into **low** vs **high** well-being risk
using Support Vector Machines (SVM) with different kernels. The dataset is synthetic with two features:
- **Social Support Score** (0-10)
- **Stress Level** (0-10)

A non-linear rule generates the risk labels, making linear separation suboptimal.
""")

# ------------------------------
# Sidebar: Data Generation
# ------------------------------
st.sidebar.header("🔧 Data Parameters")

n_samples = st.sidebar.slider("Number of samples", 50, 500, 100, step=10)
noise_points = st.sidebar.slider("Number of noise points (label flips)", 0, 20, 6, step=1)
random_seed = st.sidebar.number_input("Random seed", value=42, step=1)

# ------------------------------
# Sidebar: SVM Hyperparameters
# ------------------------------
st.sidebar.header("⚙️ SVM Hyperparameters")

# Common C parameter
C_common = st.sidebar.number_input("Regularization parameter C (all kernels)", value=1.0, step=0.5)

# Linear SVM
st.sidebar.subheader("Linear SVM")
C_linear = st.sidebar.number_input("C (Linear)", value=C_common, step=0.5, key="C_linear")

# Polynomial SVM
st.sidebar.subheader("Polynomial SVM")
C_poly = st.sidebar.number_input("C (Poly)", value=C_common, step=0.5, key="C_poly")
degree = st.sidebar.slider("Degree", 2, 5, 3, step=1)
coef0 = st.sidebar.number_input("coef0", value=0.0, step=0.5)

# RBF SVM
st.sidebar.subheader("RBF SVM")
C_rbf = st.sidebar.number_input("C (RBF)", value=C_common, step=0.5, key="C_rbf")
gamma = st.sidebar.selectbox("gamma", ["scale", "auto"], index=0)

# ------------------------------
# Data Generation Function
# ------------------------------
@st.cache_data(show_spinner=False)
def generate_data(n, noise_pts, seed):
    np.random.seed(seed)
    social_support = np.random.uniform(0, 10, n)
    stress_level = np.random.uniform(0, 10, n)
    X = np.column_stack((social_support, stress_level))
    center_distance = (social_support - 5)**2 + (stress_level - 5)**2
    y = np.where(
        ((stress_level > 6) & (social_support < 6)) | (center_distance > 18),
        1, 0
    )
    if noise_pts > 0:
        noise_idx = np.random.choice(n, size=min(noise_pts, n), replace=False)
        y[noise_idx] = 1 - y[noise_idx]
    return X, y, social_support, stress_level

# ------------------------------
# Model Training Function (cached)
# ------------------------------
@st.cache_resource(show_spinner=False)
def train_models(X_train, y_train, C_lin, C_poly, degree, coef0, C_rbf, gamma):
    models = {}
    # Linear SVM
    models["Linear SVM"] = SVC(kernel="linear", C=C_lin)
    # Polynomial SVM
    models["Polynomial SVM"] = SVC(kernel="poly", degree=degree, C=C_poly, coef0=coef0)
    # RBF SVM
    models["RBF Kernel SVM"] = SVC(kernel="rbf", C=C_rbf, gamma=gamma)
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models

# ------------------------------
# Decision Boundary Plot
# ------------------------------
def plot_decision_boundary(model, X, y, title, ax):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.RdYlBu)
    ax.set_xlabel("Social Support Score")
    ax.set_ylabel("Stress Level")
    ax.set_title(title)
    return scatter

# ------------------------------
# Main App Logic
# ------------------------------
if st.sidebar.button("🚀 Train Models & Generate Data"):
    with st.spinner("Generating data and training models..."):
        # Generate data
        X, y, support, stress = generate_data(n_samples, noise_points, random_seed)
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=random_seed, stratify=y
        )
        # Train models
        models = train_models(X_train, y_train, C_linear, C_poly, degree, coef0, C_rbf, gamma)

        # Store in session state to persist across interactions
        st.session_state['X'] = X
        st.session_state['y'] = y
        st.session_state['X_train'] = X_train
        st.session_state['X_test'] = X_test
        st.session_state['y_train'] = y_train
        st.session_state['y_test'] = y_test
        st.session_state['models'] = models
        st.session_state['data_generated'] = True

# If data exists, display results
if st.session_state.get('data_generated', False):
    X = st.session_state['X']
    y = st.session_state['y']
    X_train = st.session_state['X_train']
    X_test = st.session_state['X_test']
    y_train = st.session_state['y_train']
    y_test = st.session_state['y_test']
    models = st.session_state['models']

    # Dataset summary
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📋 Dataset Overview")
        df = pd.DataFrame({
            "Social Support Score": X[:, 0],
            "Stress Level": X[:, 1],
            "Well-being Risk": y
        })
        st.dataframe(df.head(10))
    with col2:
        st.subheader("📊 Class Distribution")
        class_counts = pd.Series(y).value_counts().sort_index()
        st.bar_chart(class_counts)

    # Scatter plot of full dataset
    st.subheader("🔍 Data Visualization")
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.RdYlBu)
    ax.set_xlabel("Social Support Score")
    ax.set_ylabel("Stress Level")
    ax.set_title("Student Well-being Risk Dataset")
    st.pyplot(fig)

    # Performance evaluation
    st.subheader("📈 Model Performance Comparison")
    results = []
    for name, model in models.items():
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        results.append({
            "Model": name,
            "Training Accuracy": train_acc,
            "Test Accuracy": test_acc,
            "Training Error Rate": 1 - train_acc,
            "Test Error Rate": 1 - test_acc
        })
    results_df = pd.DataFrame(results)
    st.dataframe(results_df.style.format("{:.4f}"))

    # Highlight best model
    best_idx = results_df["Test Error Rate"].idxmin()
    best_model_name = results_df.loc[best_idx, "Model"]
    st.success(f"🏆 Best model on test data: **{best_model_name}** with test error rate {results_df.loc[best_idx, 'Test Error Rate']:.4f}")

    # Decision boundaries
    st.subheader("🧠 Decision Boundaries")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (name, model) in zip(axes, models.items()):
        plot_decision_boundary(model, X_train, y_train, name, ax)
    st.pyplot(fig)

    # Optional: Predict custom input
    st.subheader("🔮 Predict New Student")
    col1, col2 = st.columns(2)
    with col1:
        new_support = st.slider("Social Support Score", 0.0, 10.0, 5.0, step=0.1)
    with col2:
        new_stress = st.slider("Stress Level", 0.0, 10.0, 5.0, step=0.1)
    if st.button("Predict Risk"):
        new_X = np.array([[new_support, new_stress]])
        # Use best model for prediction
        best_model = models[best_model_name]
        pred = best_model.predict(new_X)[0]
        risk_label = "High Risk" if pred == 1 else "Low Risk"
        st.write(f"**Prediction:** {risk_label}")
        # Show on decision boundary plot (optional)
        st.info(f"The chosen model `{best_model_name}` predicts this student is **{risk_label}**.")
else:
    st.info("👈 Click **'Train Models & Generate Data'** in the sidebar to begin.")
