import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Well-being Risk Prediction",
    page_icon="🎓",
    layout="wide",
)

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🎓 Student Well-being Risk Prediction")
st.markdown(
    "Compare **Linear**, **Polynomial**, and **RBF** kernel SVMs on a synthetic "
    "student well-being dataset with two features: *Social Support Score* and *Stress Level*."
)
st.divider()

# ── Sidebar – dataset & model controls ───────────────────────────────────────
st.sidebar.header("⚙️ Configuration")

n_samples = st.sidebar.slider("Number of students (n)", 50, 500, 100, step=10)
random_seed = st.sidebar.number_input("Random seed", value=42, step=1)
test_size = st.sidebar.slider("Test split ratio", 0.10, 0.40, 0.30, step=0.05)

st.sidebar.subheader("Linear SVM")
C_linear = st.sidebar.select_slider("C (Linear)", options=[0.01, 0.1, 1, 10, 100], value=1)

st.sidebar.subheader("Polynomial SVM")
C_poly = st.sidebar.select_slider("C (Poly)", options=[0.01, 0.1, 1, 10, 100], value=10)
degree_poly = st.sidebar.slider("Degree", 2, 6, 3)
coef0_poly = st.sidebar.slider("coef0", 0, 5, 1)

st.sidebar.subheader("RBF SVM")
C_rbf = st.sidebar.select_slider("C (RBF)", options=[0.01, 0.1, 1, 10, 100], value=10)
gamma_rbf = st.sidebar.selectbox("Gamma", ["scale", "auto"], index=0)

# ── Data generation ───────────────────────────────────────────────────────────
@st.cache_data
def generate_data(n, seed):
    rng = np.random.default_rng(seed)
    social_support = rng.uniform(0, 10, n)
    stress = rng.uniform(0, 10, n)
    X = np.column_stack((social_support, stress))
    center_dist = (social_support - 5) ** 2 + (stress - 5) ** 2
    y = np.where(
        ((stress > 6) & (social_support < 6)) | (center_dist > 18), 1, 0
    )
    noise_idx = np.random.choice(n, size=6, replace=False)
    y[noise_idx] = 1 - y[noise_idx]
    df = pd.DataFrame(
        {"Social Support Score": social_support, "Stress Level": stress, "Well-being Risk": y}
    )
    return X, y, df


@st.cache_data
def split_data(X, y, test_size, seed):
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


X, y, df = generate_data(n_samples, int(random_seed))
X_train, X_test, y_train, y_test = split_data(X, y, test_size, int(random_seed))

# ── Train models ──────────────────────────────────────────────────────────────
@st.cache_data
def train_models(X_train, y_train, C_linear, C_poly, degree_poly, coef0_poly, C_rbf, gamma_rbf):
    linear = SVC(kernel="linear", C=C_linear)
    poly   = SVC(kernel="poly",   degree=degree_poly, C=C_poly, coef0=coef0_poly)
    rbf    = SVC(kernel="rbf",    C=C_rbf, gamma=gamma_rbf)
    linear.fit(X_train, y_train)
    poly.fit(X_train, y_train)
    rbf.fit(X_train, y_train)
    return linear, poly, rbf


linear_svm, poly_svm, rbf_svm = train_models(
    X_train, y_train, C_linear, C_poly, degree_poly, coef0_poly, C_rbf, gamma_rbf
)

models = {
    "Linear SVM": linear_svm,
    "Polynomial SVM": poly_svm,
    "RBF SVM": rbf_svm,
}

# ── Results table ─────────────────────────────────────────────────────────────
results = []
for name, model in models.items():
    tr_acc = accuracy_score(y_train, model.predict(X_train))
    te_acc = accuracy_score(y_test,  model.predict(X_test))
    results.append({
        "Model": name,
        "Train Accuracy": tr_acc,
        "Test Accuracy": te_acc,
        "Train Error Rate": 1 - tr_acc,
        "Test Error Rate": 1 - te_acc,
    })
results_df = pd.DataFrame(results)
best_model_name = results_df.loc[results_df["Test Error Rate"].idxmin(), "Model"]

# ── KPI row ───────────────────────────────────────────────────────────────────
st.subheader("📊 Dataset Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Students", n_samples)
col2.metric("Training Set", len(X_train))
col3.metric("Test Set", len(X_test))
col4.metric("High-Risk Students", int(y.sum()))

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📁 Dataset", "📈 Model Performance", "🗺️ Decision Boundaries", "🔍 Predict a Student"]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 – Dataset
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Raw Data (first 10 rows)")
        display_df = df.copy()
        display_df["Well-being Risk"] = display_df["Well-being Risk"].map(
            {0: "🟢 Low Risk", 1: "🔴 High Risk"}
        )
        st.dataframe(display_df.head(10), use_container_width=True)

        st.subheader("Class Distribution")
        counts = df["Well-being Risk"].value_counts().rename({0: "Low Risk", 1: "High Risk"})
        st.bar_chart(counts)

    with col_right:
        st.subheader("Full Dataset Scatter Plot")
        fig, ax = plt.subplots(figsize=(6, 5))
        colors = ["#4C9BE8", "#E85D5D"]
        for risk_val, label, color in zip([0, 1], ["Low Risk", "High Risk"], colors):
            mask = df["Well-being Risk"] == risk_val
            ax.scatter(
                df.loc[mask, "Social Support Score"],
                df.loc[mask, "Stress Level"],
                c=color, edgecolors="k", linewidths=0.5, label=label, alpha=0.85, s=60,
            )
        ax.set_xlabel("Social Support Score")
        ax.set_ylabel("Stress Level")
        ax.set_title("Student Well-being Risk Dataset")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Train / Test Split")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    marker_map = [(X_train, y_train, "o", "Training"), (X_test, y_test, "s", "Test")]
    color_map = {0: "#4C9BE8", 1: "#E85D5D"}
    for X_part, y_part, marker, split in marker_map:
        for risk_val, label_prefix in [(0, "Low Risk"), (1, "High Risk")]:
            mask = y_part == risk_val
            ax2.scatter(
                X_part[mask, 0], X_part[mask, 1],
                c=color_map[risk_val], marker=marker, edgecolors="k",
                linewidths=0.5, alpha=0.85, s=65,
                label=f"{split} – {label_prefix}",
            )
    ax2.set_xlabel("Social Support Score")
    ax2.set_ylabel("Stress Level")
    ax2.set_title("Training (●) and Test (■) Sets")
    ax2.legend(ncol=2, fontsize=8)
    st.pyplot(fig2)
    plt.close(fig2)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 – Model Performance
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Model Accuracy & Error Rates")

    # Highlight best model
    def highlight_best(row):
        return [
            "background-color: #d4edda; font-weight: bold" if row["Model"] == best_model_name else ""
        ] * len(row)

    styled = (
        results_df.style
        .apply(highlight_best, axis=1)
        .format({
            "Train Accuracy": "{:.1%}",
            "Test Accuracy": "{:.1%}",
            "Train Error Rate": "{:.1%}",
            "Test Error Rate": "{:.1%}",
        })
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.caption(f"✅ Best model (lowest test error): **{best_model_name}**")

    # Error rate bar chart
    st.subheader("Error Rate Comparison")
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    x = np.arange(len(results_df))
    width = 0.35
    bars1 = ax3.bar(x - width / 2, results_df["Train Error Rate"], width, label="Train Error Rate", color="#5B8DB8")
    bars2 = ax3.bar(x + width / 2, results_df["Test Error Rate"],  width, label="Test Error Rate",  color="#E07B54")
    ax3.set_xticks(x)
    ax3.set_xticklabels(results_df["Model"])
    ax3.set_ylabel("Error Rate")
    ax3.set_title("Training vs Test Error Rates")
    ax3.legend()
    ax3.set_ylim(0, max(results_df[["Train Error Rate", "Test Error Rate"]].values.max() * 1.3, 0.1))
    for bar in bars1:
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{bar.get_height():.1%}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{bar.get_height():.1%}", ha="center", va="bottom", fontsize=9)
    st.pyplot(fig3)
    plt.close(fig3)

    # Confusion matrices
    st.subheader("Confusion Matrices (Test Set)")
    cm_cols = st.columns(3)
    for i, (name, model) in enumerate(models.items()):
        with cm_cols[i]:
            st.markdown(f"**{name}**")
            cm = confusion_matrix(y_test, model.predict(X_test))
            fig_cm, ax_cm = plt.subplots(figsize=(3.5, 3))
            disp = ConfusionMatrixDisplay(cm, display_labels=["Low Risk", "High Risk"])
            disp.plot(ax=ax_cm, colorbar=False, cmap="Blues")
            ax_cm.set_title(name, fontsize=9)
            st.pyplot(fig_cm)
            plt.close(fig_cm)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 – Decision Boundaries
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Decision Boundary Plots")
    data_source = st.radio("Show data points from:", ["Training Set", "Test Set", "All Data"], horizontal=True)

    if data_source == "Training Set":
        X_plot, y_plot = X_train, y_train
    elif data_source == "Test Set":
        X_plot, y_plot = X_test, y_test
    else:
        X_plot, y_plot = X, y

    def plot_boundary(model, X_pts, y_pts, title):
        x_min, x_max = X_pts[:, 0].min() - 1, X_pts[:, 0].max() + 1
        y_min, y_max = X_pts[:, 1].min() - 1, X_pts[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.contourf(xx, yy, Z, alpha=0.25, cmap=mcolors.ListedColormap(["#AED6F1", "#F1948A"]))
        ax.contour(xx, yy, Z, colors="k", linewidths=0.8)
        colors = ["#2471A3", "#C0392B"]
        for rv, label, color in zip([0, 1], ["Low Risk", "High Risk"], colors):
            mask = y_pts == rv
            ax.scatter(X_pts[mask, 0], X_pts[mask, 1], c=color, edgecolors="k",
                       linewidths=0.4, s=55, label=label, alpha=0.9)
        ax.set_xlabel("Social Support Score", fontsize=9)
        ax.set_ylabel("Stress Level", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        return fig

    bd_cols = st.columns(3)
    for col, (name, model) in zip(bd_cols, models.items()):
        with col:
            fig_bd = plot_boundary(model, X_plot, y_plot, name)
            st.pyplot(fig_bd)
            plt.close(fig_bd)

    st.info(
        "💡 The **Linear SVM** uses a straight boundary. "
        "**Polynomial** and **RBF** kernels create curved, more flexible boundaries "
        "better suited to this non-linear dataset."
    )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 – Predict a Student
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("🔍 Predict Well-being Risk for a New Student")
    st.markdown("Adjust the sliders below and all three models will predict the student's risk level.")

    p_col1, p_col2 = st.columns(2)
    with p_col1:
        input_support = st.slider("Social Support Score", 0.0, 10.0, 5.0, 0.1)
    with p_col2:
        input_stress = st.slider("Stress Level", 0.0, 10.0, 5.0, 0.1)

    input_point = np.array([[input_support, input_stress]])

    pred_cols = st.columns(3)
    for col, (name, model) in zip(pred_cols, models.items()):
        pred = model.predict(input_point)[0]
        risk_label = "🔴 High Risk" if pred == 1 else "🟢 Low Risk"
        with col:
            st.metric(label=name, value=risk_label)

    # Show input on scatter
    fig_pred, ax_pred = plt.subplots(figsize=(6, 4))
    for rv, label, color in zip([0, 1], ["Low Risk", "High Risk"], ["#4C9BE8", "#E85D5D"]):
        mask = y == rv
        ax_pred.scatter(X[mask, 0], X[mask, 1], c=color, edgecolors="k",
                        linewidths=0.4, s=40, label=label, alpha=0.6)
    ax_pred.scatter(input_support, input_stress, c="yellow", edgecolors="black",
                    s=200, marker="*", zorder=5, label="New Student")
    ax_pred.set_xlabel("Social Support Score")
    ax_pred.set_ylabel("Stress Level")
    ax_pred.set_title("New Student in Dataset Context")
    ax_pred.legend(fontsize=8)
    st.pyplot(fig_pred)
    plt.close(fig_pred)
