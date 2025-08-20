# =========================
# ML Lab – Interactive All-in-One 
# =========================
# Run: streamlit run ml_lab.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    mean_squared_error, r2_score, roc_curve, auc
)
from sklearn.datasets import load_iris, make_blobs
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# ---------- Page ----------
st.set_page_config(page_title="🤖 Interactive ML Lab", layout="centered")
st.title("🤖 Interactive ML Lab")
st.caption("KNN • Linear Regression • Logistic Regression • K-Means • Naive Bayes (Text)")

# ---------- Helpers ----------
def small_confmat(cm, labels):
    fig, ax = plt.subplots(figsize=(3.2, 3.0))  # small box so it doesn't look huge
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0)
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="red", fontsize=11)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    return fig

def clf_report_df(y_true, y_pred, target_names=None):
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0, target_names=target_names)
    return pd.DataFrame(rep).transpose()

# ---------- Sidebar ----------
algo = st.sidebar.radio(
    "Choose Algorithm",
    ["KNN Classification", "Linear Regression", "Logistic Regression", "K-Means Clustering", "Naive Bayes (Text)"]
)

# ============================================================
# 1) KNN Classification (Iris)
# ============================================================
if algo == "KNN Classification":
    st.header("🔹 KNN Classification (Iris)")
    iris = load_iris(as_frame=True)
    X, y = iris.data.values, iris.target.values
    names = iris.target_names

    k = st.sidebar.slider("K (neighbors)", 1, 25, 5, 1)
    test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2, 0.05)
    scale = st.sidebar.checkbox("Standardize features", True)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    if scale:
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)

    knn = KNeighborsClassifier(n_neighbors=k).fit(Xtr, ytr)
    yp = knn.predict(Xte)
    acc = accuracy_score(yte, yp)

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Accuracy", f"{acc*100:.2f}%")
        st.write("**Params**: ", f"K={k}, Standardize={scale}, Test={test_size}")
    with c2:
        st.write("**Confusion Matrix**")
        st.pyplot(small_confmat(confusion_matrix(yte, yp, labels=np.unique(y)), names), use_container_width=True)

    st.write("**Classification Report**")
    st.dataframe(clf_report_df(yte, yp, target_names=names))

# ============================================================
# 2) Linear Regression (Synthetic Hours → Score)
# ============================================================
elif algo == "Linear Regression":
    st.header("🔹 Linear Regression (Hours ➜ Score)")
    n = st.sidebar.slider("Samples", 30, 500, 120, 10)
    slope = st.sidebar.slider("Slope (m)", 0.0, 15.0, 6.0, 0.5)
    intercept = st.sidebar.slider("Intercept (c)", -10.0, 20.0, 5.0, 0.5)
    noise = st.sidebar.slider("Noise (std)", 0.0, 10.0, 3.0, 0.5)
    rng = np.random.default_rng(7)
    hours = rng.uniform(1, 10, n)
    score = intercept + slope * hours + rng.normal(0, noise, n)
    df = pd.DataFrame({"Hours Studied": hours, "Score": score})

    X = df[["Hours Studied"]].values
    y = df["Score"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression().fit(Xtr, ytr)
    yp = model.predict(Xte)
    mse = mean_squared_error(yte, yp); r2 = r2_score(yte, yp)

    st.metric("R²", f"{r2:.3f}")
    st.metric("MSE", f"{mse:.3f}")

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.scatter(X, y, s=18, label="data")
    xs = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    ax.plot(xs, model.predict(xs), label="best fit", linewidth=2)
    ax.set_xlabel("Hours Studied")
    ax.set_ylabel("Score")
    ax.legend()
    st.pyplot(fig, use_container_width=True)

# ============================================================
# 3) Logistic Regression (Hours ➜ Pass/Fail)
# ============================================================
elif algo == "Logistic Regression":
    st.header("🔹 Logistic Regression (Pass/Fail)")
    n = st.sidebar.slider("Samples", 50, 600, 200, 10)
    pass_thr = st.sidebar.slider("Pass threshold (hours)", 2, 9, 5, 1)
    noise_flip = st.sidebar.slider("Noise flip probability", 0.0, 0.5, 0.15, 0.05)

    rng = np.random.default_rng(42)
    hours = rng.integers(1, 11, size=n)
    y = (hours >= pass_thr).astype(int)
    # flip some labels to simulate noise
    flips = rng.random(n) < noise_flip
    y = np.where(flips, 1 - y, y)

    df = pd.DataFrame({"Hours Studied": hours, "Pass (1) / Fail (0)": y})
    st.subheader("Sample of the data")
    st.dataframe(df.head())

    X = hours.reshape(-1, 1)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

    logit = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    prob = logit.predict_proba(Xte)[:, 1]
    yp = (prob >= 0.5).astype(int)
    acc = accuracy_score(yte, yp)

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Accuracy", f"{acc*100:.2f}%")
        st.write("**Confusion Matrix**")
        st.pyplot(small_confmat(confusion_matrix(yte, yp), ["Fail(0)", "Pass(1)"]), use_container_width=True)

    with c2:
        fpr, tpr, _ = roc_curve(yte, prob)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(3.6, 3.0))
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], "--")
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.legend()
        st.pyplot(fig, use_container_width=True)

    # Show data with predicted probability curve
    st.subheader("Decision curve")
    xs = np.arange(1, 11).reshape(-1, 1)
    p_line = logit.predict_proba(xs)[:, 1]
    fig2, ax2 = plt.subplots(figsize=(6, 3.2))
    jitter = (np.random.rand(len(hours)) - 0.5) * 0.04
    ax2.scatter(hours, y + jitter, s=16, alpha=0.6, label="data (Pass=1/Fail=0)")
    ax2.plot(xs, p_line, linewidth=2, label="P(Pass | Hours)")
    ax2.axhline(0.5, ls="--", lw=1)
    ax2.set_xlabel("Hours Studied")
    ax2.set_ylabel("Pass Probability / Label")
    ax2.set_ylim(-0.1, 1.1)
    ax2.legend()
    st.pyplot(fig2, use_container_width=True)

# ============================================================
# 4) K-Means Clustering (with Elbow)
# ============================================================
elif algo == "K-Means Clustering":
    st.header("🔹 K-Means Clustering (with Elbow)")
    n = st.sidebar.slider("Samples", 100, 2000, 500, 50)
    true_centers = st.sidebar.slider("True centers (for generation)", 2, 8, 4, 1)
    spread = st.sidebar.slider("Cluster spread (std)", 0.2, 2.0, 0.7, 0.1)
    rng_seed = st.sidebar.number_input("Random seed", 0, 9999, 7, 1)

    X, _ = make_blobs(n_samples=n, centers=true_centers, cluster_std=spread, random_state=rng_seed)

    # Elbow (WCSS) to choose K
    st.subheader("Elbow Method – choose K")
    k_min, k_max = 1, 10
    Ks = list(range(k_min, k_max + 1))
    wcss = []
    for k in Ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(X)
        wcss.append(km.inertia_)
    fig_elbow, ax_e = plt.subplots(figsize=(5.2, 3.2))
    ax_e.plot(Ks, wcss, marker="o")
    ax_e.set_xlabel("K")
    ax_e.set_ylabel("WCSS (Inertia)")
    ax_e.set_title("Elbow Curve")
    st.pyplot(fig_elbow, use_container_width=True)

    k_chosen = st.sidebar.slider("Choose K for clustering", 2, 10, min(4, k_max), 1)
    km = KMeans(n_clusters=k_chosen, n_init=10, random_state=42).fit(X)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.scatter(X[:, 0], X[:, 1], c=km.labels_, s=22, cmap="viridis")
    ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], c="red", s=180, marker="X", label="centers")
    ax.set_title(f"K-Means Result (K={k_chosen})")
    ax.legend()
    st.pyplot(fig, use_container_width=True)

# ============================================================
# 5) Naive Bayes (Text) – Spam/Ham
# ============================================================
else:
    st.header("🔹 Naive Bayes – Text Spam Filter (Toy)")
    # A slightly larger toy dataset (balanced)
    corpus = [
        "win a free lottery now", "limited time offer discount", "urgent prize claim now",
        "exclusive deal for you", "free money guaranteed", "click to claim reward",
        "meeting at 10am", "can we reschedule our call", "project deadline update",
        "family dinner tonight", "see you at office", "lets discuss the report"
    ]
    labels = np.array([1,1,1,1,1,1, 0,0,0,0,0,0])  # 1=spam, 0=ham

    vect = CountVectorizer()
    X = vect.fit_transform(corpus)

    Xtr, Xte, ytr, yte = train_test_split(X, labels, test_size=0.33, random_state=42, stratify=labels)
    nb = MultinomialNB().fit(Xtr, ytr)
    yp = nb.predict(Xte)
    acc = accuracy_score(yte, yp)

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Test Accuracy", f"{acc*100:.2f}%")
        st.write("**Confusion Matrix**")
        st.pyplot(small_confmat(confusion_matrix(yte, yp), ["ham", "spam"]), use_container_width=True)
    with c2:
        st.write("**Classification Report**")
        st.dataframe(clf_report_df(yte, yp, target_names=["ham","spam"]))

    st.subheader("Try your own message")
    msg = st.text_input("✍️ Enter a message (English)", "you won a free prize, claim now")
    if msg.strip():
        proba_spam = nb.predict_proba(vect.transform([msg]))[0, 1]
        pred = "SPAM" if proba_spam >= 0.5 else "HAM"
        st.write(f"**Prediction:** {pred}  |  **P(spam)** = {proba_spam:.3f}")
