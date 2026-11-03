
# import libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="Obesity Level Classification",
    page_icon="📊",
    layout="wide"
)

model_directory = Path("model")
target = "NObeyesdad"

# define trained model file names
trained_model_files = {
    "Support Vector Machine": "support_vector_machine.joblib",
    "Decision Tree": "decision_tree.joblib",
    "Logistic Regression": "logistic_regression.joblib",
    "K-Nearest Neighbors": "k_nearest_neighbors.joblib",
    "Gaussian Naive Bayes": "naive_bayes_gaussian.joblib"
}


@st.cache_resource
def load_models():
    models = {}

    for name, filename in trained_model_files.items():
        model_path = model_directory / filename

        if model_path.exists():
            models[name] = joblib.load(model_path)

    return models


# to evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    class_labels = model.classes_

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision (macro)": precision_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "Recall (macro)": recall_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "F1 (macro)": f1_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)

        metrics["AUC (macro OvR)"] = roc_auc_score(
            y_test,
            y_proba,
            labels=class_labels,
            multi_class="ovr",
            average="macro"
        )
    else:
        metrics["AUC (macro OvR)"] = np.nan

    return metrics, y_pred, class_labels

# front end labels
st.title("Obesity Level Classification")
st.write(
    "Evaluate five machine learning classification models "
    "using the uploaded test dataset."
)

st.info(
    "Upload a test CSV containing all original input features "
    "and the target column NObeyesdad."
)

uploaded_file = st.file_uploader(
    "Upload Test Dataset (CSV)",
    type=["csv"]
)

# read csv file
if uploaded_file is not None:

    test_df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")
    st.write(f"Number of records: {len(test_df)}")
    st.dataframe(test_df.head())

    if target not in test_df.columns:
        st.error(f"target column '{target}' is missing.")
        st.stop()

    # drop target variable
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    # load models
    models = load_models()

    # if no models found
    if not models:
        st.error("No trained models found.")
        st.stop()

    missing_models = set(trained_model_files) - set(models)

    # if some are missing
    if missing_models:
        st.warning(
            "Some model files are missing: "
            + ", ".join(missing_models)
        )

    results = {}
    predictions = {}

    # evaluate the model performace for all models
    for name, model in models.items():
        try:
            metrics, y_pred, class_labels = evaluate_model(
                model, X_test, y_test
            )

            results[name] = metrics
            predictions[name] = (y_pred, class_labels)

        except Exception as error:
            st.error(f"Error evaluating {name}: {error}")

    # show model performance
    if results:

        st.subheader("Model Performance Comparison")

        results_df = pd.DataFrame(results).T
        st.dataframe(
            results_df.style.format("{:.4f}"),
            use_container_width=True
        )

        st.subheader("Detailed Model Evaluation")

        selected_model = st.selectbox(
            "Select a model",
            list(results.keys())
        )

        selected_metrics = results[selected_model]

        cols = st.columns(3)

        metric_names = [
            "Accuracy",
            "AUC (macro OvR)",
            "Precision (macro)",
            "Recall (macro)",
            "F1 (macro)",
            "MCC"
        ]

        for index, metric_name in enumerate(metric_names):
            value = selected_metrics[metric_name]

            cols[index % 3].metric(
                metric_name,
                f"{value:.4f}" if pd.notna(value) else "N/A"
            )

        y_pred, class_labels = predictions[selected_model]

        st.subheader("Confusion Matrix")

        confustion_matrix = confusion_matrix(
            y_test,
            y_pred,
            labels=class_labels
        )

        confustion_matrix_df = pd.DataFrame(
            confustion_matrix,
            index=class_labels,
            columns=class_labels
        )

        st.dataframe(confustion_matrix_df, use_container_width=True)

        st.subheader("Classification Report")

        report = classification_report(
            y_test,
            y_pred,
            labels=class_labels,
            zero_division=0,
            output_dict=True
        )

        st.dataframe(
            pd.DataFrame(report).T.style.format("{:.4f}"),
            use_container_width=True
        )

        st.success("Model evaluation completed successfully.")
