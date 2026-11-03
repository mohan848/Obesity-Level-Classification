
import streamlit as strlit
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

strlit.set_page_config(
    page_title="Obesity Level Classification",
    page_icon="",
    layout="wide"
)

model_dir = Path("model")
target = "NObeyesdad"

trained_model_files = {
    "Support Vector Machine": "support_vector_machine.joblib",
    "Decision Tree": "decision_tree.joblib",
    "Logistic Regression": "logistic_regression.joblib",
    "K-Nearest Neighbors": "k_nearest_neighbors.joblib",
    "Gaussian Naive Bayes": "naive_bayes_gaussian.joblib"
}


@strlit.cache_resource
def load_models():
    models = {}

    for name, filename in trained_model_files.items():
        model_path = model_dir / filename

        if model_path.exists():
            models[name] = joblib.load(model_path)

    return models


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    class_labels = model.classes_

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC (macro OvR)": np.nan,
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

    return metrics, y_pred, class_labels


# header

strlit.markdown(
    """
    <div style="
        text-align: center;
        padding: 25px 20px;
        background-color: rgba(70, 130, 180, 0.08);
        border: 1px solid rgba(70, 130, 180, 0.20);
        border-radius: 12px;
        margin-bottom: 20px;
    ">
        <h1 style="
            font-size: 38px;
            margin-bottom: 8px;
            color: inherit;
        ">
            Obesity Level Classification
        </h1>
        <p style="
            font-size: 17px;
            color: gray;
            margin-bottom: 0;
        ">
            Evaluation of Machine Learning Classification Models
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

strlit.divider()

# upload test dataset

strlit.subheader("Upload Test Dataset")

strlit.info(
    "Upload a CSV file containing the original input features "
    "and the target column 'NObeyesdad'. The saved models will be "
    "used for evaluation without retraining."
)

test_csv_file = strlit.file_uploader(
    "Choose a test data in CSV format",
    type=["csv"]
)

if test_csv_file is not None:

    try:
        test_df = pd.read_csv(test_csv_file)
    except Exception as error:
        strlit.error(f"Unable to read the uploaded CSV file: {error}")
        strlit.stop()

    if target not in test_df.columns:
        strlit.error(f"Required target column '{target}' is missing.")
        strlit.stop()

    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    if y_test.isnull().any():
        strlit.error("The target column contains missing values.")
        strlit.stop()

    strlit.success("Test data uploaded successfully.")

    col1, col2, col3 = strlit.columns(3)

    col1.metric("Total Records", len(test_df))
    col2.metric("Input Features", X_test.shape[1])
    col3.metric("Target Classes", y_test.nunique())

    with strlit.expander("Preview Uploaded Dataset"):
        strlit.dataframe(test_df.head(), use_container_width=True)

    # Load models

    models = load_models()

    if not models:
        strlit.error("No trained models found in the model folder.")
        strlit.stop()

    missing_models = set(trained_model_files) - set(models)

    if missing_models:
        strlit.warning(
            "Below model files are missing: "
            + ", ".join(missing_models)
        )

    # Model evaluation

    results = {}
    predictions = {}

    for name, model in models.items():
        try:
            metrics, y_pred, class_labels = evaluate_model(
                model, X_test, y_test
            )

            results[name] = metrics
            predictions[name] = (y_pred, class_labels)

        except Exception as error:
            strlit.error(f"Error evaluating {name}: {error}")

    if not results:
        strlit.error("Unable to evaluate any of the trained models.")
        strlit.stop()

    #MODEL SELECTION 

    strlit.divider()
    strlit.subheader("Model Selection")

    selected_model = strlit.selectbox(
        "Select a model for detailed evaluation",
        list(results.keys())
    )



    #SELECTED MODEL DETAILS 
    strlit.divider()
    strlit.subheader(f"Detailed Evaluation: {selected_model}")

    selected_metrics = results[selected_model]

    metric_names = [
        "Accuracy",
        "AUC (macro OvR)",
        "Precision (macro)",
        "Recall (macro)",
        "F1 (macro)",
        "MCC"
    ]

    eval_metric_cols = strlit.columns(3)

    for index, metric_name in enumerate(metric_names):
        value = selected_metrics[metric_name]

        eval_metric_cols[index % 3].metric(
            metric_name,
            f"{value:.4f}" if pd.notna(value) else "N/A"
        )

    y_pred, class_labels = predictions[selected_model]

    # CONFUSION MATRIX 
    strlit.divider()
    strlit.subheader("Confusion Matrix")

    confusion_matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=class_labels
    )

    confusion_matrix_df = pd.DataFrame(
        confusion_matrix,
        index=class_labels,
        columns=class_labels
    )

    strlit.caption(
        "Rows represent actual categories and columns represent "
        "predicted categories. Diagonal values indicate correct predictions."
    )

    strlit.dataframe(
        confusion_matrix_df,
        use_container_width=True
    )
    


    # CLASSIFICATION REPORT

    strlit.subheader("Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        labels=class_labels,
        zero_division=0,
        output_dict=True
    )

    report_df = pd.DataFrame(report).T

    strlit.dataframe(
        report_df.style.format("{:.4f}"),
        use_container_width=True
    )



    # PERFORMANCE COMPARISON
    strlit.divider()
    strlit.subheader("Model Performance Comparison")

    strlit.caption(
        "Note: Precision, Recall, F1-score and AUC are calculated "
        "using macro averaging. Since this is a multiclass "
        "classification problem. And 'AUC' is calculated using the "
        "One-vs-Rest (OvR) approach."
    )

    results_df = pd.DataFrame(results).T
    results_df.index.name = "Model Name"

    strlit.dataframe(
        results_df.style.format("{:.4f}", na_rep="N/A"),
        use_container_width=True
    )

    
    # OVERALL WINNER

    strlit.divider()
    strlit.subheader("Overall Model Performance Comparision")

    winning_model = max(
        results,
        key=lambda name: results[name]["F1 (macro)"]
    )

    winning_model_metrics = results[winning_model]

    strlit.success(f"Overall Best-Performing Model: {winning_model}")

    winning_model_col1, winning_model_col2, winning_model_col3 = strlit.columns(3)

    winning_model_col1.metric(
        "Macro F1-score",
        f"{winning_model_metrics['F1 (macro)']:.4f}"
    )

    winning_model_col2.metric(
        "Accuracy",
        f"{winning_model_metrics['Accuracy']:.4f}"
    )

    winning_model_col3.metric(
        "MCC",
        f"{winning_model_metrics['MCC']:.4f}"
    )

    strlit.write(
        f"Based on the highest macro F1-score, **{winning_model}** "
        "demonstrates the best overall classification performance "
        "among the evaluated models."
    )

    strlit.caption(
        "Test dataset evaluated on saved models only."
        "No model training is performed on the uploaded test data."
    )

else:
    strlit.warning("Please upload a test CSV file to begin model evaluation.")
