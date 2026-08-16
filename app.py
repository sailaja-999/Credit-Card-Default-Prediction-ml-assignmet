import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Credit Card Default Prediction",
    page_icon="💳",
    layout="wide"
)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("💳 Credit Card Default Prediction System")

st.markdown("""
This application evaluates multiple Machine Learning models for predicting whether
a credit card customer will default on payment in the following month.

### Instructions
1. Select a Machine Learning Model.
2. Upload the test_data.csv file.
3. View evaluation metrics.
4. Analyze confusion matrix and classification report.
5. Download prediction results.
""")

# --------------------------------------------------
# MODEL FILES
# --------------------------------------------------

model_files = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl"
}

# --------------------------------------------------
# MODEL SELECTION
# --------------------------------------------------

model_name = st.selectbox(
    "🤖 Select Machine Learning Model",
    list(model_files.keys())
)

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "📂 Upload Test Dataset (CSV)",
    type=["csv"]
)

# --------------------------------------------------
# PROCESS DATA
# --------------------------------------------------

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head())

    if "Actual" not in df.columns:

        st.error(
            "The uploaded CSV must contain a column named 'Actual'."
        )

    else:

        try:

            y_true = df["Actual"]
            X = df.drop("Actual", axis=1)

            # Load Model
            model = joblib.load(
                model_files[model_name]
            )

            # Load Scaler
            scaler = joblib.load(
                "scaler.pkl"
            )

            # Apply scaling
            if model_name in [
                "Logistic Regression",
                "KNN"
            ]:
                X_input = scaler.transform(X)
            else:
                X_input = X

            # Predictions
            predictions = model.predict(X_input)

            # AUC
            try:
                probabilities = model.predict_proba(
                    X_input
                )[:, 1]

                auc = roc_auc_score(
                    y_true,
                    probabilities
                )

            except:
                auc = 0

            # Metrics
            accuracy = accuracy_score(
                y_true,
                predictions
            )

            precision = precision_score(
                y_true,
                predictions
            )

            recall = recall_score(
                y_true,
                predictions
            )

            f1 = f1_score(
                y_true,
                predictions
            )

            mcc = matthews_corrcoef(
                y_true,
                predictions
            )

            # --------------------------------------------------
            # METRICS
            # --------------------------------------------------

            st.subheader("📈 Evaluation Metrics")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Accuracy",
                f"{accuracy:.4f}"
            )

            col2.metric(
                "AUC",
                f"{auc:.4f}"
            )

            col3.metric(
                "Precision",
                f"{precision:.4f}"
            )

            col4, col5, col6 = st.columns(3)

            col4.metric(
                "Recall",
                f"{recall:.4f}"
            )

            col5.metric(
                "F1 Score",
                f"{f1:.4f}"
            )

            col6.metric(
                "MCC",
                f"{mcc:.4f}"
            )

            # --------------------------------------------------
            # CONFUSION MATRIX
            # --------------------------------------------------

            st.subheader("📉 Confusion Matrix")

            cm = confusion_matrix(
                y_true,
                predictions
            )

            cm_df = pd.DataFrame(
                cm,
                index=[
                    "Actual No Default",
                    "Actual Default"
                ],
                columns=[
                    "Predicted No Default",
                    "Predicted Default"
                ]
            )

            st.dataframe(cm_df)

            # --------------------------------------------------
            # CLASSIFICATION REPORT
            # --------------------------------------------------

            st.subheader("📄 Classification Report")

            report = classification_report(
                y_true,
                predictions,
                output_dict=True
            )

            report_df = pd.DataFrame(
                report
            ).transpose()

            st.dataframe(report_df)

            # --------------------------------------------------
            # PREDICTIONS
            # --------------------------------------------------

            st.subheader("🔍 Prediction Results")

            result_df = pd.DataFrame({
                "Actual": y_true,
                "Predicted": predictions
            })

            st.dataframe(result_df)

            # --------------------------------------------------
            # DOWNLOAD
            # --------------------------------------------------

            csv = result_df.to_csv(
                index=False
            )

            st.download_button(
                label="📥 Download Prediction Results",
                data=csv,
                file_name="prediction_results.csv",
                mime="text/csv"
            )

        except Exception as e:

            st.error(
                f"Error while processing file: {str(e)}"
            )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")
st.markdown(
    "Developed for BITS Pilani Machine Learning Assignment 2"
)