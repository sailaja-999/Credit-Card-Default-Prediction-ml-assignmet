import streamlit as st
import pandas as pd
import joblib

st.header("💳 Credit Card Default Prediction")

model_name = st.selectbox(
    "Choose Prediction Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest"
    ]
)

model_files = {
    "Logistic Regression":"logistic_regression.pkl",
    "Decision Tree":"decision_tree.pkl",
    "KNN":"knn.pkl",
    "Naive Bayes":"naive_bayes.pkl",
    "Random Forest":"random_forest.pkl"
}

with st.form("prediction_form"):

    st.subheader("Customer Information")

    col1, col2 = st.columns(2)

    with col1:
        LIMIT_BAL = st.number_input(
            "Credit Limit",
            min_value=10000,
            value=50000
        )

        SEX = st.selectbox(
            "Gender",
            [1, 2],
            format_func=lambda x:
            "Male" if x == 1 else "Female"
        )

        EDUCATION = st.selectbox(
            "Education",
            [1,2,3,4]
        )

        MARRIAGE = st.selectbox(
            "Marital Status",
            [1,2,3]
        )

        AGE = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30
        )

    with col2:
        PAY_0 = st.number_input(
            "Sep Repayment Status",
            value=0
        )

        PAY_2 = st.number_input(
            "Aug Repayment Status",
            value=0
        )

        PAY_3 = st.number_input(
            "Jul Repayment Status",
            value=0
        )

        PAY_4 = st.number_input(
            "Jun Repayment Status",
            value=0
        )

        PAY_5 = st.number_input(
            "May Repayment Status",
            value=0
        )

        PAY_6 = st.number_input(
            "Apr Repayment Status",
            value=0
        )

    st.subheader("Bill Amounts")

    bcol1, bcol2, bcol3 = st.columns(3)

    with bcol1:
        BILL_AMT1 = st.number_input("Bill Amount Sept", value=0)
        BILL_AMT2 = st.number_input("Bill Amount Aug", value=0)

    with bcol2:
        BILL_AMT3 = st.number_input("Bill Amount Jul", value=0)
        BILL_AMT4 = st.number_input("Bill Amount Jun", value=0)

    with bcol3:
        BILL_AMT5 = st.number_input("Bill Amount May", value=0)
        BILL_AMT6 = st.number_input("Bill Amount Apr", value=0)

    st.subheader("Previous Payments")

    pcol1, pcol2, pcol3 = st.columns(3)

    with pcol1:
        PAY_AMT1 = st.number_input("Payment Sept", value=0)
        PAY_AMT2 = st.number_input("Payment Aug", value=0)

    with pcol2:
        PAY_AMT3 = st.number_input("Payment Jul", value=0)
        PAY_AMT4 = st.number_input("Payment Jun", value=0)

    with pcol3:
        PAY_AMT5 = st.number_input("Payment May", value=0)
        PAY_AMT6 = st.number_input("Payment Apr", value=0)

    predict_btn = st.form_submit_button(
        "🔍 Predict Default Risk"
    )

if predict_btn:

    model = joblib.load(model_files[model_name])

    scaler = joblib.load(
        "scaler.pkl"
    )

    input_df = pd.DataFrame([{
        "LIMIT_BAL": LIMIT_BAL,
        "SEX": SEX,
        "EDUCATION": EDUCATION,
        "MARRIAGE": MARRIAGE,
        "AGE": AGE,
        "PAY_0": PAY_0,
        "PAY_2": PAY_2,
        "PAY_3": PAY_3,
        "PAY_4": PAY_4,
        "PAY_5": PAY_5,
        "PAY_6": PAY_6,
        "BILL_AMT1": BILL_AMT1,
        "BILL_AMT2": BILL_AMT2,
        "BILL_AMT3": BILL_AMT3,
        "BILL_AMT4": BILL_AMT4,
        "BILL_AMT5": BILL_AMT5,
        "BILL_AMT6": BILL_AMT6,
        "PAY_AMT1": PAY_AMT1,
        "PAY_AMT2": PAY_AMT2,
        "PAY_AMT3": PAY_AMT3,
        "PAY_AMT4": PAY_AMT4,
        "PAY_AMT5": PAY_AMT5,
        "PAY_AMT6": PAY_AMT6
    }])

    if model_name in [
        "Logistic Regression",
        "KNN"
    ]:
        input_data = scaler.transform(input_df)
    else:
        input_data = input_df

    prediction = model.predict(
        input_data
    )[0]

    try:
        probability = model.predict_proba(
            input_data
        )[0][1]
    except:
        probability = 0

    st.markdown("---")

    if prediction == 1:

        st.error(
            "⚠️ HIGH RISK: Customer is likely to DEFAULT next month"
        )

        st.metric(
            "Default Probability",
            f"{probability*100:.2f}%"
        )

    else:

        st.success(
            "✅ LOW RISK: Customer is unlikely to default next month"
        )

        st.metric(
            "Confidence",
            f"{(1-probability)*100:.2f}%"
        )