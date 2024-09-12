import streamlit as st
import pandas as pd
import xgboost as xgb
import pickle

# Load your trained model and preprocessing steps
def load_model_and_preprocessing():
    # Load the XGBoost model
    model = xgb.Booster()
    model.load_model('pro1.h5')
    
    # Load preprocessing steps (in this case, no encoders)
    with open('pro2.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    
    return model, label_encoders

xgb_model, label_encoders = load_model_and_preprocessing()

def predict_loan_status(inputs, model):
  
    input_df = pd.DataFrame([inputs])

    input_df = input_df.astype(float)

    dmatrix = xgb.DMatrix(input_df)

    # Predict
    prediction = model.predict(dmatrix)
    return 1 if prediction > 0.5 else 0

# Streamlit app
def main():
    st.title("Loan Status Prediction")

    st.write("Enter the details below to predict the loan status:")

    loan_id = st.text_input("Loan ID")
    no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
    income_annum = st.number_input("Annual Income", min_value=0, value=50000)
    loan_amount = st.number_input("Loan Amount", min_value=0, value=100000)
    loan_term = st.number_input("Loan Term (in days)", min_value=0, value=360)
    cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=750)
    residential_assets_value = st.number_input("Residential Assets Value", min_value=0, value=0)
    commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0, value=0)
    luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0, value=0)
    bank_asset_value = st.number_input("Bank Asset Value", min_value=0, value=0)

    inputs = {
        'no_of_dependents': no_of_dependents,
        'income_annum': income_annum,
        'loan_amount': loan_amount,
        'loan_term': loan_term,
        'cibil_score': cibil_score,
        'residential_assets_value': residential_assets_value,
        'commercial_assets_value': commercial_assets_value,
        'luxury_assets_value': luxury_assets_value,
        'bank_asset_value': bank_asset_value
    }

    if st.button("Predict Loan Status"):
        # Predict the loan status
        prediction = predict_loan_status(inputs, xgb_model)

        if prediction == 1:
            st.success("Loan Status: Approved")
        else:
            st.error("Loan Status: Rejected")

if __name__ == "__main__":
    main()
