import streamlit as st
import pickle
import pandas as pd


# Load model

model = pickle.load(
    open(
        "models/diabetes_model.pkl",
        "rb"
    )
)


scaler = pickle.load(
    open(
        "models/diabetes_scaler.pkl",
        "rb"
    )
)


encoder = pickle.load(
    open(
       "models/gender_encoder.pkl",
        "rb"
    )
)

encoder = pickle.load(
    open(
       "models/smoking_encoder.pkl",
        "rb"
    )
)



st.title(
    "🩺 AI-Powered Diabetes Predictor"
)


st.write(
    "Predict diabetes risk using health parameters"
)



gender = st.selectbox(
    "Gender",
    ["Male","Female","Other"]
)


age = st.number_input(
    "Age",
    1,
    100
)


hypertension = st.selectbox(
    "Hypertension",
    [0,1]
)


heart = st.selectbox(
    "Heart Disease",
    [0,1]
)


smoking = st.selectbox(
    "Smoking History",
    [
        "never",
        "former",
        "current",
        "No Info"
    ]
)


bmi = st.number_input(
    "BMI",
    10.0,
    60.0
)


hba1c = st.number_input(
    "HbA1c Level",
    3.0,
    15.0
)


glucose = st.number_input(
    "Blood Glucose Level",
    50,
    400
)



if st.button(
    "Predict"
):

    input_data = pd.DataFrame(
        {
        "gender":[gender],
        "age":[age],
        "hypertension":[hypertension],
        "heart_disease":[heart],
        "smoking_history":[smoking],
        "bmi":[bmi],
        "HbA1c_level":[hba1c],
        "blood_glucose_level":[glucose]
        }
    )


    categorical=[
        "gender",
        "smoking_history"
    ]


    for col in categorical:
        input_data[col]=encoder.transform(
            input_data[col]
        )



    scaled=scaler.transform(
        input_data
    )


    result=model.predict(
        scaled
    )[0]



    probability=model.predict_proba(
        scaled
    )[0][1]



    if result==1:

        st.error(
            f"Diabetes Risk Detected ({probability*100:.2f}%)"
        )

    else:

        st.success(
            f"No Diabetes Risk ({probability*100:.2f}%)"
        )
