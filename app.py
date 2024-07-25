import streamlit as st
import pickle
import numpy as np

# Load the saved model
with open('randomForest.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Streamlit UI
st.title("BestCard Credit Default Prediction")
st.markdown("<h3 style='color: white; text-align: center;'>Presented by Team 6</h3>", unsafe_allow_html=True)

# Input features
LIMIT_BAL = st.text_input("LIMIT_BAL", "5000")
st.divider()

SEX = st.selectbox("SEX", [1, 2])
st.write("1 = male; 2 = female")
st.divider()

EDUCATION = st.selectbox("EDUCATION", [1, 2, 3, 4])
st.write("1 = graduate school; 2 = university; 3 = high school; 4 = others")
st.divider()

MARRIAGE = st.selectbox("MARRIAGE", [1, 2, 3])
st.write("1 = married; 2 = single; 3 = others")
st.divider()
AGE = st.slider("AGE", 18, 80, 25)
st.divider()
st.write("PAY1 - PAY6: History of past 6 months payment.")
st.write("The measurement scale for the repayment status is: -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above.")

PAY_1 = st.slider("PAY_1", -2, 8, 0)
PAY_2 = st.slider("PAY_2", -2, 8, 0)
PAY_3 = st.slider("PAY_3", -2, 8, 0)
PAY_4 = st.slider("PAY_4", -2, 8, 0)
PAY_5 = st.slider("PAY_5", -2, 8, 0)
PAY_6 = st.slider("PAY_6", -2, 8, 0)
st.divider()
st.write("BILL_AMT1 - BILL_AMT6: Amount of bill statement of past 6 months (NT dollar). ")
BILL_AMT1 = st.text_input("BILL_AMT1", "5000")
BILL_AMT2 = st.text_input("BILL_AMT2", "5000")
BILL_AMT3 = st.text_input("BILL_AMT3", "5000")
BILL_AMT4 = st.text_input("BILL_AMT4", "5000")
BILL_AMT5 = st.text_input("BILL_AMT5", "5000")
BILL_AMT6 = st.text_input("BILL_AMT6", "5000")
st.divider()
st.write("PAY_AMT1 - PAY_AMT6: Amount of previous payment (NT dollar) ")
PAY_AMT1 = st.text_input("PAY_AMT1", "1000")
PAY_AMT2 = st.text_input("PAY_AMT2", "1000")
PAY_AMT3 = st.text_input("PAY_AMT3", "1000")
PAY_AMT4 = st.text_input("PAY_AMT4", "1000")
PAY_AMT5 = st.text_input("PAY_AMT5", "1000")
PAY_AMT6 = st.text_input("PAY_AMT6", "1000")
st.divider()

# Make prediction
features = np.array([LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, PAY_1, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6,
                     BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6, PAY_AMT1, PAY_AMT2,
                     PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6]).reshape(1, -1)

if st.button("Predict"):
    prediction = model.predict(features)
    if prediction[0] == 0:
        st.write("Prediction:", prediction[0], " :", "Not Default")
    elif prediction[0] == 1:
        st.write("Prediction:", prediction[0], " :", "Default")


