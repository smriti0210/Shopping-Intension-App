import streamlit as st
import numpy as np
import pickle

st.markdown("""
# Online Shopper Purchase Prediction App

Welcome to the **Online Shopper Purchase Prediction** web app!  
This tool helps e-commerce platforms predict whether a visitor will make a purchase, based on their browsing behavior.

### How It Works:
- The app uses a **Random Forest Classifier** trained on the [UCI Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset).
- You can input various session details like:
  - Number of pages visited
  - Time spent on different types of pages
  - Bounce/Exit rates
  - Visitor type, OS, month, and more!
- The model instantly predicts whether the session is **likely to result in a purchase**.

Explore the power of **machine learning in e-commerce analytics** in a user-friendly and interactive way!
""")


# Load the trained model and scaler
model = pickle.load(open("rf_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Online Shopper Purchase Prediction")
st.write("Provide session details below to predict if a user will complete a purchase.")

# 1. Session Activity Features
st.header("Session Activity")
administrative = st.number_input("Number of Administrative Pages Visited", 0, 100, 1)
administrative_duration = st.number_input("Time Spent on Administrative Pages (seconds)", 0.0, 10000.0, 60.0)

informational = st.number_input("Number of Informational Pages Visited", 0, 100, 1)
informational_duration = st.number_input("Time Spent on Informational Pages (seconds)", 0.0, 10000.0, 30.0)

product_related = st.number_input("Number of Product-Related Pages Visited", 0, 2000, 10)
product_related_duration = st.number_input("Time Spent on Product Pages (seconds)", 0.0, 10000.0, 120.0)

# 2. Behavior Metrics
st.header("Engagement Metrics")
bounce_rate = st.slider("Bounce Rate (0 = low, 1 = high)", 0.0, 1.0, 0.2)
exit_rate = st.slider("Exit Rate (0 = low, 1 = high)", 0.0, 1.0, 0.3)
page_value = st.number_input("Page Value (Estimated Revenue per Page)", 0.0, 500.0, 10.0)
special_day = st.selectbox("Special Day Score (0 = Normal Day, 1 = Very Close to Holiday)", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

# 3. Technical & Session Info
st.header("Technical Details")
month_dict = {
    "Jan": 0, "Feb": 1, "Mar": 2, "Apr": 3, "May": 4, "June": 5,
    "Jul": 6, "Aug": 7, "Sep": 8, "Oct": 9, "Nov": 10, "Dec": 11
}
month_name = st.selectbox("Month of the Visit", list(month_dict.keys()))
month = month_dict[month_name]

os_dict = {
    "Windows": 1, "Mac": 2, "Linux": 3, "Other": 4
}
os_name = st.selectbox("Operating System", list(os_dict.keys()))
operating_systems = os_dict[os_name]

browser = st.slider("Browser Type (1=Chrome, 2=Firefox, etc.)", 1, 13, 2)
region = st.slider("Region (1–9)", 1, 9, 1)
traffic_type = st.slider("Traffic Type (e.g., Direct=1, Referral=2...)", 1, 20, 1)

# 4. Visitor Info
st.header("Visitor Details")
visitor_dict = {
    "Returning Visitor": 0,
    "New Visitor": 1,
    "Other": 2
}
visitor_label = st.selectbox("Visitor Type", list(visitor_dict.keys()))
visitor_type = visitor_dict[visitor_label]

weekend = st.radio("Did the Visit Happen on a Weekend?", ["No", "Yes"])
weekend = 1 if weekend == "Yes" else 0

# Prediction
if st.button("Predict Purchase Intent"):
    input_data = np.array([[
        administrative, administrative_duration,
        informational, informational_duration,
        product_related, product_related_duration,
        bounce_rate, exit_rate, page_value, special_day,
        month, operating_systems, browser, region,
        traffic_type, visitor_type, weekend
    ]])

    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)[0]

    st.markdown("---")
    if prediction == 1:
        st.success("This shopper is likely to make a **purchase**.")
    else:
        st.warning("This shopper is **unlikely to make a purchase**.")
