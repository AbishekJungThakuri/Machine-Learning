import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the scaler
with open("scalar.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("🏠 House Price Prediction")

# Reusable Yes/No selector
def yes_no_to_binary(label):
    return 1 if st.selectbox(label, ['Yes', 'No']) == 'Yes' else 0

# User Inputs
area = st.number_input("Area", value=5000)
bedrooms = st.number_input("Bedrooms", value=3)
bathrooms = st.number_input("Bathrooms", value=2)
stories = st.number_input("Stories", value=1)
mainroad = yes_no_to_binary("Main Road")
guestroom = yes_no_to_binary("Guest Room")
basement = yes_no_to_binary("Basement")
hotwaterheating = yes_no_to_binary("Hot Water Heating")
airconditioning = yes_no_to_binary("Air Conditioning")
parking = st.number_input("Parking", value=1)
prefarea = yes_no_to_binary("Preferred Area")

# Furnishing status (one-hot encoding)
furnishing = st.selectbox("Furnishing Status", ["furnished", "semi-furnished", "unfurnished"])
furnishingstatus_furnished = 1 if furnishing == "furnished" else 0
furnishingstatus_semi_furnished = 1 if furnishing == "semi-furnished" else 0
furnishingstatus_unfurnished = 1 if furnishing == "unfurnished" else 0

# Scale the area input
area_scaled = scaler.transform(pd.DataFrame([[area]],columns=['area']))[0][0]


# Final input DataFrame
input_data = pd.DataFrame([{
    "area": area_scaled,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "stories": stories,
    "mainroad": mainroad,
    "guestroom": guestroom,
    "basement": basement,
    "hotwaterheating": hotwaterheating,
    "airconditioning": airconditioning,
    "parking": parking,
    "prefarea": prefarea,
    "furnishingstatus_furnished": furnishingstatus_furnished,
    "furnishingstatus_semi-furnished": furnishingstatus_semi_furnished,
    "furnishingstatus_unfurnished": furnishingstatus_unfurnished
}])

# # Prediction
# if st.button("Predict Price"):
#     price = model.predict(input_data)[0]   
#     st.success(f"🏷️ Predicted House Price: RS {int(price):,}")


# Prediction
if st.button("Predict Price"):
    log_price = model.predict(input_data)[0]  # this is in log scale
    actual_price = np.expm1(log_price)        # reverse log1p
    st.success(f"🏷️ Predicted House Price: RS {int(actual_price):,}")
