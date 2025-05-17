import pandas as pd
import pickle
import streamlit as st
from datetime import datetime

# Set the page title and description
st.title("Used Car Price Predictor")
st.write("""
This app predicts the sale price of used cars produced after year 1990.
""")

# # Optional password protection (remove if not needed)
# password_guess = st.text_input("Please enter your password?")
# # this password is stores in streamlit secrets
# if password_guess != st.secrets["password"]:
#     st.stop()

# Load dataset to get dropdown values
df = pd.read_csv("data/processed/Cleaned_Car_Price.csv")

# Load the pre-trained model and encoder
with open("models/RFmodel.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open("models/loo_encoder.pkl", "rb") as f:
    loo_encoder = pickle.load(f)


# Prepare the form to collect user inputs
with st.form("user_inputs"):
    st.subheader("Car Details")
    
    # Dropdown inputs
    manufacturer = st.selectbox("Manufacturer", sorted(df['Manufacturer'].unique()))
    model = st.selectbox("Model", sorted(df['Model'].unique()))
    fuel_type = st.selectbox("Fuel Type", sorted(df['Fuel type'].dropna().unique()))
    engine_volume = st.selectbox("Engine Volume", sorted(df['Engine volume'].dropna().unique()))
    gear_box = st.selectbox("Gear Box Type", sorted(df['Gear box type'].dropna().unique()))
    drive_wheels = st.selectbox("Drive Wheels", sorted(df['Drive wheels'].dropna().unique()))

    # Numeric inputs
    mileage = st.number_input("Mileage (in km)", min_value=0, step=1000)
    cylinders = st.number_input("Cylinders", min_value=1, max_value = 16, step=1)
    prod_year = st.number_input("Production Year", min_value=1990, max_value=datetime.now().year, step=1)
    car_age = datetime.now().year - prod_year
            
    # Submit button
    submitted = st.form_submit_button("Predict Car Price")


# Handle the dummy variables to pass to the model
if submitted:
    input = {
        "Model": model,
        "Mileage": mileage,
        "Cylinders": cylinders,
        "Car Age": car_age,
        "Manufacturer": manufacturer,
        "Fuel type": fuel_type,
        "Engine volume": engine_volume,
        "Gear box type": gear_box,
        "Drive wheels": drive_wheels      
    }

    input_df = pd.DataFrame([input])

    def preprocess_input(input_df, ref_df):
        # Combine input with ref data to ensure same dummy columns
        df = pd.concat([input_df, ref_df], ignore_index=True)

        # Encode 'Model' feature
        df['Model'] = loo_encoder.transform(df[['Model']])

        # One-hot encode other categorical features
        df = pd.get_dummies(df, 
                            columns=[
                                        'Manufacturer', 
                                        'Fuel type', 
                                        'Engine volume',
                                        'Gear box type', 
                                        'Drive wheels'
                                        ], 
                            dtype=int)
               
        # Keep only first row (user input)
        return df.iloc[0:1]
     
    # Prepare the input for prediction. This has to go in the same order as it was trained
    prediction_input = preprocess_input(input_df, df)
    print(prediction_input)
    
    # Make prediction and display result
    try:
        new_prediction = rf_model.predict(prediction_input)[0]
        st.success(f"Estimated Price: ${new_prediction:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.write(
    """We used a machine learning (Random Forest) model to predict the price, the features used in this prediction are ranked by relative
    importance below."""
)
st.image("feature_importance.png")
