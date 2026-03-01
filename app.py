import streamlit as st
import pandas as pd
import joblib

# 1. Page Configuration
st.set_page_config(page_title="Flight Fare Predictor", layout="wide")
st.title("✈️ Flight Fare Prediction Dashboard")

# 2. Load Model and Metadata (Cached for performance)
@st.cache_resource
def load_artifacts():
    model = joblib.load("artifacts/models/xgboost_fare_model.pkl")
    metadata = joblib.load("artifacts/models/model_metadata.pkl")
    return model, metadata

try:
    model, metadata = load_artifacts()
except FileNotFoundError:
    st.error("Model artifacts not found. Please run main.py first to generate them.")
    st.stop()

# 3. Build the User Input Form
st.sidebar.header("Enter Flight Details")

def user_input_features():
    # Use the metadata to populate dropdowns dynamically
    cats = metadata["categories"]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Origin Details (City 1)")
        city_1 = st.selectbox("Origin City", cats["city_1"])
        state_1 = st.selectbox("Origin State", cats["state_1"])
        metro_1 = st.selectbox("Origin Metro", cats["metro_1"])
        median_income_1 = st.number_input("Median Income 1", value=50000)
        tot_pax_1 = st.number_input("Total Fared Pax (City 1)", value=1000)
        tot_lf_1 = st.number_input("Total Per LF Mkts (City 1)", value=500)
        tot_prem_1 = st.number_input("Total Per Prem (City 1)", value=100)

    with col2:
        st.subheader("Destination Details (City 2)")
        city_2 = st.selectbox("Destination City", cats["city_2"])
        state_2 = st.selectbox("Destination State", cats["state_2"])
        metro_2 = st.selectbox("Destination Metro", cats["metro_2"])
        median_income_2 = st.number_input("Median Income 2", value=50000)
        tot_pax_2 = st.number_input("Total Fared Pax (City 2)", value=1000)
        tot_lf_2 = st.number_input("Total Per LF Mkts (City 2)", value=500)
        tot_prem_2 = st.number_input("Total Per Prem (City 2)", value=100)

    st.subheader("Route & Carrier Details")
    col3, col4, col5 = st.columns(3)
    with col3:
        carrier_low = st.selectbox("Low Cost Carrier", cats["carrier_low"])
        year = st.number_input("Year", min_value=2000, max_value=2030, value=2024)
        quarter = st.selectbox("Quarter", [1, 2, 3, 4])
    with col4:
        nsmiles = st.number_input("Distance (miles)", value=1000.0)
        passengers = st.number_input("Passengers", value=200.0)
        fare_real = st.number_input("Real Fare", value=150.0)
    with col5:
        large_ms = st.number_input("Large MS", value=0.5)
        lf_ms = st.number_input("LF MS", value=0.5)

    # Compile into a dictionary
    data = {
        "city_1": city_1, "city_2": city_2, "state_1": state_1, "state_2": state_2,
        "carrier_low": carrier_low, "metro_1": metro_1, "metro_2": metro_2,
        "Year": year, "quarter": quarter, "nsmiles": nsmiles, "passengers": passengers,
        "fare_real": fare_real, "large_ms": large_ms, "lf_ms": lf_ms,
        "TotalFaredPax_city1": tot_pax_1, "TotalPerLFMkts_city1": tot_lf_1, "TotalPerPrem_city1": tot_prem_1,
        "TotalFaredPax_city2": tot_pax_2, "TotalPerLFMkts_city2": tot_lf_2, "TotalPerPrem_city2": tot_prem_2,
        "median_income_1": median_income_1, "median_income_2": median_income_2
    }
    
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# 4. Format and Predict
st.markdown("---")
if st.button("Predict Fare Gap / Target", type="primary"):
    # Ensure column order matches training EXACTLY
    input_df = input_df[metadata["columns"]]
    
    # Cast categories explicitly to prevent XGBoost ValueError
    for col, categories in metadata["categories"].items():
        input_df[col] = pd.Categorical(input_df[col], categories=categories)

    # Predict
    prediction = model.predict(input_df)
    
    st.success(f"### Predicted Value: {prediction[0]:.2f}")
    
    with st.expander("View Input Data"):
        st.dataframe(input_df)