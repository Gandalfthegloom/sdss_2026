import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Flight Fare Predictor", layout="wide")
st.title("✈️ Smart Flight Fare Predictor")

# --- Helper Function for SHAP ---
def make_shap_friendly(df, cat_cols):
    df2 = df.copy()
    for col in cat_cols:
        # Ensure true categorical dtype first
        df2[col] = df2[col].astype("category")
        # Convert categories to integer codes for SHAP
        df2[col] = df2[col].cat.codes
        # Replace -1 (missing categories) with NaN
        df2[col] = df2[col].replace(-1, np.nan)
    
    for col in df2.columns:
        if df2[col].dtype == "object":
            df2[col] = pd.to_numeric(df2[col], errors="coerce")
    return df2

# 1. Load Model and Lookup Data
@st.cache_resource
def load_model():
    model = joblib.load("artifacts/models/xgboost_fare_model.pkl")
    metadata = joblib.load("artifacts/models/model_metadata.pkl")
    return model, metadata

@st.cache_data
def load_data():
    return pd.read_csv("Data/Processed/model_ready_airline_fares.csv")

model, metadata = load_model()
df = load_data()

# Identify categorical columns from metadata
cat_cols = list(metadata["categories"].keys())

# 2. Smart User Inputs
st.sidebar.header("Plan Your Flight")

origins = sorted(df['city_1'].dropna().unique())
selected_origin = st.sidebar.selectbox("Origin City", origins, index=None, placeholder="Select an Origin...")

if not selected_origin:
    st.info("👋 Welcome! Please select an Origin City in the sidebar to begin.")
    st.stop()

valid_destinations = sorted(df[df['city_1'] == selected_origin]['city_2'].dropna().unique())
selected_dest = st.sidebar.selectbox("Destination City", valid_destinations, index=None, placeholder="Select a Destination...")

if not selected_dest:
    st.stop()

col1, col2 = st.sidebar.columns(2)
with col1:
    selected_year = st.selectbox("Year", sorted(df['Year'].unique(), reverse=True))
with col2:
    selected_quarter = st.selectbox("Quarter", [1, 2, 3, 4])

# 3. The Lookup Engine
route_df = df[(df['city_1'] == selected_origin) & (df['city_2'] == selected_dest)]
exact_match = route_df[(route_df['Year'] == selected_year) & (route_df['Quarter'] == selected_quarter)]

if not exact_match.empty:
    st.success(f"✅ Found exact historical data for {selected_origin} to {selected_dest} in {selected_year} Q{selected_quarter}.")
    lookup_row = exact_match.iloc[0:1].copy()
else:
    st.warning("⚠️ Exact time period missing. Falling back to the most recent data for this route.")
    lookup_row = route_df.sort_values(by=['Year', 'Quarter']).iloc[-1:].copy()
    lookup_row['Year'] = selected_year
    lookup_row['Quarter'] = selected_quarter

# 4. Display Hidden Features
with st.expander("Peek under the hood (Features being sent to model)"):
    display_cols = [c for c in metadata["columns"] if c != 'fare_real']
    display_cols = [c for c in display_cols if c in lookup_row.columns]
    st.dataframe(lookup_row[display_cols])

# 5. Predict and Explain
st.markdown("---")
if st.button("Predict Fare & Show Explanation", type="primary"):
    
    # Isolate columns
    input_features = lookup_row[metadata["columns"]].copy()
    
    # Format categories for XGBoost prediction
    for col, categories in metadata["categories"].items():
        input_features[col] = pd.Categorical(input_features[col], categories=categories)
        
    # Make Prediction
    prediction = model.predict(input_features)
    st.metric(label="Predicted Real Fare", value=f"${prediction[0]:.2f}")
    
    st.markdown("### Why this fare?")
    st.write("This waterfall chart shows how each feature pushed the predicted fare up or down from the baseline average.")
    
    # Create a SHAP-friendly copy of the exact row we just predicted
    shap_input = make_shap_friendly(input_features, cat_cols)
    
    # Generate SHAP Explanation
    # We use explainer(shap_input) instead of explainer.shap_values() because 
    # the waterfall plot specifically requires an Explanation object
    explainer = shap.TreeExplainer(model)
    shap_explanation = explainer(shap_input)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(shap_explanation[0], show=False)
    plt.tight_layout()
    
    # Render the matplotlib figure in Streamlit
    st.pyplot(fig)