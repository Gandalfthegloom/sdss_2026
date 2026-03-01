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
    
    # Isolate columns for the ACTUAL expected market conditions
    input_features = lookup_row[metadata["columns"]].copy()
    
    # --- NEW: Create Counterfactual "Fair Market" Features ---
    fair_features = input_features.copy()
    
    # 1. Break monopolies: Cap dominant carrier market share at 40%
    if 'large_ms' in fair_features.columns:
        fair_features.loc[fair_features['large_ms'] > 0.40, 'large_ms'] = 0.40
        
    # 2. Remove Fortress Hub Premiums: Set premium to 0% (if it's currently positive)
    if 'TotalPerPrem_city1' in fair_features.columns:
        fair_features.loc[fair_features['TotalPerPrem_city1'] > 0, 'TotalPerPrem_city1'] = 0.0
    if 'TotalPerPrem_city2' in fair_features.columns:
        fair_features.loc[fair_features['TotalPerPrem_city2'] > 0, 'TotalPerPrem_city2'] = 0.0
        
    # 3. Ensure LCC Presence: Set minimum low-cost carrier passenger fraction to 50%
    if 'TotalPerLFMkts_city1' in fair_features.columns:
        fair_features.loc[fair_features['TotalPerLFMkts_city1'] < 0.50, 'TotalPerLFMkts_city1'] = 0.50
    if 'TotalPerLFMkts_city2' in fair_features.columns:
        fair_features.loc[fair_features['TotalPerLFMkts_city2'] < 0.50, 'TotalPerLFMkts_city2'] = 0.50

    # Format categories for XGBoost prediction
    for col, categories in metadata["categories"].items():
        input_features[col] = pd.Categorical(input_features[col], categories=categories)
        fair_features[col] = pd.Categorical(fair_features[col], categories=categories)
        
    # Make Predictions
    # We predict the FAIR fare using our neutralized features
    predicted_fair_fare = model.predict(fair_features)[0]
    
    # Extract the actual fare from your lookup row 
    actual_fare = lookup_row['fare_real'].values[0] 
    
    # Calculate the difference (Premium / Rip-off amount)
    fare_difference = actual_fare - predicted_fair_fare
    
    # Determine the label
    if fare_difference > 15: 
        deal_status = "🚨 Overpriced (Monopoly Premium)"
        delta_color = "inverse" 
    elif fare_difference < -15:
        deal_status = "🎉 Underpriced (Consumer Deal)"
        delta_color = "normal" 
    else:
        deal_status = "⚖️ Fairly Priced"
        delta_color = "off"

    # --- THE UI VISUALIZATION ---
    st.markdown(f"### Route Status: {deal_status}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Actual Average Fare", value=f"${actual_fare:.2f}")
        
    with col2:
        st.metric(label="Model's Fair Price", value=f"${predicted_fair_fare:.2f}")
        
    with col3:
        st.metric(label="Competition Penalty", 
                  value=f"${abs(fare_difference):.2f}", 
                  delta=f"${fare_difference:.2f} vs Fair Price",
                  delta_color=delta_color)
    
    st.markdown("### Why is this route priced this way?")
    st.write("This waterfall chart shows how the route's current structural features (like monopolies or hub premiums) push the expected fare up or down from the national baseline.")
    
    # Create a SHAP-friendly copy of the ACTUAL row (not the fair row)
    # We want SHAP to explain the actual market conditions so the user sees the penalties!
    shap_input = make_shap_friendly(input_features, cat_cols)
    
    explainer = shap.TreeExplainer(model)
    shap_explanation = explainer(shap_input)
    
    # Group Anti-Competitive Features and Rename for UI ---
    import copy
    
    # Get the explanation for the single row we just predicted
    exp = shap_explanation[0] 
    
    # Define the features that make up the "Anti-Competitive" group
    anti_comp_cols = [
        'large_ms', 
        'TotalPerPrem_city1', 'TotalPerPrem_city2', 
        'TotalPerLFMkts_city1', 'TotalPerLFMkts_city2'
    ]
    
    # Map the remaining technical columns to plain English
    friendly_names = {
        'nsmiles': 'Flight Distance',
        'passengers': 'Daily Passenger Demand',
        'lf_ms': 'Budget Airline Market Share',
        'carrier_low': 'Lowest Fare Carrier',
        'TotalFaredTotal': 'Total Market Traffic',
        'Quarter': 'Seasonality (Quarter)'
    }
    
    new_values = []
    new_data = []
    new_names = []
    grouped_penalty_value = 0.0
    
    # Loop through the original SHAP values
    for i, col_name in enumerate(exp.feature_names):
        if col_name in anti_comp_cols:
            # Add the penalty to our grouped total
            grouped_penalty_value += exp.values[i]
        else:
            # Keep the feature separate, but rename it to plain English
            new_values.append(exp.values[i])
            new_data.append(exp.data[i])
            new_names.append(friendly_names.get(col_name, col_name))
            
    # Append our new grouped feature to the list
    new_values.append(grouped_penalty_value)
    new_data.append("") # Leave the raw data label blank for grouped categories
    new_names.append("🚨 Anti-Competitive Penalties (Monopoly/Hub)")
    
    # Rebuild the Explanation object with our clean data
    new_exp = copy.deepcopy(exp)
    new_exp.values = np.array(new_values)
    new_exp.data = np.array(new_data)
    new_exp.feature_names = new_names
    
    # Plot the clean, presentation-ready chart
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(new_exp, show=False)
    plt.tight_layout()
    st.pyplot(fig)