import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import copy
from scripts.real_to_nominal import real_to_nominal_simple

st.set_page_config(page_title="Fair Fare Explorer", layout="wide")
st.title("✈️ The 'Fair Fare' Route Explorer")

# --- Helper Function for SHAP if using XGBOOST---
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
    model = joblib.load("artifacts/models/lgbm_fare_model.pkl") # use lgbm
    metadata = joblib.load("artifacts/models/lgbm_model_metadata.pkl")
    return model, metadata

@st.cache_data
def load_data():
    df = pd.read_csv("Data/Processed/model_ready_airline_fares.csv")
    
    # --- CRITICAL FIX: Re-apply binning logic here ---
    # If the CSV wasn't actively overwritten with the new string labels, this 
    # catches the old floats and safely converts them so they don't become NaN!
    if 'large_ms' in df.columns and pd.api.types.is_numeric_dtype(df['large_ms']):
        df['large_ms'] = pd.cut(df['large_ms'], bins=[-np.inf, 0.40, 0.70, np.inf], labels=['Highly_Competitive', 'Moderately_Concentrated', 'Monopoly_Route']).astype(str)
        
    for col in ['TotalPerLFMkts_city1', 'TotalPerLFMkts_city2']:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.cut(df[col], bins=[-np.inf, 0.20, 0.60, np.inf], labels=['LCC_Deficient', 'Healthy_Competition', 'LCC_Monopoly']).astype(str)
            
    for col in ['TotalPerPrem_city1', 'TotalPerPrem_city2']:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.cut(df[col], bins=[-np.inf, 0.0, 0.10, np.inf], labels=['Discount_Hub', 'Neutral/Slight_Premium', 'High_Premium_Hub']).astype(str)
            
    return df

@st.cache_data
def load_cpi_data():
    # Adjust path if needed
    return pd.read_excel("Data/Raw/CPI US.xlsx", sheet_name="Monthly")

# --- NARRATIVE GENERATOR ---
def generate_route_narrative(features_row, penalty_val):
    """Snaps together pre-written text blocks based on categorical features"""
    sentences = []
    
    # 1. Evaluate Monopoly/Dominance
    dom = features_row['large_ms'].iloc[0]
    if dom == 'Monopoly_Route':
        sentences.append("This route is heavily dominated by a single carrier, acting as a **Monopoly** that significantly inflates prices.")
    elif dom == 'Moderately_Concentrated':
        sentences.append("This route is **moderately concentrated**, meaning a few airlines control most of the traffic, which limits competitive pricing.")
    else:
        sentences.append("Passengers benefit from a **highly competitive** market on this route, helping to keep base fares in check.")

    # 2. Evaluate Low Cost Carrier (LCC) Presence
    lcc_o = features_row['TotalPerLFMkts_city1'].iloc[0]
    lcc_d = features_row['TotalPerLFMkts_city2'].iloc[0]
    if 'LCC_Monopoly' in [lcc_o, lcc_d]:
         sentences.append("Paradoxically, a budget airline has established a **low-cost monopoly** at one or both of these airports; without fierce competition, they are pricing closer to legacy carriers.")
    elif 'LCC_Deficient' in [lcc_o, lcc_d]:
         sentences.append("A **lack of budget airlines** at the departure or arrival airport leaves consumers with fewer cheap alternatives.")
    elif lcc_o == 'Healthy_Competition' and lcc_d == 'Healthy_Competition':
         sentences.append("A **healthy presence of budget airlines** at both airports provides strong downward pressure on ticket prices.")

    # 3. Evaluate Hub Premiums
    hub_o = features_row['TotalPerPrem_city1'].iloc[0]
    hub_d = features_row['TotalPerPrem_city2'].iloc[0]
    if 'High_Premium_Hub' in [hub_o, hub_d]:
         sentences.append("Additionally, you are paying a **fortress hub premium**—major airlines use their dominance at these specific airports to charge higher baseline fees.")
    elif 'Discount_Hub' in [hub_o, hub_d]:
         sentences.append("Fortunately, flying through a **discount-friendly hub** is helping to subsidize the overall cost of the trip.")

    # 4. Conclusion based on the math
    if penalty_val > 15:
        sentences.append(f"Combined, these structural factors result in an estimated **${penalty_val:.2f} penalty** compared to a fair, open market.")
    elif penalty_val < -15:
        sentences.append(f"Overall, the structural market dynamics are working in your favor, saving you roughly **${abs(penalty_val):.2f}** compared to national baselines.")

    return " ".join(sentences)

model, metadata = load_model()
df = load_data()
cpi_df = load_cpi_data()

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
    
    # --- Create Counterfactual "Fair Market" Features ---
    fair_features = input_features.copy()
    
    # 1. Break monopolies: Demote monopoly routes to highly competitive
    if 'large_ms' in fair_features.columns:
        fair_features['large_ms'] = fair_features['large_ms'].replace(
            ['Moderately_Concentrated', 'Monopoly_Route'], 'Highly_Competitive'
        )
        
    # 2. Remove Fortress Hub Premiums: Set premium to neutral
    if 'TotalPerPrem_city1' in fair_features.columns:
        fair_features['TotalPerPrem_city1'] = fair_features['TotalPerPrem_city1'].replace('High_Premium_Hub', 'Neutral/Slight_Premium')
    if 'TotalPerPrem_city2' in fair_features.columns:
        fair_features['TotalPerPrem_city2'] = fair_features['TotalPerPrem_city2'].replace('High_Premium_Hub', 'Neutral/Slight_Premium')
        
    # 3. Ensure LCC Presence: Eliminate LCC deficiencies and monopolies (bringing them to healthy competition)
    if 'TotalPerLFMkts_city1' in fair_features.columns:
        fair_features['TotalPerLFMkts_city1'] = fair_features['TotalPerLFMkts_city1'].replace(
            ['LCC_Deficient', 'LCC_Monopoly'], 'Healthy_Competition'
        )
    if 'TotalPerLFMkts_city2' in fair_features.columns:
        fair_features['TotalPerLFMkts_city2'] = fair_features['TotalPerLFMkts_city2'].replace(
            ['LCC_Deficient', 'LCC_Monopoly'], 'Healthy_Competition'
        )

    # Format categories for XGBoost/LGBM prediction
    for col, categories in metadata["categories"].items():
        input_features[col] = pd.Categorical(input_features[col], categories=categories)
        fair_features[col] = pd.Categorical(fair_features[col], categories=categories)
        
    # Make Predictions
    predicted_expected_fare = model.predict(input_features)[0] # Based on current, actual market stats
    predicted_fair_fare = model.predict(fair_features)[0]      # Based on competitive benchmark stats
    actual_fare = lookup_row['fare_real'].values[0]            # Historical reality
    
    # --- Convert Real to Nominal Dollars ---
    actual_fare_nominal = real_to_nominal_simple(
        real_value=actual_fare, 
        year=selected_year, 
        quarter=selected_quarter, 
        cpi_df=cpi_df
    )
    
    predicted_expected_fare_nominal = real_to_nominal_simple(
        real_value=predicted_expected_fare, 
        year=selected_year, 
        quarter=selected_quarter, 
        cpi_df=cpi_df
    )
    
    predicted_fair_fare_nominal = real_to_nominal_simple(
        real_value=predicted_fair_fare, 
        year=selected_year, 
        quarter=selected_quarter, 
        cpi_df=cpi_df
    )
    
    # Calculate the differences
    total_consumer_premium = actual_fare_nominal - predicted_fair_fare_nominal
    competition_penalty = predicted_expected_fare_nominal - predicted_fair_fare_nominal
    
    # Determine the label based on the gap between reality and a fair market
    if total_consumer_premium > 15: 
        deal_status = "🚨 Overpriced (Monopoly Premium)"
    elif total_consumer_premium < -15:
        deal_status = "🎉 Underpriced (Consumer Deal)"
    else:
        deal_status = "⚖️ Fairly Priced"

    # --- THE UI VISUALIZATION ---
    st.markdown(f"### Route Status: {deal_status}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Actual Average Fare", 
                  value=f"${actual_fare_nominal:.2f}",
                  help="The actual historical price consumers paid.")
        
    with col2:
        st.metric(label="Model Expected Fare", 
                  value=f"${predicted_expected_fare_nominal:.2f}",
                  help="What the model predicts the fare SHOULD be given the CURRENT competition levels (monopolies, hubs).")
        
    with col3:
        st.metric(label="Fair Market Benchmark", 
                  value=f"${predicted_fair_fare_nominal:.2f}",
                  help="What the model predicts the fare would be IF the route had healthy competition and no hub premiums.")
    
    # --- NEW FEATURE: ACTIONABLE INSIGHTS (Best Model/Insights Rubric) ---
    dest_state = lookup_row['state_2'].iloc[0] if 'state_2' in lookup_row.columns else None
    
    if dest_state:
        # Find flights to the same state in the same quarter/year
        alt_routes = df[(df['city_1'] == selected_origin) & 
                        (df['state_2'] == dest_state) & 
                        (df['Year'] == selected_year) & 
                        (df['Quarter'] == selected_quarter) & 
                        (df['city_2'] != selected_dest)].copy()
        
        if not alt_routes.empty:
            alt_routes_sorted = alt_routes.sort_values('fare_real')
            cheapest_alt = alt_routes_sorted.iloc[0]
            
            cheapest_alt_nominal = real_to_nominal_simple(
                real_value=cheapest_alt['fare_real'], 
                year=selected_year, 
                quarter=selected_quarter, 
                cpi_df=cpi_df
            )
            
            savings = actual_fare_nominal - cheapest_alt_nominal
            
            if savings > 10:
                st.success(f"💡 **Traveler Hack (Secondary Airport):** Consider flying into **{cheapest_alt['city_2']}** instead. The average fare is **${cheapest_alt_nominal:.2f}**, saving you roughly **${savings:.2f}** while keeping you in the same destination state ({dest_state}).")
            elif savings < 0:
                st.info(f"💡 **Smart Choice:** You are already looking at the cheapest major destination in {dest_state} for this origin! The next best alternative is **{cheapest_alt['city_2']}** at **${cheapest_alt_nominal:.2f}**.")

    # Generate and display the dynamic story
    st.markdown("### 📖 Route Story: The \"Why\" Behind the Price")
    story_paragraph = generate_route_narrative(input_features, competition_penalty)
    st.info(story_paragraph)
    
    # --- NEW FEATURE: COMPETITION TREND CHART (Best Visualizations Rubric) ---
    st.markdown("---")
    st.markdown("### 📈 Route History: Fare vs. Low-Cost Carrier Presence")
    st.write("Does competition actually lower prices? This chart tracks the route's historical fares against the market share of budget airlines (LCCs) over time.")
    
    # Prepare historical data and sort chronologically
    route_history = route_df.sort_values(['Year', 'Quarter']).copy()
    route_history['Period'] = route_history['Year'].astype(str) + " Q" + route_history['Quarter'].astype(str)
    
    if len(route_history) > 1 and 'lf_ms' in route_history.columns:
        fig_trend, ax1 = plt.subplots(figsize=(10, 4))
        ax2 = ax1.twinx() # Create a twin y-axis sharing the x-axis
        
        # Plot lines
        ax1.plot(route_history['Period'], route_history['fare_real'], color='#1f77b4', marker='o', linewidth=2, label='Average Fare (Real $)')
        ax2.plot(route_history['Period'], route_history['lf_ms'] * 100, color='#2ca02c', marker='s', linestyle='--', linewidth=2, label='Budget Airline Route Share (%)')
        
        # Formatting
        ax1.set_ylabel('Average Fare (Real $)', color='#1f77b4', fontweight='bold')
        ax2.set_ylabel('Budget Carrier Route Share (%)', color='#2ca02c', fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='#1f77b4')
        ax2.tick_params(axis='y', labelcolor='#2ca02c')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        # Combined Legend
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
        
        fig_trend.tight_layout()
        st.pyplot(fig_trend)
    else:
        st.write("*Not enough historical data points to plot a trend for this specific route.*")

    st.markdown("---")
    st.markdown("### The Mathematical Breakdown")
    st.write(f"The waterfall chart below verifies the story above. It explains the **Model Expected Fare** (${predicted_expected_fare_nominal:.2f}) and shows how the route's *actual* current structural features push the expected fare up or down.")
    st.write("Key anti-competitive features **highlighted in red or blue text** in this chart include:")
    st.write("- **Dominant Carrier Market Share**")
    st.write("- **Origin and Destination Hub Premiums**")
    st.write("- **Origin and Destination LCC Penetration**")
    
    # SHAP explainer runs on the ACTUAL input features so it ties back to the Model Expected Fare
    explainer = shap.TreeExplainer(model)
    shap_explanation = explainer(input_features)
    
    # Map Features and Rename for UI ---
    
    # Get the explanation for the single row we just predicted
    exp = shap_explanation[0] 
    
    # Define the features that make up the "Anti-Competitive" group
    anti_comp_cols = [
        'large_ms', 
        'TotalPerPrem_city1', 'TotalPerPrem_city2', 
        'TotalPerLFMkts_city1', 'TotalPerLFMkts_city2'
    ]
    
    # Map ALL technical columns to plain English
    friendly_names = {
        'large_ms': 'Dominant Carrier Market Share',
        'TotalPerPrem_city1': 'Origin Hub Premium', 
        'TotalPerPrem_city2': 'Destination Hub Premium',
        'TotalPerLFMkts_city1': 'Origin LCC Penetration', 
        'TotalPerLFMkts_city2': 'Destination LCC Penetration',
        'nsmiles': 'Flight Distance',
        'passengers': 'Daily Passenger Demand',
        'lf_ms': 'Budget Airline Route Share',
        'carrier_low': 'Lowest Fare Carrier',
        'TotalFaredTotal': 'Total Market Traffic',
        'Quarter': 'Seasonality (Quarter)',
        'TotalFaredPax_city2': 'Total passengers through destination',
        'TotalFaredPax_city1': 'Total passengers through origin',
        'median_income_2': 'Median income in state of destination'
    }
    
    anti_comp_friendly_names = [friendly_names[col] for col in anti_comp_cols]
    
    # Translate the feature names in the SHAP explanation
    new_names = []
    clean_data = []
    
    for i, col_name in enumerate(exp.feature_names):
        # 1. Map to plain English name
        new_names.append(friendly_names.get(col_name, col_name))
        
        # 2. Fetch the actual raw string/numeric value from our dataframe
        # (This bypasses the issue where LightGBM/SHAP replaces string labels with NaN)
        if col_name in input_features.columns:
            val = input_features[col_name].iloc[0]
            if isinstance(val, (float, int)) and not pd.isna(val):
                val = round(val, 2)
            clean_data.append(val)
        else:
            clean_data.append(exp.data[i])
    
    # Rebuild the Explanation object with our clean names AND clean data
    new_exp = copy.deepcopy(exp)
    new_exp.feature_names = new_names
    new_exp.data = np.array(clean_data, dtype=object) # dtype=object prevents numpy string conversion errors
    
    # Create a dictionary to easily look up the SHAP value for a given feature name
    val_lookup = dict(zip(new_exp.feature_names, new_exp.values))
    
    # Plot the clean, presentation-ready chart
    # Made figure slightly taller to accommodate unbundled features
    fig, ax = plt.subplots(figsize=(8, 7)) 
    
    # Increase max_display so our unbundled features don't get hidden in "Other"
    shap.plots.waterfall(new_exp, show=False, max_display=15)
    
    # Highlight the anti-competitive features on the y-axis dynamically based on their impact
    for tick in ax.get_yticklabels():
        tick_text = tick.get_text()
        for ac_name in anti_comp_friendly_names:
            if ac_name in tick_text:
                shap_val = val_lookup.get(ac_name, 0)
                # Only color it RED if it is actively driving the price UP (a penalty)
                if shap_val > 0:
                    tick.set_color("#d62728") # Red
                    tick.set_fontweight("bold")
                # If the feature is driving the price DOWN (a consumer benefit), color it BLUE
                elif shap_val < 0:
                    tick.set_color("#008bfb") # SHAP standard blue
                    tick.set_fontweight("bold")
    
    # Add a clear title explaining exactly what the chart represents
    plt.title("Breakdown of Model Expected Fare\n(Values shown in Real/Inflation-Adjusted Dollars)", pad=20, fontsize=12, fontweight="bold")
    
    plt.tight_layout()
    st.pyplot(fig)