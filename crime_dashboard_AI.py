import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime, timedelta

# Page config
st.set_page_config(page_title="AI-Enhanced Crime Dashboard", page_icon=":bar_chart:", layout="wide")

# Data processing function
@st.cache_data
def get_crime_data():
    # Same data loading as before
    df1 = pd.DataFrame({
        'Region': ['Sumatera U', 'Sumatera S', 'Maluku Uta', 'Gorontalo', 'Metro Jaya', 
                  'Sumatera U', 'Jawa Timur', 'Sumatera U', 'Metro Jaya', 'Kalimantan'],
        'Crime_Type': ['Property Cr', 'Property Cr', 'Property Cr', 'Property Cr', 'Drug-relate',
                      'Drug-relate', 'Property Cr', 'Fraud, Emb', 'Fraud, Emb', 'Fraud, Emb'],
        'Year': [2020, 2020, 2020, 2020, 2020, 2020, 2022, 2022, 2022, 2022],
        'Incident_Count': [780, 563, 14, 4, 5981, 5932, 9917, 5376, 9729, 84]
    })

    df2 = pd.DataFrame({
        'Region': ['Jawa Timur', 'Sumatera U', 'Metro Jaya', 'Maluku Uta', 'Kalimantan',
                  'Sulawesi Ba', 'Sulawesi Ut', 'Papua Bara', 'Sulawesi Se'],
        'Year': [2020, 2021, 2022, 2020, 2021, 2022, 2020, 2021, 2022],
        'Total_Incidents': [51905, 43555, 32534, 1220, 1280, 2027, 364, 353, 314]
    })
    
    return df1, df2

# AI Analysis Functions
class CrimeAnalyzer:
    def __init__(self, df1, df2):
        self.df1 = df1
        self.df2 = df2
        self.anomaly_detector = None
        self.prediction_model = None
        
    def detect_anomalies(self, region=None):
        """Detect unusual crime patterns using Isolation Forest"""
        data = self.df1.copy()
        if region:
            data = data[data['Region'] == region]
            
        X = data[['Incident_Count', 'Year']].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        anomalies = self.anomaly_detector.fit_predict(X_scaled)
        
        return data[anomalies == -1]
    
    def predict_future_trends(self, region, crime_type):
        """Predict future crime trends using Random Forest"""
        data = self.df1[
            (self.df1['Region'] == region) & 
            (self.df1['Crime_Type'] == crime_type)
        ].copy()
        
        if len(data) < 2:
            return None, None
            
        X = data[['Year']].values
        y = data['Incident_Count'].values
        
        self.prediction_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.prediction_model.fit(X, y)
        
        # Predict next 2 years
        future_years = np.array([[year] for year in range(data['Year'].max() + 1, data['Year'].max() + 3)])
        predictions = self.prediction_model.predict(future_years)
        
        return future_years.flatten(), predictions
    
    def get_crime_patterns(self):
        """Analyze temporal and regional crime patterns"""
        patterns = []
        
        # Analyze year-over-year changes
        yearly_totals = self.df1.groupby('Year')['Incident_Count'].sum()
        yoy_change = yearly_totals.pct_change() * 100
        
        if not yoy_change.empty:
            for year, change in yoy_change.items():
                if not pd.isna(change):
                    direction = "increased" if change > 0 else "decreased"
                    patterns.append(f"Crime incidents {direction} by {abs(change):.1f}% in {year}")
        
        # Analyze regional hotspots
        region_totals = self.df1.groupby('Region')['Incident_Count'].sum()
        top_regions = region_totals.nlargest(3)
        patterns.append("\nTop 3 regions by crime incidents:")
        for region, count in top_regions.items():
            patterns.append(f"- {region}: {count:,.0f} incidents")
            
        return patterns

# Load data
df1, df2 = get_crime_data()

# Initialize AI analyzer
analyzer = CrimeAnalyzer(df1, df2)

# Sidebar filters (same as before)
st.sidebar.header("Filter Options:")

regions = sorted(list(set(df1['Region'].unique()) | set(df2['Region'].unique())))
selected_regions = st.sidebar.multiselect(
    "Select Regions:",
    options=regions,
    default=regions[:5]
)

crime_types = sorted(df1['Crime_Type'].unique())
selected_crime_types = st.sidebar.multiselect(
    "Select Crime Types:",
    options=crime_types,
    default=crime_types
)

years = sorted(list(set(df1['Year'].unique()) | set(df2['Year'].unique())))
selected_years = st.sidebar.multiselect(
    "Select Years:",
    options=years,
    default=years
)

# Filter DataFrames
df1_filtered = df1.query(
    "Region in @selected_regions & Crime_Type in @selected_crime_types & Year in @selected_years"
)

df2_filtered = df2.query(
    "Region in @selected_regions & Year in @selected_years"
)

# Main Dashboard
st.title("🔍 AI-Enhanced Crime Dashboard")
st.markdown("##")

# Add AI Analysis Section
st.sidebar.markdown("---")
st.sidebar.header("AI Analysis Options")
ai_analysis = st.sidebar.selectbox(
    "Choose Analysis Type:",
    ["Overview", "Anomaly Detection", "Trend Prediction", "Pattern Analysis"]
)

if ai_analysis == "Overview":
    # Original dashboard content
    total_incidents = int(df1_filtered['Incident_Count'].sum())
    avg_incidents_per_region = round(df1_filtered.groupby('Region')['Incident_Count'].mean().mean(), 1)
    max_crime_region = df1_filtered.groupby('Region')['Incident_Count'].sum().idxmax()

    left_column, middle_column, right_column = st.columns(3)
    with left_column:
        st.subheader("Total Reported Incidents:")
        st.subheader(f"{total_incidents:,}")
    with middle_column:
        st.subheader("Avg. Incidents per Region:")
        st.subheader(f"{avg_incidents_per_region:,.1f}")
    with right_column:
        st.subheader("Highest Crime Region:")
        st.subheader(f"{max_crime_region}")

elif ai_analysis == "Anomaly Detection":
    st.subheader("🔍 Anomaly Detection")
    selected_region = st.selectbox("Select Region for Analysis:", regions)
    
    anomalies = analyzer.detect_anomalies(selected_region)
    
    if not anomalies.empty:
        st.warning("Unusual Crime Patterns Detected:")
        for _, row in anomalies.iterrows():
            st.write(f"⚠️ {row['Crime_Type']} in {row['Year']}: {row['Incident_Count']} incidents")
            
        # Visualize anomalies
        fig = px.scatter(
            df1[df1['Region'] == selected_region],
            x='Year',
            y='Incident_Count',
            color='Crime_Type',
            title=f"Crime Patterns in {selected_region} (Anomalies Highlighted)"
        )
        
        # Add anomaly points
        fig.add_scatter(
            x=anomalies['Year'],
            y=anomalies['Incident_Count'],
            mode='markers',
            marker=dict(size=15, symbol='x', color='red'),
            name='Anomalies'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("No significant anomalies detected in the selected region.")

elif ai_analysis == "Trend Prediction":
    st.subheader("📈 Crime Trend Prediction")
    
    pred_region = st.selectbox("Select Region:", regions)
    pred_crime_type = st.selectbox("Select Crime Type:", crime_types)
    
    future_years, predictions = analyzer.predict_future_trends(pred_region, pred_crime_type)
    
    if future_years is not None and predictions is not None:
        st.write("### Predicted Crime Incidents")
        for year, pred in zip(future_years, predictions):
            st.write(f"📅 {int(year)}: {pred:,.0f} predicted incidents")
        
        # Visualization
        historical_data = df1[
            (df1['Region'] == pred_region) & 
            (df1['Crime_Type'] == pred_crime_type)
        ]
        
        fig = px.line(
            x=list(historical_data['Year']) + list(future_years),
            y=list(historical_data['Incident_Count']) + list(predictions),
            title=f"Crime Trend Prediction for {pred_crime_type} in {pred_region}",
            labels={'x': 'Year', 'y': 'Incident Count'}
        )
        
        # Mark predicted values
        fig.add_scatter(
            x=future_years,
            y=predictions,
            mode='markers+lines',
            line=dict(dash='dash'),
            name='Predictions'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Insufficient data for prediction. Please select a different region or crime type.")

elif ai_analysis == "Pattern Analysis":
    st.subheader("🔍 Crime Pattern Analysis")
    patterns = analyzer.get_crime_patterns()
    
    st.write("### Key Insights")
    for pattern in patterns:
        st.write(pattern)
        
    # Add correlation heatmap
    st.write("### Crime Type Correlations")
    pivot_data = df1.pivot_table(
        index='Year',
        columns='Crime_Type',
        values='Incident_Count',
        aggfunc='sum'
    ).fillna(0)
    
    correlation = pivot_data.corr()
    
    fig = px.imshow(
        correlation,
        title="Crime Type Correlation Matrix",
        color_continuous_scale="RdBu"
    )
    st.plotly_chart(fig, use_container_width=True)

# Original visualizations
st.markdown("---")
st.subheader("📊 Basic Crime Statistics")

# Crimes by Region [Bar Chart]
crimes_by_region = df1_filtered.groupby('Region')['Incident_Count'].sum().sort_values(ascending=True)
fig_region_crimes = px.bar(
    crimes_by_region,
    x=crimes_by_region.values,
    y=crimes_by_region.index,
    orientation="h",
    title="<b>Total Crimes by Region</b>",
    color_discrete_sequence=["#0083B8"] * len(crimes_by_region),
    template="plotly_white",
)
fig_region_crimes.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(showgrid=False)
)

# Crimes by Type [Bar Chart]
crimes_by_type = df1_filtered.groupby('Crime_Type')['Incident_Count'].sum()
fig_type_crimes = px.bar(
    crimes_by_type,
    x=crimes_by_type.index,
    y=crimes_by_type.values,
    title="<b>Total Crimes by Type</b>",
    color_discrete_sequence=["#0083B8"] * len(crimes_by_type),
    template="plotly_white",
)
fig_type_crimes.update_layout(
    xaxis=dict(tickmode="linear"),
    plot_bgcolor="rgba(0,0,0,0)",
    yaxis=dict(showgrid=False)
)

# Display charts
left_column, right_column = st.columns(2)
left_column.plotly_chart(fig_region_crimes, use_container_width=True)
right_column.plotly_chart(fig_type_crimes, use_container_width=True)

# Hide Streamlit Style
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)