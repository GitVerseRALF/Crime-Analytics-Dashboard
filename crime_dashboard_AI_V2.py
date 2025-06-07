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
st.set_page_config(page_title="Crime Dashboard", page_icon=":bar_chart:", layout="wide")

# Data processing function
@st.cache_data
def get_crime_data():
    # DataFrame for specific crime types (translated and full names)
    df1 = pd.DataFrame({
        'Region': [
            'North Sumatra', 'South Sumatra', 'North Maluku', 'Gorontalo', 
            'Greater Jakarta', 'North Sumatra', 'East Java', 'North Sumatra', 
            'Greater Jakarta', 'Kalimantan'
        ],
        'Crime_Type': [
            'Property Crime', 'Property Crime', 'Property Crime', 'Property Crime', 
            'Drug-related', 'Drug-related', 'Property Crime', 'Fraud & Embezzlement', 
            'Fraud & Embezzlement', 'Fraud & Embezzlement'
        ],
        'Year': [2020, 2020, 2020, 2020, 2020, 2020, 2022, 2022, 2022, 2022],
        'Incident_Count': [780, 563, 14, 4, 5981, 5932, 9917, 5376, 9729, 84]
    })

    # DataFrame for total crimes
    df2 = pd.DataFrame({
        'Region': [
            'East Java', 'North Sumatra', 'Greater Jakarta', 'North Maluku',
            'Kalimantan', 'West Sulawesi', 'North Sulawesi', 'West Papua',
            'South Sulawesi'
        ],
        'Crime_Type': ['Total Crime'] * 9,  # Adding crime type column for consistency
        'Year': [2020, 2021, 2022, 2020, 2021, 2022, 2020, 2021, 2022],
        'Total_Incidents': [51905, 43555, 32534, 1220, 1280, 2027, 364, 353, 314]
    })
    
    # Rename Total_Incidents to Incident_Count for consistency
    df2 = df2.rename(columns={'Total_Incidents': 'Incident_Count'})
    
    # Combine datasets
    combined_df = pd.concat([
        df1,
        df2[['Region', 'Crime_Type', 'Year', 'Incident_Count']]
    ], ignore_index=True)
    
    return combined_df

# AI Analysis Functions
class CrimeAnalyzer:
    def __init__(self, df):
        self.df = df
        self.anomaly_detector = None
        self.prediction_model = None
        
    def detect_anomalies(self, region=None, crime_type=None):
        """Detect unusual crime patterns using Isolation Forest"""
        data = self.df.copy()
        if region:
            data = data[data['Region'] == region]
        if crime_type and crime_type != "All":
            data = data[data['Crime_Type'] == crime_type]
            
        if len(data) < 2:
            return pd.DataFrame()
            
        X = data[['Incident_Count', 'Year']].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        anomalies = self.anomaly_detector.fit_predict(X_scaled)
        
        return data[anomalies == -1]
    
    def predict_future_trends(self, region, crime_type):
        """Predict future crime trends using Random Forest"""
        data = self.df[
            (self.df['Region'] == region) & 
            (self.df['Crime_Type'] == crime_type)
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
        yearly_totals = self.df.groupby(['Year', 'Crime_Type'])['Incident_Count'].sum().reset_index()
        
        for crime_type in self.df['Crime_Type'].unique():
            crime_data = yearly_totals[yearly_totals['Crime_Type'] == crime_type]
            if len(crime_data) > 1:
                yoy_change = crime_data.set_index('Year')['Incident_Count'].pct_change() * 100
                for year, change in yoy_change.items():
                    if not pd.isna(change):
                        direction = "increased" if change > 0 else "decreased"
                        patterns.append(f"{crime_type} incidents {direction} by {abs(change):.1f}% in {year}")
        
        # Analyze regional hotspots
        patterns.append("\nTop regions by crime incidents:")
        for crime_type in self.df['Crime_Type'].unique():
            region_totals = self.df[self.df['Crime_Type'] == crime_type].groupby('Region')['Incident_Count'].sum()
            top_regions = region_totals.nlargest(3)
            patterns.append(f"\n{crime_type}:")
            for region, count in top_regions.items():
                patterns.append(f"- {region}: {count:,.0f} incidents")
            
        return patterns

# Load data
df = get_crime_data()

# Initialize AI analyzer
analyzer = CrimeAnalyzer(df)

# Sidebar filters
st.sidebar.header("Filter Options:")

regions = sorted(df['Region'].unique())
selected_regions = st.sidebar.multiselect(
    "Select Regions:",
    options=regions,
    default=regions[:5]
)

crime_types = sorted(df['Crime_Type'].unique())
selected_crime_types = st.sidebar.multiselect(
    "Select Crime Types:",
    options=crime_types,
    default=crime_types
)

years = sorted(df['Year'].unique())
selected_years = st.sidebar.multiselect(
    "Select Years:",
    options=years,
    default=years
)

# Filter DataFrame
df_filtered = df.query(
    "Region in @selected_regions & Crime_Type in @selected_crime_types & Year in @selected_years"
)

# Main Dashboard
st.title("🔍 Indonesian Crime Dashboard")
st.markdown("##")

# Add AI Analysis Section
st.sidebar.markdown("---")
st.sidebar.header("AI Analysis Options")
ai_analysis = st.sidebar.selectbox(
    "Choose Analysis Type:",
    ["Overview", "Anomaly Detection", "Trend Prediction", "Pattern Analysis"]
)

if ai_analysis == "Overview":
    # Overview statistics
    total_incidents = int(df_filtered['Incident_Count'].sum())
    avg_incidents_per_region = round(df_filtered.groupby('Region')['Incident_Count'].mean().mean(), 1)
    max_crime_region = df_filtered.groupby('Region')['Incident_Count'].sum().idxmax()

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
    
    col1, col2 = st.columns(2)
    with col1:
        selected_region = st.selectbox("Select Region for Analysis:", ["All"] + regions)
    with col2:
        selected_crime = st.selectbox("Select Crime Type for Analysis:", ["All"] + crime_types)
    
    anomalies = analyzer.detect_anomalies(
        region=selected_region if selected_region != "All" else None,
        crime_type=selected_crime if selected_crime != "All" else None
    )
    
    if not anomalies.empty:
        st.warning("Unusual Crime Patterns Detected:")
        for _, row in anomalies.iterrows():
            st.write(f"⚠️ {row['Crime_Type']} in {row['Region']} ({row['Year']}): {row['Incident_Count']:,} incidents")
            
        # Visualize anomalies
        fig = px.scatter(
            df if selected_region == "All" else df[df['Region'] == selected_region],
            x='Year',
            y='Incident_Count',
            color='Crime_Type',
            title=f"Crime Patterns {f'in {selected_region}' if selected_region != 'All' else '(All Regions)'}"
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
        st.success("No significant anomalies detected with the current selection.")

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
        historical_data = df[
            (df['Region'] == pred_region) & 
            (df['Crime_Type'] == pred_crime_type)
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
    st.write("### Crime Patterns by Region and Type")
    pivot_data = df.pivot_table(
        index='Region',
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

# Basic visualizations
st.markdown("---")
st.subheader("📊 Crime Statistics")

# Crimes by Region [Bar Chart]
crimes_by_region = df_filtered.groupby('Region')['Incident_Count'].sum().sort_values(ascending=True)
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
    xaxis=dict(showgrid=False),
    yaxis_title="Region",
    xaxis_title="Number of Incidents"
)

# Crimes by Type [Bar Chart]
crimes_by_type = df_filtered.groupby('Crime_Type')['Incident_Count'].sum()
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
    yaxis=dict(showgrid=False),
    xaxis_title="Crime Type",
    yaxis_title="Number of Incidents"
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