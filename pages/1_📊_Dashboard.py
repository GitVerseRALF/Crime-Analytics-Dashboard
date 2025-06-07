import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score, average_precision_score, precision_recall_curve
)
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import os

# At the top of your dashboard
if 'data_timestamp' not in st.session_state:
    st.session_state.data_timestamp = 0

# Page config
st.set_page_config(page_title="Crime Dashboard", page_icon=":bar_chart:", layout="wide")

# ---- DATA LOADING FUNCTION ----
@st.cache_data
def load_preset_data():
    # Preset data - you can modify this according to your needs
    data = {
        'Region': ['Jakarta', 'Surabaya', 'Bandung', 'Medan', 'Semarang'] * 3,
        'Crime_Type': ['Theft', 'Assault', 'Fraud'] * 5,
        'Year': [2020, 2021, 2022] * 5,
        'Incident_Count': np.random.randint(100, 1000, 15)
    }
    return pd.DataFrame(data)

@st.cache_data(ttl=None, show_spinner=True)
def get_user_data():
    # Force function to rerun when timestamp changes
    _ = st.session_state.data_timestamp
    if os.path.exists('user_uploaded_data.csv'):
        return pd.read_csv('user_uploaded_data.csv')
    return None

# Load either user data or preset data
df = get_user_data() if get_user_data() is not None else load_preset_data()

# Initialize AI analyzer
class CrimeAnalyzer:
    def __init__(self, df):
            self.df = df
            self.anomaly_detector = None
            self.prediction_model = None

    def perform_precision_analysis(self, target_crime_type=None):
            """Perform precision-recall and classification analysis on crime predictions."""
            # Prepare the data
            data = self.df.copy()

            # Filter by crime type if provided
            if target_crime_type:
                data = data[data['Crime_Type'] == target_crime_type]

            # Create binary classification target (high vs low crime rate)
            median_incidents = data['Incident_Count'].median()
            data['High_Crime'] = (data['Incident_Count'] > median_incidents).astype(int)

            # Encode categorical features
            le_region = LabelEncoder()
            data['Region_Encoded'] = le_region.fit_transform(data['Region'])
            le_crime_type = LabelEncoder()
            data['Crime_Type_Encoded'] = le_crime_type.fit_transform(data['Crime_Type'])

            # Prepare features and target
            X = data[['Region_Encoded', 'Crime_Type_Encoded', 'Year']]
            y = data['High_Crime']

            # Split the data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)
            y_scores = model.predict_proba(X_test)[:, 1]

            # Compute classification metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=1)
            recall = recall_score(y_test, y_pred, zero_division=1)
            f1 = f1_score(y_test, y_pred)

            # Calculate precision-recall curve
            precision_curve, recall_curve, thresholds = precision_recall_curve(y_test, y_scores)
            avg_precision = average_precision_score(y_test, y_scores)

            # Calculate metrics for each region
            region_metrics = {}
            for region in data['Region'].unique():
                region_data = data[data['Region'] == region]
                if len(region_data) > 1:
                    X_region = region_data[['Region_Encoded', 'Crime_Type_Encoded', 'Year']]
                    y_region = region_data['High_Crime']
                    y_pred_region = model.predict(X_region)
                    region_metrics[region] = {
                        "accuracy": accuracy_score(y_region, y_pred_region),
                        "precision": precision_score(y_region, y_pred_region, zero_division=1),
                        "recall": recall_score(y_region, y_pred_region, zero_division=1),
                        "f1_score": f1_score(y_region, y_pred_region)
                    }

            # Return overall metrics and per-region metrics
            return {
                "overall": {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "average_precision": avg_precision
                },
                "region_metrics": region_metrics,
                "precision_recall_curve": {
                    "precision": precision_curve,
                    "recall": recall_curve,
                    "thresholds": thresholds
                }
            }
        
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

    def perform_clustering(self, n_clusters=3):
        """Perform K-means clustering on crime data"""
        # Prepare data for clustering
        data = self.df.pivot_table(
            index='Region',
            columns='Crime_Type',
            values='Incident_Count',
            aggfunc='sum'
        ).fillna(0)
        
        # Standardize the features
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(data_scaled)
        
        # Add cluster labels to the data
        data['Cluster'] = clusters
        
        # Calculate cluster characteristics
        cluster_stats = []
        for i in range(n_clusters):
            cluster_regions = data[data['Cluster'] == i].index.tolist()
            total_crimes = self.df[self.df['Region'].isin(cluster_regions)]['Incident_Count'].sum()
            avg_crimes = total_crimes / len(cluster_regions)
            dominant_crime = data[data['Cluster'] == i].mean().drop('Cluster').idxmax()
            
            cluster_stats.append({
                'Cluster': i,
                'Regions': cluster_regions,
                'Total_Crimes': total_crimes,
                'Avg_Crimes': avg_crimes,
                'Dominant_Crime': dominant_crime
            })
        
        return data, cluster_stats
        
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

# Initialize the analyzer with the loaded data
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

# Add navigation button to upload page
st.sidebar.markdown("---")
if st.sidebar.button("📤 Upload Custom Data"):
    st.switch_page("pages/2_📤_Upload_Data.py")

# Main Dashboard
st.title("🔍 Indonesian Crime Dashboard")
st.markdown("##")

# Add AI Analysis Section
st.sidebar.markdown("---")
st.sidebar.header("AI Analysis Options")
ai_analysis = st.sidebar.selectbox(
    "Choose Analysis Type:",
    ["Overview", "Anomaly Detection", "Cluster Analysis", "Pattern Analysis", "Precision Analysis"]
)

if ai_analysis == "Overview":
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

elif ai_analysis == "Precision Analysis":
    st.subheader("🔍 Precision Analysis")

    target_crime = st.selectbox("Select Crime Type for Analysis (optional):", ["All"] + sorted(df['Crime_Type'].unique()))
    target_crime_type = target_crime if target_crime != "All" else None

    results = analyzer.perform_precision_analysis(target_crime_type=target_crime_type)

    # Display overall metrics
    st.write("### Overall Metrics:")
    st.write(f"**Accuracy:** {results['overall']['accuracy']:.2f}")
    st.write(f"**Precision:** {results['overall']['precision']:.2f}")
    st.write(f"**Recall:** {results['overall']['recall']:.2f}")
    st.write(f"**F1 Score:** {results['overall']['f1_score']:.2f}")
    st.write(f"**Average Precision (PR AUC):** {results['overall']['average_precision']:.2f}")

    # Display per-region metrics
    st.write("### Per-Region Metrics:")
    for region, metrics in results['region_metrics'].items():
        st.write(f"**{region}:**")
        st.write(f"- Accuracy: {metrics['accuracy']:.2f}")
        st.write(f"- Precision: {metrics['precision']:.2f}")
        st.write(f"- Recall: {metrics['recall']:.2f}")
        st.write(f"- F1 Score: {metrics['f1_score']:.2f}")

    # Optional: Visualize precision-recall curve
    st.write("### Precision-Recall Curve:")
    pr_fig = px.area(
        x=results['precision_recall_curve']['recall'],
        y=results['precision_recall_curve']['precision'],
        labels={"x": "Recall", "y": "Precision"},
        title="Precision-Recall Curve",
        template="plotly_white"
    )
    st.plotly_chart(pr_fig, use_container_width=True)

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

elif ai_analysis == "Cluster Analysis":
    st.subheader("🔍 Region Clustering Analysis")
    
    n_clusters = st.slider("Number of Clusters", min_value=2, max_value=5, value=3)
    
    clustered_data, cluster_stats = analyzer.perform_clustering(n_clusters)
    
    # Display cluster insights
    st.write("### Cluster Insights")
    for stat in cluster_stats:
        with st.expander(f"Cluster {stat['Cluster']} Analysis"):
            st.write(f"**Regions in this cluster:** {', '.join(stat['Regions'])}")
            st.write(f"**Total crimes:** {stat['Total_Crimes']:,.0f}")
            st.write(f"**Average crimes per region:** {stat['Avg_Crimes']:,.0f}")
            st.write(f"**Dominant crime type:** {stat['Dominant_Crime']}")
    
    # Visualize clusters
    fig = px.scatter(
        clustered_data.reset_index(),
        x=clustered_data.columns[0],  # First crime type
        y=clustered_data.columns[1],  # Second crime type
        color='Cluster',
        hover_data=['Region'],
        title="Region Clusters based on Crime Patterns",
        labels={'x': clustered_data.columns[0], 'y': clustered_data.columns[1]}
    )
    st.plotly_chart(fig, use_container_width=True)
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



def perform_precision_analysis(self, target_crime_type=None):
    """Perform precision-recall analysis on crime predictions"""
    # Prepare the data
    data = self.df.copy()
    
    # Create binary classification target (high vs low crime rate)
    median_incidents = data['Incident_Count'].median()
    data['High_Crime'] = (data['Incident_Count'] > median_incidents).astype(int)
    
    # Prepare features
    le = LabelEncoder()
    data['Region_Encoded'] = le.fit_transform(data['Region'])
    data['Crime_Type_Encoded'] = le.fit_transform(data['Crime_Type'])
    
    X = data[['Region_Encoded', 'Crime_Type_Encoded', 'Year']]
    y = data['High_Crime']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Get prediction probabilities
    y_scores = model.predict_proba(X_test)[:, 1]
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
    avg_precision = average_precision_score(y_test, y_scores)
    
    # Calculate prediction metrics for each region
    region_metrics = {}
    for region in data['Region'].unique():
        region_data = data[data['Region'] == region]
        if len(region_data) > 0:
            X_region = region_data[['Region_Encoded', 'Crime_Type_Encoded', 'Year']]
            y_region = region_data['High_Crime']
            y_pred_proba = model.predict_proba(X_region)[:, 1]
            avg_precision_region = average_precision_score(y_region, y_pred_proba)
            region_metrics[region] = avg_precision_region
    
    return precision, recall, thresholds, avg_precision, region_metrics
