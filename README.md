# Crime-Analytics-Dashboard
An advanced crime analytics dashboard that combines traditional statistical analysis with AI/ML capabilities to analyze crime patterns across Indonesia. Built with Python, this tool helps law enforcement and analysts understand crime trends, detect anomalies, and make data-driven decisions.

🌟 Features
Data Management:
- Multi-file Excel (.xlsx) upload support
- Automated data standardization and combination
- Real-time data processing and visualization

AI/ML Capabilities:
- Anomaly Detection: Uses Isolation Forest algorithm to identify unusual crime patterns
- Trend Prediction: Implements Random Forest for future crime forecasting
- Pattern Analysis: Advanced temporal and regional crime pattern analysis
- Correlation Analysis: Crime type relationship visualization

Interactive Visualizations:
- Regional crime distribution maps
- Crime type analysis charts
- Time-series trend analysis
- Interactive filtering by:
  - Region
  - Crime Type
  - Year
- Correlation heatmaps

🛠️ Tech Stack:
- Streamlit: Web interface and dashboard framework
- Pandas: Data manipulation and analysis
- Plotly Express: Interactive data visualization
- Scikit-learn: Machine learning implementations
- NumPy: Numerical operations

📋 Requirements:
streamlit==1.24.0
pandas==2.0.2
plotly==5.15.0
scikit-learn==1.2.2
numpy==1.24.3
joblib==1.2.0

📊 Usage:
1. Launch the dashboard
2. Upload three .xlsx files containing crime data
3. Use sidebar filters to select:
     - Regions of interest
     - Crime types
     - Time period
4. Choose analysis type:
      - Overview
      - Anomaly Detection
      - Trend Prediction
      - Pattern Analysis
  
📝 Data Format:
Expected Excel file structure:

- Sheet name: "Sheet1"
- Columns:
  - Region
  - Crime_Type
  - Year
  - Incident_Count

🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check issues page.

