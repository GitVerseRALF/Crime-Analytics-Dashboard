# Create a simple home page that redirects to the dashboard
# Home.py
import streamlit as st

st.set_page_config(page_title="Crime Dashboard Home", page_icon="🏠", layout="wide")

st.title("🏠 Welcome to the Crime Dashboard")
st.write("""
This dashboard allows you to analyze crime statistics across different regions.
You can use our preset data or upload your own custom data.
""")

if st.button("Go to Dashboard"):
    st.switch_page("pages/1_📊_Dashboard.py")