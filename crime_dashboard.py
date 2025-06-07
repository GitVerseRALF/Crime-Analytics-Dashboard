import pandas as pd  # pip install pandas openpyxl
import plotly.express as px  # pip install plotly-express
import streamlit as st  # pip install streamlit

# emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="Crime Dashboard", page_icon=":bar_chart:", layout="wide")

# ---- READ EXCEL ----
@st.cache_data
def get_data_from_excel():
    # Load data from both files
    file_path_1 = 'B:\S3\Software Engineering\SE Project\Data\crime_data_indonesia.xlsx'
    file_path_2 = 'B:\S3\Software Engineering\SE Project\Data\indonesia_crime_data_by_region.xlsx'

    # Load sheets
    data_file_1 = pd.read_excel(file_path_1, sheet_name="Sheet1")
    data_file_2 = pd.read_excel(file_path_2, sheet_name="Sheet1")

    # Standardize column names for merging
    data_file_1.columns = ["Region", "Crime Type", "Year", "Incident Count"]
    data_file_2.columns = ["Region", "Crime Type", "Year", "Incident Count"]

    # Combine both sheets into a single dataframe
    df = pd.concat([data_file_1, data_file_2], ignore_index=True)

    return df

df = get_data_from_excel()

# ---- SIDEBAR ----
st.sidebar.header("Please Filter Here:")
city = st.sidebar.multiselect(
    "Select the Region:",
    options=df["Region"].unique(),
    default=df["Region"].unique()
)

crime_type = st.sidebar.multiselect(
    "Select the Crime Type:",
    options=df["Crime Type"].unique(),
    default=df["Crime Type"].unique(),
)

year = st.sidebar.multiselect(
    "Select the Year:",
    options=df["Year"].unique(),
    default=df["Year"].unique()
)

# Filter the dataframe based on selections
df_selection = df.query(
    "Region == @city & `Crime Type` == @crime_type & Year == @year"
)

# Check if the dataframe is empty:
if df_selection.empty:
    st.warning("No data available based on the current filter settings!")
    st.stop()  # This will halt the app from further execution.

# ---- MAINPAGE ----
st.title(":bar_chart: Crime Dashboard")
st.markdown("##")

# TOP KPI's
total_incidents = int(df_selection["Incident Count"].sum())
average_incidents = round(df_selection["Incident Count"].mean(), 1)

left_column, right_column = st.columns(2)
with left_column:
    st.subheader("Total Crimes:")
    st.subheader(f"{total_incidents:,}")
with right_column:
    st.subheader("Average Crimes per Type:")
    st.subheader(f"{average_incidents}")

st.markdown("""---""")

# CRIMES BY TYPE [BAR CHART]
crimes_by_type = df_selection.groupby(by=["Crime Type"])[["Incident Count"]].sum().sort_values(by="Incident Count")
fig_crime_type = px.bar(
    crimes_by_type,
    x="Incident Count",
    y=crimes_by_type.index,
    orientation="h",
    title="<b>Total Crimes by Type</b>",
    color_discrete_sequence=["#0083B8"] * len(crimes_by_type),
    template="plotly_white",
)
fig_crime_type.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=(dict(showgrid=False))
)

# CRIMES BY REGION [BAR CHART]
crimes_by_region = df_selection.groupby(by=["Region"])[["Incident Count"]].sum()
fig_crime_region = px.bar(
    crimes_by_region,
    x=crimes_by_region.index,
    y="Incident Count",
    title="<b>Total Crimes by Region</b>",
    color_discrete_sequence=["#0083B8"] * len(crimes_by_region),
    template="plotly_white",
)
fig_crime_region.update_layout(
    xaxis=dict(tickmode="linear"),
    plot_bgcolor="rgba(0,0,0,0)",
    yaxis=(dict(showgrid=False)),
)

left_column, right_column = st.columns(2)
left_column.plotly_chart(fig_crime_region, use_container_width=True)
right_column.plotly_chart(fig_crime_type, use_container_width=True)

# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
