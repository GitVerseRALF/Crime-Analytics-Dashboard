import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Upload Data", page_icon="📤", layout="wide")

st.title("📤 Upload Custom Crime Data")

# Display template screenshot
st.header("1. Template Overview")
st.image("SS_Template.png", caption="Excel Template Structure", use_column_width=True)

# Upload section
st.header("2. Upload Your Data")
st.write("Upload 1-3 Excel files with crime data:")

# Create three file uploaders
uploaded_files = [
    st.file_uploader(f"Upload File {i+1}", type=["xlsx", "xls"], key=f"file_{i}") 
    for i in range(3)
]

# Process and validate files
def process_files(files):
    # Filter out None values (unselected files)
    valid_files = [f for f in files if f is not None]
    
    if not valid_files:
        st.error("❌ Please upload at least one file.")
        return None
    
    # List to store processed dataframes
    dataframes = []
    
    for file in valid_files:
        try:
            # Read the Excel file
            df = pd.read_excel(file)
            
            # Check required columns
            required_columns = ['Region', 'Crime Type', 'Year', 'Incident Count']
            if all(col in df.columns for col in required_columns):
                dataframes.append(df)
            else:
                st.error(f"❌ File {file.name} does not match the required template.")
                return None
        
        except Exception as e:
            st.error(f"❌ Error processing {file.name}: {str(e)}")
            return None
    
    return dataframes

# Upload button
if st.button("Upload and Process Files"):
    processed_data = process_files(uploaded_files)
    
    if processed_data:
        # Combine dataframes if multiple files are uploaded
        if len(processed_data) > 1:
            combined_df = pd.concat(processed_data, ignore_index=True)
        else:
            combined_df = processed_data[0]
        
        # Rename columns to match the expected format
        combined_df = combined_df.rename(columns={
            'Crime Type': 'Crime_Type', 
            'Incident Count': 'Incident_Count'
        })
        
        # Store in a persistent file
        combined_df.to_csv('user_uploaded_data.csv', index=False)
        
        st.success(f"✅ Successfully uploaded and processed {len(processed_data)} file(s)!")
        
        # Clear cache before switching pages
        st.cache_data.clear()
        
        # Switch to dashboard
        st.switch_page("pages/1_📊_Dashboard.py")

# Instructions
st.markdown("---")
st.header("Instructions")
st.write("""
### Data Upload Guidelines
1. Review the template structure in the screenshot above
2. Prepare your data following these guidelines:
   - Region: Name of the region/city
   - Crime Type: Type of crime
   - Year: Year of incidents (numeric)
   - Incident Count: Number of incidents (numeric)
3. You can upload 1-3 files:
   - Files must follow the same template structure
   - If multiple files are uploaded, data will be combined
4. Click "Upload and Process Files" to proceed
5. Return to the dashboard to view your data
""")

# Additional guidance
st.info("""
💡 Tips:
- You don't need to upload all three files
- Each file should have the same column structure
- Multiple files will be merged into a single dataset
""")