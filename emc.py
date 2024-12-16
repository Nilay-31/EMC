pip install openpyxl

import streamlit as st  # For deployment
import pandas as pd  # For data handling

def main():
    st.title("Email Marketing Campaign Success Predictor")
    st.write("Upload a file to analyze.")

    # File uploader for .xlsx and .csv files
    uploaded_file = st.file_uploader("Upload Dataset", type=["xlsx", "csv"])
    
    if uploaded_file is not None:
        # Read the uploaded file based on its extension
        try:
            if uploaded_file.name.endswith('.xlsx'):
                data = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                st.error("Unsupported file format!")
                return
            
            # Display the uploaded data
            st.write("Uploaded Dataset:")
            st.write(data.head())
            
        except Exception as e:
            st.error(f"Error reading the file: {e}")
            return

if __name__ == '__main__':
    main()
