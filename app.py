import streamlit as st  # For deployment
def main():
    st.title("Email Marketing Campaign Success Predictor")
    st.write("Upload a file to analyze.")
    
    uploaded_file = st.file_uploader("Email_Marketing_Campaign_Dataset_Rounded.xlsx", type=["xlsx", "csv"])
    
    if uploaded_file is not None:
        data = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)
        st.write("Uploaded Dataset:")
        st.write(data.head())
        
        if st.button("Predict"):
            # Load the saved model
            loaded_model = joblib.load("email_campaign_model.pkl")
            
            # Make predictions
            predictions = loaded_model.predict(data)
            data['Predicted_Open'] = predictions
            st.write("Predicted Results:")
            st.write(data)

if __name__ == '__main__':
    main()
