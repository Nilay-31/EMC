import streamlit as st
import pandas as pd
import joblib  # For loading the saved model
import os  # For handling file paths

def main():
    st.title("Email Marketing Campaign Success Predictor")
    st.write("Upload your dataset and predict campaign success.")

    # File uploader
    uploaded_file = st.file_uploader("Email_Marketing_Campaign_Dataset_Rounded", type=["xlsx", "csv"])
    
    if uploaded_file is not None:
        try:
            # Load the dataset
            if uploaded_file.name.endswith('.xlsx'):
                data = pd.read_excel(uploaded_file)
            else:
                data = pd.read_csv(uploaded_file)
            
            st.write("Uploaded Dataset Preview:")
            st.write(data.head())
            
            # Button to trigger predictions
            if st.button("Predict"):
                try:
                    # Load the pre-trained model
                    model_path = "email_campaign_model.pkl"
                    if not os.path.exists(model_path):
                        st.error("Model file not found. Please ensure the model is uploaded to the app directory.")
                        return
                    
                    loaded_model = joblib.load(model_path)
                    
                    # Validate the data columns
                    expected_columns = loaded_model.feature_names_in_  # Assuming the model was saved with this attribute
                    if not all(col in data.columns for col in expected_columns):
                        st.error("Dataset does not match the expected format. Please upload a valid dataset.")
                        return
                    
                    # Make predictions
                    predictions = loaded_model.predict(data[expected_columns])
                    data['Predicted_Open'] = predictions
                    
                    st.success("Predictions completed!")
                    st.write("Predicted Results Preview:")
                    st.write(data.head())
                    
                    # Provide download link for the results
                    csv = data.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Predicted Results", data=csv, file_name="predicted_results.csv")
                
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
        
        except Exception as e:
            st.error(f"Error loading file: {e}")

if __name__ == '__main__':
    main()
