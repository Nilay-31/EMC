# Import necessary libraries
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load the trained model
model_path = 'email_open_prediction_model.pkl'
model = joblib.load(model_path)

# Define numerical columns used in the model
numerical_columns = [
    'Customer_Age',
    'Emails_Opened',
    'Emails_Clicked',
    'Purchase_History',
    'Time_Spent_On_Website',
    'Days_Since_Last_Open',
    'Customer_Engagement_Score'
]

# Title of the app
st.title("Email Marketing Campaign Prediction")
st.markdown("""
This app predicts whether a customer is likely to open previous emails based on historical data.
""")

# Sidebar for user input
st.sidebar.header("Input Features")
def user_input_features():
    # Collect user inputs for each feature
    data = {
        'Customer_Age': st.sidebar.slider('Customer Age', 18, 80, 35),
        'Emails_Opened': st.sidebar.slider('Emails Opened', 0, 50, 5),
        'Emails_Clicked': st.sidebar.slider('Emails Clicked', 0, 50, 2),
        'Purchase_History': st.sidebar.slider('Purchase History', 0.0, 1000.0, 200.0),
        'Time_Spent_On_Website': st.sidebar.slider('Time Spent on Website (minutes)', 0.0, 500.0, 30.0),
        'Days_Since_Last_Open': st.sidebar.slider('Days Since Last Open', 0, 365, 50),
        'Customer_Engagement_Score': st.sidebar.slider('Customer Engagement Score', 0.0, 100.0, 50.0),
    }
    return pd.DataFrame([data])

# Collect user inputs
input_df = user_input_features()

# Preprocess input data using the same scaler used during training
scaler = StandardScaler()
scaled_input = scaler.fit_transform(input_df[numerical_columns])

# Display the user input
st.subheader("User Input Features")
st.write(input_df)

# Predict using the trained model
prediction = model.predict(scaled_input)
prediction_proba = model.predict_proba(scaled_input)

# Display the prediction
st.subheader("Prediction")
if prediction[0] == 1:
    st.success("The customer is likely to open previous emails.")
else:
    st.warning("The customer is unlikely to open previous emails.")

# Display prediction probabilities
st.subheader("Prediction Probability")
st.write(f"Not Opened: {prediction_proba[0][0]:.2f}")
st.write(f"Opened: {prediction_proba[0][1]:.2f}")

# Visualization
st.subheader("Feature Importance (Random Forest)")
feature_importance = model.named_steps['classifier'].feature_importances_
fig, ax = plt.subplots()
sns.barplot(x=feature_importance, y=numerical_columns, ax=ax)
ax.set_title("Feature Importance")
st.pyplot(fig)
