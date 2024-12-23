import streamlit as st  # For deployment
import pandas as pd  # For data handling
def main():
    st.title('Email Marketing Campaign Success Predictor')

    # User input
    age = st.number_input('Customer Age', min_value=18, max_value=100, value=35)
    emails_opened = st.number_input('Emails Opened', min_value=0, max_value=50, value=5)
    emails_clicked = st.number_input('Emails Clicked', min_value=0, max_value=20, value=2)
    purchase_history = st.number_input('Purchase History', min_value=0.0, value=1500.0)
    time_spent = st.number_input('Time Spent on Website', min_value=0.0, value=5.0)
    days_since_last_open = st.number_input('Days Since Last Open', min_value=0, value=30)
    engagement_score = st.number_input('Customer Engagement Score', min_value=0.0, value=70.0)
    device_type = st.radio('Device Type', ('Desktop', 'Mobile'))

    # Encode device type
    device_type = 1 if device_type == 'Mobile' else 0

  # Prediction
    user_data = [[age, emails_opened, emails_clicked, purchase_history, time_spent, days_since_last_open, engagement_score, device_type]]
    user_data_scaled = scaler.transform(user_data)
    prediction = model.predict(user_data_scaled)

    st.write('Prediction:', 'Opened' if prediction[0] == 1 else 'Not Opened')
    main()

if __name__ == '__main__':
   
