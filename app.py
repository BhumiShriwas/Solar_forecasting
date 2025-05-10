import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Load and preprocess the data
@st.cache_data
def load_data():
    df = pd.read_csv("Plant_1_Generation_Data.csv")
    df = df.dropna()
    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])
    df['Hour'] = df['DATE_TIME'].dt.hour
    df['Day'] = df['DATE_TIME'].dt.day
    df['Month'] = df['DATE_TIME'].dt.month
    return df

# Train the model
@st.cache_resource
def train_model(df):
    X = df[['Hour', 'Day', 'Month']]
    y = df['DC_POWER']
    model = RandomForestRegressor()
    model.fit(X, y)
    return model

# Load data and train model
df = load_data()
model = train_model(df)

# Streamlit UI
st.title("🔆 Solar Power Forecasting App")

st.markdown("Enter the time details below to predict the expected **DC power output** of a solar plant:")

hour = st.slider("Select Hour", 0, 23, 12)
day = st.slider("Select Day of Month", 1, 31, 15)
month = st.slider("Select Month", 1, 12, 6)

if st.button("Predict DC Power"):
    input_data = pd.DataFrame([[hour, day, month]], columns=['Hour', 'Day', 'Month'])
    prediction = model.predict(input_data)[0]
    st.success(f"🌞 Predicted DC Power Output: **{prediction:.2f}** kW")

# Optional: Show sample data and chart
with st.expander("🔍 View Sample Data & Chart"):
    st.write(df.head())
    st.line_chart(df['DC_POWER'].reset_index(drop=True))
