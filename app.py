import streamlit as st
import numpy as np
import joblib
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Load the trained model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# Set page config for dark blue theme
st.set_page_config(page_title="Diabetes Prediction", layout="wide")

# Custom CSS for dark mode
st.markdown("""
    <style>
        body {
            background-color: #0a1a2f;
            color: white;
        }
        .stApp {
            background-color: #0a1a2f;
            color: white;
        }
        .stButton>button {
            background-color: #1f6feb;
            color: white;
            border-radius: 8px;
            padding: 10px 24px;
        }
        .stNumberInput>div>div>input {
            background-color: #1e1e1e;
            color: white;
        }
        label, .css-1y0tads, .css-1cpxqw2, .stTextInput>div>input {
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🧬 Diabetes Prediction System")
st.write("Please enter your medical information below:")

# Layout: Two input columns
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
    glucose = st.number_input("Glucose", min_value=0)
    bp = st.number_input("Blood Pressure", min_value=0)
    skin = st.number_input("Skin Thickness", min_value=0)

with col2:
    insulin = st.number_input("Insulin", min_value=0)
    bmi = st.number_input("BMI", min_value=0.0, format="%.2f")
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
    age = st.number_input("Age", min_value=0, step=1)

# Prediction logic
if st.button("🔍 Predict"):
    # Prepare and scale input
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]  # Risk probability
    percentage = round(probability * 100, 2)

    # Display result
    st.markdown("---")
    if prediction[0] == 1:
        st.error(f"⚠️ High chance of diabetes! Risk Probability: {percentage}%")
    else:
        st.success(f"✅ You are likely safe from diabetes. Risk Probability: {percentage}%")

    # Layout for visualizations
    st.markdown("## 📊 Diabetes Risk Visualizations")
    donut_col, bar_col = st.columns(2)

    # Donut Chart
    with donut_col:
        st.markdown("### 🍩 Risk Probability Donut Chart")
        fig_donut = go.Figure(data=[go.Pie(
            labels=['Risk', 'No Risk'],
            values=[percentage, 100 - percentage],
            hole=0.5,
            marker=dict(colors=['#EF553B', '#00CC96']),
            textinfo='label+percent',
            insidetextorientation='radial'
        )])
        fig_donut.update_layout(showlegend=True, height=300, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_donut, use_container_width=True)

    # Bar Chart
    with bar_col:
        st.markdown("### 📶 Risk Probability Bar Chart")
        fig_bar = go.Figure(data=[
            go.Bar(name='Risk', x=['Prediction'], y=[percentage], marker_color='#EF553B'),
            go.Bar(name='No Risk', x=['Prediction'], y=[100 - percentage], marker_color='#00CC96')
        ])
        fig_bar.update_layout(
            barmode='group',
            yaxis=dict(title='Percentage'),
            height=300,
            margin=dict(t=30, b=0, l=0, r=0)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Health Parameters Overview
    st.markdown("## 🧾 Your Health Parameters")
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    df_input = pd.DataFrame(input_data, columns=columns).T.reset_index()
    df_input.columns = ['Feature', 'Value']

    fig_param = px.bar(
        df_input,
        x='Feature',
        y='Value',
        title="Health Parameters Overview",
        color='Value',
        color_continuous_scale='Blues'
    )
    fig_param.update_layout(height=400)
    st.plotly_chart(fig_param, use_container_width=True)
