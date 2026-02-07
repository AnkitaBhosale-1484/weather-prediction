import streamlit as st
import numpy as np
import joblib
import pandas as pd
from fpdf import FPDF
from datetime import datetime
import os

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Weather Temperature Prediction",
    page_icon="🌦️",
    layout="wide"
)

# ---------------- Load Model ----------------
model = joblib.load("model/temperature_model.pkl")

# ---------------- Session State ----------------
if "page" not in st.session_state:
    st.session_state.page = "input"

# ---------------- CSS (NO HIDING ANYTHING) ----------------
st.markdown("""
<style>
/* Overall font size */
html, body, [class*="css"]  {
    font-size: 18px;
}

/* Main container */
.main {
    padding-top: 20px;
}

/* Card style */
.card {
    background: rgba(255, 255, 255, 0.85);
    border-radius: 18px;
    padding: 30px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    margin-bottom: 25px;
}

/* Big temperature */
.temp {
    font-size: 64px;
    font-weight: 800;
    color: #e74c3c;
}

/* Buttons */
.stButton > button {
    font-size: 20px;
    padding: 12px 28px;
    border-radius: 12px;
}

/* Inputs */
input, select, textarea {
    font-size: 18px !important;
}

/* Titles */
h1, h2, h3 {
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# ===================== INPUT PAGE ========================
# =========================================================
if st.session_state.page == "input":

    st.title("🌦️ Weather Temperature Prediction System")
    st.write("Advanced Machine Learning Based Weather Forecasting")

    with st.form("weather_form"):
        col1, col2 = st.columns(2)

        with col1:
            MinTemp = st.number_input("Minimum Temperature (°C)", value=22.0)
            Humidity = st.number_input("Humidity (%)", value=65.0, min_value=0.0, max_value=150.0)
            WindSpeed = st.number_input("Wind Speed (km/h)", value=12.0)
            Pressure = st.number_input("Pressure (hPa)", value=1012.0)
            Rainfall = st.number_input("Rainfall (mm)", value=3.0)

        with col2:
            CloudCover = st.slider("Cloud Cover (%)", 0, 100, 40)
            Sunshine = st.number_input("Sunshine Hours", value=7.0)
            DewPoint = st.number_input("Dew Point (°C)", value=18.0)
            MonthName = st.selectbox(
                "Month",
                ["January","February","March","April","May","June",
                 "July","August","September","October","November","December"]
            )
            Month = ["January","February","March","April","May","June",
                     "July","August","September","October","November","December"].index(MonthName) + 1

        submit = st.form_submit_button("🔮 Predict Temperature")

    if submit:

        if Humidity > 100:
            st.warning(f"⚠️ Humidity should not exceed 100%. Entered: {Humidity}%")

        if Sunshine > 12:
            st.warning(f"⚠️ Sunshine hours usually ≤ 12. Entered: {Sunshine} hours")

        # Save inputs
        st.session_state.input_data = np.array([[
            MinTemp, Humidity, WindSpeed, Pressure,
            Rainfall, CloudCover, Sunshine, DewPoint, Month
        ]])

        st.session_state.raw_inputs = {
            "MinTemp": MinTemp,
            "Humidity": Humidity,
            "WindSpeed": WindSpeed,
            "Pressure": Pressure,
            "Rainfall": Rainfall,
            "CloudCover": CloudCover,
            "Sunshine": Sunshine,
            "DewPoint": DewPoint,
            "Month": Month
        }

        st.session_state.page = "result"
        st.rerun()

# =========================================================
# ===================== RESULT PAGE =======================
# =========================================================
if st.session_state.page == "result":

    prediction = model.predict(st.session_state.input_data)[0]

    if prediction > 35:
        summary_ui = "🔥 Very Hot Weather Expected"
        summary_pdf = "Very Hot Weather Expected"
        emoji = "🔥"
    elif prediction > 25:
        summary_ui = "🌤️ Moderate Weather Conditions"
        summary_pdf = "Moderate Weather Conditions"
        emoji = "🌤️"
    else:
        summary_ui = "❄️ Cool Weather Conditions"
        summary_pdf = "Cool Weather Conditions"
        emoji = "❄️"

    st.markdown(f"""
    <div class="card">
        <h2>{emoji} Predicted Maximum Temperature</h2>
        <div class="temp">{prediction:.1f} °C</div>
        <p><b>{summary_ui}</b></p>
    </div>
    """, unsafe_allow_html=True)

    # Chart
    st.subheader("📊 Temperature Visualization")
    chart_df = pd.DataFrame({
        "Day": range(1, 6),
        "Predicted Temperature": [prediction] * 5
    }).set_index("Day")
    st.line_chart(chart_df)

    # PDF
    os.makedirs("reports", exist_ok=True)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(0, 10, "Weather Temperature Prediction Report", ln=True)
    pdf.cell(0, 10, f"Generated On: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}", ln=True)
    pdf.ln(5)

    pdf.cell(0, 10, f"Predicted Temperature: {prediction:.1f} C", ln=True)
    pdf.cell(0, 10, f"Weather Summary: {summary_pdf}", ln=True)
    pdf.ln(5)

    for key, value in st.session_state.raw_inputs.items():
        pdf.cell(0, 10, f"{key}: {value}", ln=True)

    report_path = "reports/weather_prediction_report.pdf"
    pdf.output(report_path)

    with open(report_path, "rb") as file:
        st.download_button(
            "📄 Download Prediction Report (PDF)",
            file,
            file_name="weather_prediction_report.pdf",
            mime="application/pdf"
        )

    if st.button("⬅️ Back to Input Page"):
        st.session_state.page = "input"
        st.rerun()