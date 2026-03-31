import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import folium
from streamlit_folium import st_folium
import time

st.set_page_config(page_title="HK HVAC Live Selector", layout="wide")

def fetch_weather(lat, lon, start=None, end=None, is_forecast=False):
    url = "https://api.open-meteo.com/v1/forecast" if is_forecast else "https://archive-api.open-meteo.com/v1/archive"
    p = {"latitude": lat, "longitude": lon, "daily": ["temperature_2m_mean", "relative_humidity_2m_mean", "wind_speed_10m_max"], "timezone": "Asia/Hong_Kong"}
    if is_forecast: p["forecast_days"] = 14
    else: p["start_date"], p["end_date"] = start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')
    try:
        res = requests.get(url, params=p, timeout=20)
        return res.json() if res.status_code == 200 else None
    except: return None

# --- UI ---
st.title("🏢 HVAC Predictive Analytics Dashboard")
st.markdown("---")

col_map, col_input = st.columns([1.5, 1])

# Initialize session state for coordinates if not set
if 'lat' not in st.session_state: st.session_state.lat = 22.3193
if 'lon' not in st.session_state: st.session_state.lon = 114.1694

with col_map:
    st.subheader("🗺️ Click the map to select building location")
    m = folium.Map(location=[st.session_state.lat, st.session_state.lon], zoom_start=12)
    # Add a marker at the currently selected point
    folium.Marker([st.session_state.lat, st.session_state.lon], tooltip="Selected Building").add_to(m)
    
    # Render map and capture clicks
    map_data = st_folium(m, height=450, width=700)
    
    if map_data and map_data['last_clicked']:
        st.session_state.lat = map_data['last_clicked']['lat']
        st.session_state.lon = map_data['last_clicked']['lng']
        st.rerun() # Refresh to update the input fields and marker

with col_input:
    st.header("📋 Analysis Controls")
    st.info(f"Selected: {st.session_state.lat:.4f}N, {st.session_state.lon:.4f}E")
    uploaded_file = st.file_uploader("Upload Historical ElectricityConsumption Data", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.columns = ['Date', 'kWh']
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    if st.button("🚀 Run Forecast Analysis"):
        with st.status("Performing Data Science Workflow...", expanded=True) as status:
            st.write("Step 1: Syncing Historical Weather...")
            raw_w = fetch_weather(st.session_state.lat, st.session_state.lon, df['Date'].min(), df['Date'].max())
            w_df = pd.DataFrame({'Date': pd.to_datetime(raw_w['daily']['time']), 'T': raw_w['daily']['temperature_2m_mean'], 'H': raw_w['daily']['relative_humidity_2m_mean'], 'W': raw_w['daily']['wind_speed_10m_max']})
        
            m_df = pd.merge(df, w_df, on='Date', how='inner')
            m_df['L1'], m_df['L7'], m_df['R7'] = m_df['kWh'].shift(1), m_df['kWh'].shift(7), m_df['kWh'].rolling(window=7).mean()
            train_set = m_df.dropna()

            st.write("Step 2: Scaling & Training Model...")
            feats = ['T', 'H', 'W', 'L1', 'L7', 'R7']
            X_raw, y = train_set[feats], train_set['kWh']
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_raw)
            model = LinearRegression().fit(X_scaled, y)
            
            st.write("Step 3: Recursive 14-Day Forecast...")
            raw_f = fetch_weather(st.session_state.lat, st.session_state.lon, is_forecast=True)
            f_df = pd.DataFrame({'Date': pd.to_datetime(raw_f['daily']['time']), 'Forecasted Mean Temperature (°C)': raw_f['daily']['temperature_2m_mean'], 'Forecasted Relative Humidity (%)': raw_f['daily']['relative_humidity_2m_mean'], 'Forecasted Wind Speed (km/h)': raw_f['daily']['wind_speed_10m_max']})
            
            hist = m_df.tail(14).copy()
            preds = []
            for i in range(len(f_df)):
                row = f_df.iloc[i]
                l1, l7, r7 = hist['kWh'].iloc[-1], hist['kWh'].iloc[-7], hist['kWh'].tail(7).mean()
                curr_x = np.array([[row['Forecasted Mean Temperature (°C)'], row['Forecasted Relative Humidity (%)'], row['Forecasted Wind Speed (km/h)'], l1, l7, r7]])
                p = model.predict(scaler.transform(curr_x))[0]
                preds.append(p)
                hist = pd.concat([hist, pd.DataFrame({'Date':[row['Date']], 'kWh':[p]})]).reset_index(drop=True)
            
            f_df['Forecasted Consumption (kWh)'] = preds
            status.update(label="✅ Analysis Complete!", state="complete", expanded=False)

        # Result Displays
        st.columns(2)[0].metric("Model Confidence (R²)", f"{r2_score(y, model.predict(X_scaled)):.4f}")
        st.plotly_chart(go.Figure([go.Scatter(x=df['Date'], y=df['kWh'], name="Actual"), go.Scatter(x=f_df['Date'], y=f_df['Forecasted Consumption (kWh)'], name="Future Forecast", line=dict(color='orange'))]), use_container_width=True)

        st.write("### 📅 Forecasted Weather & Consumption Data")
        st.dataframe(f_df.style.format({
            "Forecasted Mean Temperature (°C)": "{:.1f}",
            "Forecasted Relative Humidity (%)": "{:.0f}",
            "Forecasted Wind Speed (km/h)": "{:.1f}",
            "Forecasted Consumption (kWh)": "{:.2f}"
        }))
