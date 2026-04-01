import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import folium
from streamlit_folium import st_folium

# --- 1. LIVE DATA FETCHING (CHINA MARKET -> HKD) ---
@st.cache_data(ttl=3600)
def get_live_china_carbon_price_hkd():
    # Current exchange rate estimate: 1 CNY = 1.09 HKD (Updated for 2026)
    EXCHANGE_RATE = 1.09 
    DEFAULT_HKD = 105.70
    
    try:
        url = "https://carboncredits.com/carbon-prices-today/"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Target the China row specifically
        rows = soup.find_all('tr')
        for row in rows:
            if 'China' in row.text:
                cells = row.find_all('td')
                for cell in cells:
                    if '¥' in cell.text:
                        # Convert ¥97.00 -> 97.0
                        cny_price = float(cell.text.replace('¥', '').replace(',', '').strip())
                        return round(cny_price * EXCHANGE_RATE, 2)
        return DEFAULT_HKD
    except:
        return DEFAULT_HKD

LIVE_PRICE_HKD = get_live_china_carbon_price_hkd()

st.set_page_config(page_title="HVAC Carbon Planning Suite", layout="wide")

def fetch_weather(lat, lon, start=None, end=None, is_forecast=False):
    url = "https://api.open-meteo.com/v1/forecast" if is_forecast else "https://archive-api.open-meteo.com/v1/archive"
    p = {"latitude": lat, "longitude": lon, "daily": ["temperature_2m_mean", "relative_humidity_2m_mean", "wind_speed_10m_max"], "timezone": "Asia/Hong_Kong"}
    if is_forecast: p["forecast_days"] = 14
    else: p["start_date"], p["end_date"] = start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')
    try:
        res = requests.get(url, params=p, timeout=20)
        return res.json() if res.status_code == 200 else None
    except: return None

# --- UI HEADER ---
st.title("📊 Strategic HVAC Carbon & Energy Planning")
st.caption("Reference Market: China Compliance (Calculated in HKD)")
st.markdown("---")

# --- 2. SIDEBAR PLANNING CONTROLS ---
st.sidebar.header("📈 Planning Parameters")

# Dynamic Grid Factor for local utility context
grid_factor = st.sidebar.slider(
    "Grid Emission Factor (kg CO2/kWh)", 
    0.0, 1.2, 0.60, 0.01,
    help="HK Electric: ~0.60 | CLP: ~0.39. Adjust based on your utility provider."
)

mitigation_target = st.sidebar.slider("Target Savings Rate (%)", 0, 40, 15)

st.sidebar.metric("Live China Carbon Price", f"${LIVE_PRICE_HKD:.2f} HKD/Ton")

# --- 3. INPUT SECTION ---
col_map, col_input = st.columns([1.5, 1])

if 'lat' not in st.session_state: st.session_state.lat = 22.3193
if 'lon' not in st.session_state: st.session_state.lon = 114.1694

with col_map:
    st.subheader("🗺️ Building Location Selection")
    m = folium.Map(location=[st.session_state.lat, st.session_state.lon], zoom_start=12)
    folium.Marker([st.session_state.lat, st.session_state.lon]).add_to(m)
    map_data = st_folium(m, height=350, width=600)
    if map_data and map_data['last_clicked']:
        st.session_state.lat, st.session_state.lon = map_data['last_clicked']['lat'], map_data['last_clicked']['lng']
        st.rerun()

with col_input:
    st.header("📋 Historical Baseline Upload")
    uploaded_file = st.file_uploader("Upload Electricity Data (.xlsx)", type=["xlsx"])

# --- 4. FULL DATASET ML PIPELINE ---
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.columns = ['Date', 'kWh']
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    if st.button("🚀 Generate Planning Forecast"):
        with st.status("Training Model on Full Historical Dataset...", expanded=True) as status:
            raw_w = fetch_weather(st.session_state.lat, st.session_state.lon, df['Date'].min(), df['Date'].max())
            w_df = pd.DataFrame({
                'Date': pd.to_datetime(raw_w['daily']['time']), 
                'T': raw_w['daily']['temperature_2m_mean'], 
                'H': raw_w['daily']['relative_humidity_2m_mean'], 
                'W': raw_w['daily']['wind_speed_10m_max']
            })
            m_df = pd.merge(df, w_df, on='Date', how='inner')
            
            m_df['L1'], m_df['L7'], m_df['R7'] = m_df['kWh'].shift(1), m_df['kWh'].shift(7), m_df['kWh'].rolling(window=7).mean()
            train_set = m_df.dropna()

            feats = ['T', 'H', 'W', 'L1', 'L7', 'R7']
            X_raw, y = train_set[feats], train_set['kWh']
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_raw)
            model = LinearRegression().fit(X_scaled, y)
            
            raw_f = fetch_weather(st.session_state.lat, st.session_state.lon, is_forecast=True)
            f_df = pd.DataFrame({
                'Date': pd.to_datetime(raw_f['daily']['time']), 
                'Forecasted Temperature (°C)': raw_f['daily']['temperature_2m_mean'], 
                'Forecasted Relative Humidity (%)': raw_f['daily']['relative_humidity_2m_mean'], 
                'Forecasted Wind Speed (km/h)': raw_f['daily']['wind_speed_10m_max']
            })
            
            hist = m_df.tail(14).copy()
            preds = []
            for i in range(len(f_df)):
                row = f_df.iloc[i]
                l1, l7, r7 = hist['kWh'].iloc[-1], hist['kWh'].iloc[-7], hist['kWh'].tail(7).mean()
                curr_x = np.array([[row['Forecasted Temperature (°C)'], row['Forecasted Relative Humidity (%)'], row['Forecasted Wind Speed (km/h)'], l1, l7, r7]])
                p = model.predict(scaler.transform(curr_x))[0]
                preds.append(p)
                hist = pd.concat([hist, pd.DataFrame({'Date':[row['Date']], 'kWh':[p]})]).reset_index(drop=True)
            
            f_df['Baseline_kWh'] = preds
            status.update(label="✅ Comprehensive Model Ready!", state="complete", expanded=False)

        # --- 5. CARBON PLANNING CALCULATIONS (HKD Focused) ---
        f_df['Projected_Mitigation_kWh'] = f_df['Baseline_kWh'] * (1 - (mitigation_target / 100))
        f_df['Baseline_CO2_Tons'] = (f_df['Baseline_kWh'] * grid_factor) / 1000
        f_df['Mitigated_CO2_Tons'] = (f_df['Projected_Mitigation_kWh'] * grid_factor) / 1000
        
        total_saved_co2 = f_df['Baseline_CO2_Tons'].sum() - f_df['Mitigated_CO2_Tons'].sum()
        est_offset_value_hkd = total_saved_co2 * LIVE_PRICE_HKD

        # --- 6. PLANNING DASHBOARD ---
        st.subheader("🗓️ 14-Day Strategic Outlook")
        m1, m2, m3 = st.columns(3)
        m1.metric("Historical Model R²", f"{r2_score(y, model.predict(X_scaled)):.4f}")
        m2.metric("Est. Carbon Mitigation", f"{total_saved_co2:.3f} tCO2e")
        m3.metric("Est. Offset Value (HKD)", f"${est_offset_value_hkd:.2f}")

        fig = go.Figure()
        # Full historical background
        fig.add_trace(go.Scatter(x=df['Date'], y=df['kWh'], name="Historical Baseline", line=dict(color='gray', width=1)))
        # Forecasted lines
        fig.add_trace(go.Scatter(x=f_df['Date'], y=f_df['Baseline_kWh'], name="Projected Baseline", line=dict(dash='dash', color='blue')))
        fig.add_trace(go.Scatter(x=f_df['Date'], y=f_df['Projected_Mitigation_kWh'], name="Mitigation Target", fill='tonexty', line=dict(color='green')))
        
        fig.update_layout(title="Future Energy Planning: Baseline vs. Mitigation Target", xaxis_title="Date", yaxis_title="kWh")
        st.plotly_chart(fig, use_container_width=True)

        st.write("### 📂 Planning Summary Data")
        
        # Applying 1 decimal place to Weather and 2 to 4 to the rest
        st.dataframe(f_df.style.format({
            "Forecasted Temperature (°C)": "{:.1f}",
            "Forecasted Relative Humidity (%)": "{:.1f}",
            "Forecasted Wind Speed (km/h)": "{:.1f}",
            "Baseline_kWh": "{:.2f}", 
            "Projected_Mitigation_kWh": "{:.2f}", 
            "Baseline_CO2_Tons": "{:.4f}",
            "Mitigated_CO2_Tons": "{:.4f}"
        }))

