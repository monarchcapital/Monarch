# Home.py

import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Global Market Dashboard")

# --- Data Configuration ---
SYMBOLS = {
    'Stock Indices': {
        '^GSPC': 'S&P 500', '^IXIC': 'NASDAQ Composite', '^FTSE': 'FTSE 100 (UK)',
        '^N225': 'Nikkei 225 (Japan)', '^GDAXI': 'DAX 30 (Germany)', '^NSEI': 'Nifty 50 (India)',
        '^BVSP': 'Bovespa (Brazil)', '^MXX': 'IPC Mexico', '^HSI': 'Hang Seng (Hong Kong)',
        '^STOXX50E': 'Euro Stoxx 50', '^FCHI': 'CAC 40 (France)',
        '^KS11': 'KOSPI (South Korea)', '^AXJO': 'S&P/ASX 200 (Australia)',
    },
    'Currencies': {
        'DX-Y.NYB': 'US Dollar Index (DXY)', 'EURUSD=X': 'Euro/USD', 'JPY=X': 'USD/JPY',
        'GBPUSD=X': 'British Pound/USD', 'INR=X': 'USD/INR', 'CNY=X': 'USD/CNY',
        'AUDUSD=X': 'Australian Dollar/USD', 'BRL=X': 'USD/BRL',
    },
    'Commodities': {
        'CL=F': 'Crude Oil (WTI)', 'BZ=F': 'Brent Crude Oil', 'NG=F': 'Natural Gas',
        'GC=F': 'Gold', 'SI=F': 'Silver', 'PL=F': 'Platinum', 'HG=F': 'Copper',
    },
    'Government Yields': {
        '^TNX': 'US 10-Year Yield', '^FVX': 'US 5-Year Yield', '^TYX': 'US 30-Year Yield',
    }
}

# --- Helper Functions ---

@st.cache_data(ttl=3600)
def fetch_data(start_date, end_date):
    """
    Fetches and processes market data for all symbols efficiently.
    It retrieves the last two available data points within the date range for comparison.
    """
    all_symbols = [symbol for category in SYMBOLS.values() for symbol in category.keys()]
    
    fetch_start = start_date - timedelta(days=7)
    
    try:
        data = yf.download(all_symbols, start=fetch_start, end=end_date, progress=False)
        if data.empty:
            return {}, None, None

        cleaned_data = data.dropna(how='all').sort_index()
        
        if len(cleaned_data) < 2:
            st.warning("Not enough trading data in the selected range to calculate changes.")
            return {}, None, None

        last_day_data = cleaned_data.iloc[-1]
        prev_day_data = cleaned_data.iloc[-2]

        date_2 = last_day_data.name.date()
        date_1 = prev_day_data.name.date()

        change = last_day_data['Close'] - prev_day_data['Close']
        percent_change = (change / prev_day_data['Close'].replace(0, pd.NA)) * 100

        results_df = pd.DataFrame({
            'Last Close': last_day_data['Close'],
            'Previous Close': prev_day_data['Close'],
            'Open': last_day_data['Open'],
            'High': last_day_data['High'],
            'Low': last_day_data['Low'],
            'Change ($)': change,
            'Change (%)': percent_change
        })

        category_dataframes = {}
        for category_name, category_symbols in SYMBOLS.items():
            df_cat = results_df.loc[results_df.index.isin(category_symbols.keys())].copy()
            df_cat['Indicator'] = df_cat.index.map({k: v for k, v in category_symbols.items()})
            category_dataframes[category_name] = df_cat.dropna(subset=['Last Close', 'Indicator'])

        return category_dataframes, date_1, date_2

    except Exception as e:
        st.error(f"An error occurred during data fetching: {e}")
        return {}, None, None

def color_change(val):
    """Applies color to a value based on its sign."""
    if pd.isna(val):
        return ''
    color = '#2ECC71' if val > 0 else '#E74C3C' if val < 0 else 'white'
    return f'color: {color};'

def generate_heatmap_grid(df, title, num_columns):
    """Generates a responsive, color-coded grid of market indicators."""
    st.markdown(f"<h4 style='text-align: center;'>{title} Heatmap</h4>", unsafe_allow_html=True)
    if not df.empty:
        sorted_df = df.dropna(subset=["Change (%)"]).sort_values("Change (%)", ascending=False)
        
        with st.container(border=True):
            cols = st.columns(num_columns)
            for i, (_, row) in enumerate(sorted_df.iterrows()):
                pct = row["Change (%)"]
                alpha = min(1, abs(pct) / 2.5)
                base_color = "46, 204, 113" if pct > 0 else "231, 76, 60"
                color = f"rgba({base_color}, {alpha + 0.2})"
                
                cols[i % num_columns].markdown(
                    f"""
                    <div class='heatmap-item' style='background-color: {color};'>
                        {row["Indicator"]}<br>{pct:+.2f}%
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

# --- UI Layout ---

# Sidebar for Settings
with st.sidebar:
    st.title("Settings")
    st.session_state.view_mode = st.radio(
        "Select View Mode",
        ('Heatmap', 'Table'),
        key='view_mode_radio'
    )
    if st.session_state.view_mode == 'Heatmap':
        st.session_state.num_columns = st.slider(
            "Heatmap Columns", min_value=2, max_value=10, value=5,
            key='num_cols_slider'
        )
    st.title("Navigation")
    st.info("Select an analysis page. Streamlit automatically lists files from the 'pages' directory.")

# Main Page Content
st.markdown("""
<style>
    .main-header { font-size: 2.5em; font-weight: bold; text-align: center; }
    .subheader { font-size: 1.2em; text-align: center; margin-bottom: 1.5em; color: #AAB7B8; }
    .heatmap-item {
        padding: 10px; border-radius: 8px; text-align: center;
        margin-bottom: 8px; font-size: 0.85rem; font-weight: bold;
        color: white; border: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
<div class="main-header">Global Market Dashboard</div>
<div class="subheader">Daily changes for key stocks, commodities, currencies, and yields</div>
""", unsafe_allow_html=True)

# --- Date Selection ---
col1, col2 = st.columns(2)
today = datetime.now().date()
default_end = today - timedelta(days=1)
default_start = default_end - timedelta(days=1)

start_date_input = col1.date_input("Compare From (Start Date)", value=default_start)
end_date_input = col2.date_input("Compare To (End Date)", value=default_end)
if st.button('Refresh Data', type="primary"):
    st.cache_data.clear()
    st.rerun()

# --- Data Fetching and Display ---
with st.spinner('Fetching market data...'):
    market_data_dfs, date_1, date_2 = fetch_data(start_date_input, end_date_input)

if date_1 and date_2:
    st.header(f"Market Performance: {date_1.strftime('%b %d, %Y')} vs {date_2.strftime('%b %d, %Y')}")
else:
    st.header("Market Performance")

if not market_data_dfs:
    st.error("Could not fetch market data for the selected date range. Please try different dates or check your network connection.")
else:
    if st.session_state.view_mode == 'Table':
        for category in SYMBOLS.keys():
            st.markdown(f"<h5>{category}</h5>", unsafe_allow_html=True)
            if category in market_data_dfs and not market_data_dfs[category].empty:
                df_display = market_data_dfs[category].sort_values("Change (%)", ascending=False).reset_index(drop=True)
                format_dict = {
                    'Last Close': "{:,.2f}", 'Previous Close': "{:,.2f}", 'Open': "{:,.2f}",
                    'High': "{:,.2f}", 'Low': "{:,.2f}", 'Change ($)': "{:+.2f}", 'Change (%)': "{:+.2f}%"
                }
                styled_df = df_display.style.applymap(color_change, subset=['Change ($)', 'Change (%)']).format(format_dict, na_rep='-')
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
            else:
                st.warning(f"No data available for {category} for the selected dates.")
    
    elif st.session_state.view_mode == 'Heatmap':
        for category in SYMBOLS.keys():
            if category in market_data_dfs:
                generate_heatmap_grid(market_data_dfs[category], category, st.session_state.num_columns)
            else:
                st.warning(f"No data available for {category} heatmap.")