import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
import numpy as np
from datetime import datetime, timedelta, timezone


# ---------------------- Asset Universe Configuration ----------------------

# Large-cap and Small/Mid-cap Equity Indices by Region with Yahoo Finance Tick Symbols

INDEX_TICKERS = [
    # North America
    ('^GSPC',  'S&P 500 (Large Cap)',              'North America', 'Equity Index'),
    ('^RUT',   'Russell 2000 (Small Cap)',         'North America', 'Equity Small Cap Index'),
    ('^MID',   'S&P MidCap 400 (Mid Cap)',         'North America', 'Equity Mid Cap Index'),

    # United Kingdom
    ('^FTSE',  'FTSE 100 (Large Cap)',             'UK',            'Equity Index'),
    ('^FTMC',  'FTSE 250 (Mid Cap)',                'UK',            'Equity Mid Cap Index'),
    ('^FTSC',  'FTSE Small Cap Index',              'UK',            'Equity Small Cap Index'),

    # Europe ex-UK
    ('^GDAXI', 'DAX 40 (Large Cap Germany)',       'Europe',        'Equity Index'),
    ('^MDAXI', 'MDAX (Mid Cap Germany)',            'Europe',        'Equity Mid Cap Index'),
    ('^SDAXI', 'SDAX (Small Cap Germany)',          'Europe',        'Equity Small Cap Index'),
    ('^FCHI',  'CAC 40 (Large Cap France)',         'Europe',        'Equity Index'),
    #('^CACMID','CAC Mid 60 (Mid Cap France)',       'Europe',        'Equity Mid Cap Index'),

    # Asia Pacific
    ('^N225',  'Nikkei 225 (Large Cap Japan)',     'Asia',          'Equity Index'),
    #('^TOPIX', 'TOPIX (Mid Cap/Broad Japan)',       'Asia',          'Equity Mid Cap Index'),
    #('^MOTHERS','Mothers Index (Small Growth Japan)','Asia',         'Equity Small Cap Index'),
    ('^HSI',   'Hang Seng (Large Cap HK)',          'Asia',          'Equity Index'),
    ('^HSCE',  'HSCE (China Enterprises Large-Mid)', 'Asia',         'Equity Mid Cap Index'),
    ('^AAXJ',  'MSCI Asia ex-Japan (Large Cap ETF proxy)', 'Asia',    'Equity Index'),
    ('^AXJO',  'S&P/ASX 200 (Large Cap Australia)', 'Asia',          'Equity Index'),
    ('^AXSO',  'S&P/ASX Small Ordinaries (Small Cap Australia)', 'Asia', 'Equity Small Cap Index'),

    # Latin America
    ('^BVSP',  'Bovespa (Large Cap Brazil)',        'LatAm',         'Equity Index'),
    ('^MXX',   'S&P/BMV IPC (Large Cap Mexico)',    'LatAm',         'Equity Index'),
    ('EWZS',   'MSCI Brazil Small Cap ETF proxy',   'LatAm',         'Equity Small Cap Index'),
    ('EWW',    'Mexico ETF broad proxy',             'LatAm',         'Equity Index'),
]

# Sovereign Bonds (2y, 5y, 10y yields)
BOND_TICKERS = [
    # US Treasuries
    #('^UST2Y', 'US 2Y Treasury Yield',     'North America', 'Fixed Income Sovereign'),
    ('^FVX',   'US 5Y Treasury Yield',     'North America', 'Fixed Income Sovereign'),
    ('^TNX',   'US 10Y Treasury Yield',    'North America', 'Fixed Income Sovereign'),

    # UK Gilts (Yahoo tickers for yields may vary or require ETFs as proxy)
    #('GB2:GOV',  'UK 2Y Gilt Yield',       'UK',            'Fixed Income Sovereign'),
    #('GB5:GOV',  'UK 5Y Gilt Yield',       'UK',            'Fixed Income Sovereign'),
    #('GB10:GOV', 'UK 10Y Gilt Yield',      'UK',            'Fixed Income Sovereign'),
]

BOND_TICKER_SYMBOLS = {t[0] for t in BOND_TICKERS}


# Commodities (Energy, Metals, Agricultural)
COMMODITY_TICKERS = [
    ('GC=F',  'Gold',              'Global', 'Commodity Metal'),
    ('SI=F',  'Silver',            'Global', 'Commodity Metal'),
    ('HG=F',  'Copper',            'Global', 'Commodity Metal'),
    ('CL=F',  'WTI Crude Oil',     'Global', 'Commodity Energy'),
    ('BZ=F',  'Brent Crude',       'Global', 'Commodity Energy'),
    ('NG=F',  'Natural Gas',       'Global', 'Commodity Energy'),
    ('PL=F',  'Platinum',          'Global', 'Commodity Metal'),
    ('PA=F',  'Palladium',         'Global', 'Commodity Metal'),
    ('ZS=F',  'Soybeans',          'Global', 'Commodity Agri'),
    ('ZW=F',  'Wheat',             'Global', 'Commodity Agri'),
    ('ZC=F',  'Corn',              'Global', 'Commodity Agri'),
    ('KC=F',  'Coffee',            'Global', 'Commodity Soft'),
    ('SB=F',  'Sugar',             'Global', 'Commodity Soft'),
    ('CT=F',  'Cotton',            'Global', 'Commodity Soft')
]

# Crypto assets
CRYPTO_TICKERS = [
    ('BTC-USD', 'Bitcoin',    'Global', 'Crypto'),
    ('ETH-USD', 'Ethereum',   'Global', 'Crypto'),
]

# Placeholder for Top Companies; user should populate with tuples: (symbol, name, region, asset class, market cap)
TOP_COMPANIES = [
    # ('AAPL', 'Apple Inc.', 'North America', 'Equity', 3000000000000),
    # Add your top 50 companies per region here with market caps in USD
]

# Combine all assets
ALL_TICKERS = INDEX_TICKERS + BOND_TICKERS + COMMODITY_TICKERS + CRYPTO_TICKERS + TOP_COMPANIES


@st.cache_data(show_spinner=False)
def fetch_asset_history(symbol, periods):
    import re
    end = datetime.now(timezone.utc)
    errors = []

    # Define tickers with limited valid periods (e.g. AAXJ ETF, newer indices)
    limited_period_tickers = ['^AAXJ']  # add any others here

    try:
        ticker = yf.Ticker(symbol)
        hist = None

        # Pick appropriate history period to fetch
        # If ticker in limited_period_tickers, only allow max 5d
        if symbol in limited_period_tickers:
            # Fetch at most 5 days
            hist = ticker.history(period="5d")
        else:
            # Try max or longest to have all data
            hist = ticker.history(period="max")

        if hist is None or hist.empty:
            return None, {k: np.nan for k in periods.keys()}, [f"{symbol}: No historical data."]
        hist = hist[~hist.index.duplicated(keep='first')]

        last_price = hist['Close'].iloc[-1]

        end_date = hist.index[-1].date() if hist is not None and not hist.empty else pd.NaT
        

        def perf_at(period_days):
            if period_days == 0:
                prev = hist['Close'].iloc[-2] if len(hist) > 1 else None
            else:
                period_start_date = end - timedelta(days=period_days)
                prev_idx = hist.index.searchsorted(period_start_date, side='left')
                if prev_idx >= len(hist):
                    return np.nan
                prev = hist['Close'].iloc[prev_idx]
            
            if prev is None or last_price is None:
                return np.nan

            if symbol in BOND_TICKER_SYMBOLS:
                return last_price - prev
            else:
                if prev > 0:
                    return 100 * (last_price - prev) / prev
                else:
                    return np.nan

        perf_dict = {}
        for key, ndays in periods.items():
            try:
                # Limit period performance to 5d max for limited tickers
                if symbol in limited_period_tickers and ndays > 5:
                    perf_dict[key] = np.nan
                else:
                    perf_dict[key] = perf_at(ndays) if last_price is not None else np.nan
            except Exception:
                perf_dict[key] = np.nan

        return last_price, perf_dict, None, end_date

    except Exception as e:
        errors.append(f"{symbol}: {str(e)}")
        return None, {k: np.nan for k in periods.keys()}, errors, pd.NaT


# Data collection producing a numeric-clean DataFrame
def collect_dashboard_data():
    periods = {
        'Perf 1D': 1,
        'Perf 1W': 7,
        'Perf 1M': 30,
        'Perf 3M': 90,
        'Perf 6M': 182,
        'Perf 1Y': 365,
        'Perf 2Y': 365*2,
        'Perf 3Y': 365*3
    }

    rows = []
    errors = []

    for asset in ALL_TICKERS:
        if len(asset) == 4:
            symbol, name, region, asset_class = asset
            mcap = np.nan
        else:
            symbol, name, region, asset_class, mcap = asset
        
        # Convert market cap to numeric nan if needed
        if not isinstance(mcap, (float, int)):
            try:
                mcap = float(mcap)
            except Exception:
                mcap = np.nan

        last_price, perf, err, end_date = fetch_asset_history(symbol, periods)
        
        if err:
            errors.extend(err)
        perf = perf if perf else {k: np.nan for k in periods.keys()}

                
        row = {
            'Symbol': symbol,
            'Name': name,
            'Region': region,
            'Asset Class': asset_class,
            'Market Cap': mcap,
            'End Date': end_date,
            'Current Price': np.nan if last_price is None else last_price,
            **{k: (perf.get(k, np.nan)) for k in periods.keys()}
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    return df, errors

# ---------------------- Streamlit Dashboard ----------------------

def run_dashboard():
    st.title("Global Financial Markets Dashboard")

    dashboard_df, errors = collect_dashboard_data()

    with st.expander("Show/Hide Data Pull Warnings"):
        if errors:
            for err in errors:
                st.warning(err)
        else:
            st.info("No errors detected.")

    regions = sorted(list(dashboard_df['Region'].dropna().unique()))
    asset_classes = sorted(list(dashboard_df['Asset Class'].dropna().unique()))
    region_filter = st.multiselect("Filter by Region", regions, default=regions)
    class_filter = st.multiselect("Filter by Asset Class", asset_classes, default=asset_classes)

    filtered_df = dashboard_df[
        dashboard_df['Region'].isin(region_filter) &
        dashboard_df['Asset Class'].isin(class_filter)
    ]

    sort_columns = ['Perf 1D', 'Perf 1W', 'Perf 1M', 'Perf 3M', 'Perf 6M',
                    'Perf 1Y', 'Perf 2Y', 'Perf 3Y', 'Market Cap', 'Current Price']
    sort_column = st.selectbox("Sort by", sort_columns, index=0)
    ascending = st.checkbox("Sort Ascending", value=False)
    filtered_df = filtered_df.sort_values(by=sort_column, ascending=ascending, na_position='last')

    st.dataframe(filtered_df, width=3000)
    
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download filtered data as CSV", csv, "dashboard_export.csv", "text/csv")

    # Charting
    st.header("Interactive Price Chart")
    chart_asset_name = st.selectbox("Select asset for chart", filtered_df['Name'].unique())

    # Retrieve the corresponding symbol for that Name
    chart_asset = filtered_df.loc[filtered_df['Name'] == chart_asset_name, 'Symbol'].iloc[0] if 'Symbol' in filtered_df.columns else None

    chart_period = st.selectbox("Historical period", ['1mo','3mo','6mo','1y','2y','3y','5y','max'], index=3)
    chart_data = yf.Ticker(chart_asset).history(period=chart_period)


    if not chart_data.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=chart_data.index,
            y=chart_data['Close'],
            mode='lines',
            name=chart_asset
        ))
        fig.update_layout(
            title=f"{chart_asset_name} Price History ({chart_period})",
            xaxis_title="Date",
            yaxis_title="Price (LCL)"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.download_button(
            "Download chart data as CSV",
            chart_data.to_csv(index=True),
            f"{chart_asset}_{chart_period}_history.csv",
            "text/csv"
        )
    else:
        st.info("No historical data available for the selected asset and period.")

    st.caption("Data sourced from Yahoo Finance via yfinance. Some indices or assets may have incomplete data or no coverage.")


# To launch the dashboard with Streamlit, call:
if __name__ == "__main__":
    run_dashboard()
