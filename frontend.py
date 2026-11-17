"""
frontend.py - Streamlit dashboard for the
INDIA AIR QUALITY MONITORING AND ENVIRONMENTAL ANALYSIS SYSTEM

This file only builds the user interface. Every calculation is done by a
function in backend.py (imported below as `bk`).

Run with:   streamlit run frontend.py
"""

from datetime import date, timedelta

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st

import backend as bk

# --------------------------------------------------------------------------
# PAGE SETUP AND STYLE
# --------------------------------------------------------------------------
st.set_page_config(page_title="India Air Quality Monitor", page_icon="🌫️", layout="wide")

st.markdown(
    """
    <style>
    .hero {padding: 18px 24px; border-radius: 14px;
           background: linear-gradient(120deg, #0b5394 0%, #1c8fa3 100%); color: white; margin-bottom: 14px;}
    .hero h1 {color: white; margin: 0; font-size: 1.8rem;}
    .hero p {margin: 4px 0 0 0; opacity: 0.92;}
    .aqi-card {padding: 18px 22px; border-radius: 14px; color: white; margin-bottom: 10px;}
    .aqi-card h2 {color: white; margin: 0; font-size: 2.2rem;}
    .aqi-card p {margin: 2px 0 0 0;}
    div[data-testid="stMetric"] {background: rgba(128,128,128,0.10); border-left: 5px solid #1c8fa3;
                                 padding: 10px 14px; border-radius: 10px;}
    </style>
    """,
    unsafe_allow_html=True,
)

SOURCE_SAMPLE = "Sample dataset (synthetic demo data)"
SOURCE_UPLOAD = "Uploaded file"
SOURCE_URL = "Dataset downloaded from URL"
SOURCE_LIVE = "Live OpenAQ API data"

PAGES = [
    "🏠 Dashboard Overview",
    "📤 Data Upload",
    "🌐 Live Data Collection",
    "🧹 Data Cleaning",
    "📊 Statistical Analysis",
    "🔥 Correlation Heatmap",
    "📈 Pollution Visualization",
    "🌍 Environmental Classification",
    "📉 Trend Analysis",
    "📥 Reports & Downloads",
]
ANALYSIS_PAGES = PAGES[0:1] + PAGES[4:]   # pages that use the cleaned data and sidebar filters

DEFAULT_STATE = {
    "raw_df": None, "clean_df": None, "source": None, "cleaning_report": None,
    "co_unit": "mg/m3", "live_latest": None, "live_long": None, "live_messages": [],
}
for _key, _value in DEFAULT_STATE.items():
    st.session_state.setdefault(_key, _value)


# --------------------------------------------------------------------------
# SMALL HELPERS
# --------------------------------------------------------------------------
def set_dataset(df: pd.DataFrame, source: str, co_unit: str = "mg/m3") -> None:
    """Store a newly loaded dataset; the cleaned version must be recreated."""
    st.session_state.raw_df = df
    st.session_state.clean_df = None
    st.session_state.cleaning_report = None
    st.session_state.source = source
    st.session_state.co_unit = co_unit


def source_badge() -> None:
    """Always show where the data comes from (live API vs sample vs file)."""
    source = st.session_state.source
    if source is None:
        st.info("No dataset loaded yet.")
    elif source == SOURCE_LIVE:
        st.success("🟢 LIVE DATA - real measurements fetched from the OpenAQ API.")
    elif source == SOURCE_SAMPLE:
        st.warning("🟠 SAMPLE DATA - synthetic demonstration data, NOT real measurements.")
    else:
        st.info(f"📁 Data source: {source}")


def get_clean_data():
    """Return the cleaned DataFrame or show a helpful message and return None."""
    if st.session_state.raw_df is None:
        st.info("Load a dataset first: open **📤 Data Upload** or **🌐 Live Data Collection** in the sidebar.")
        return None
    if st.session_state.clean_df is None:
        st.warning("Your dataset has not been cleaned yet.")
        if st.button("🧹 Clean now with default settings", type="primary"):
            run_cleaning("interpolate", "flag", 1.5)
            st.rerun()
        st.caption("Or open **🧹 Data Cleaning** to choose the options yourself.")
        return None
    return st.session_state.clean_df


def run_cleaning(missing_method: str, outlier_action: str, outlier_k: float) -> bool:
    try:
        clean, report = bk.clean_data(st.session_state.raw_df, missing_method=missing_method,
                                      outlier_action=outlier_action, outlier_k=outlier_k,
                                      co_unit=st.session_state.co_unit)
    except ValueError as exc:
        st.error(f"Cleaning failed: {exc}")
        return False
    st.session_state.clean_df = clean
    st.session_state.cleaning_report = report
    return True


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """City and date filters in the sidebar; returns the filtered DataFrame."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔎 Filters")
    signature = f"{st.session_state.source}|{len(df)}|{df['Date'].min()}|{df['Date'].max()}"
    cities = sorted(df["City"].unique())
    chosen = st.sidebar.multiselect("Cities", cities, default=cities, key=f"cities|{signature}")
    d_min, d_max = df["Date"].min().date(), df["Date"].max().date()
    picked = st.sidebar.date_input("Date range", value=(d_min, d_max), min_value=d_min, max_value=d_max,
                                   key=f"dates|{signature}")
    if isinstance(picked, (tuple, list)):
        start, end = (picked[0], picked[1]) if len(picked) == 2 else (picked[0], picked[0])
    else:
        start = end = picked
    mask = (df["City"].isin(chosen) & (df["Date"] >= pd.Timestamp(start))
            & (df["Date"] < pd.Timestamp(end) + pd.Timedelta(days=1)))
    return df[mask].copy()


def style_map(styler, func, subset):
    """pandas renamed Styler.applymap to Styler.map; support both."""
    return styler.map(func, subset=subset) if hasattr(styler, "map") else styler.applymap(func, subset=subset)


def colour_category(value):
    colour = bk.CATEGORY_COLORS.get(value, "#9E9E9E")
    return f"background-color: {colour}; color: white; font-weight: bold"


def show_classification_table(table: pd.DataFrame) -> None:
    view = table.copy()
    if "Date" in view.columns:
        view["Date"] = pd.to_datetime(view["Date"]).dt.strftime("%Y-%m-%d")
    styled = style_map(view.style, colour_category, ["AQI_Category"])
    st.dataframe(styled, hide_index=True)


@st.cache_data(show_spinner=False)
def cached_daily(df: pd.DataFrame) -> pd.DataFrame:
    return bk.prepare_daily(df)


@st.cache_data(show_spinner=False)
def cached_classification(daily: pd.DataFrame, strict: bool, as_of=None) -> pd.DataFrame:
    return bk.city_classification(daily, strict, as_of)


def aqi_bar_chart(table: pd.DataFrame):
    data = table.dropna(subset=["AQI"])
    fig = px.bar(data, x="City", y="AQI", color="AQI_Category", text="AQI",
                 color_discrete_map=bk.CATEGORY_COLORS,
                 category_orders={"AQI_Category": bk.CATEGORY_ORDER},
                 title="AQI by city (CPCB scale 0-500)")
    fig.update_traces(textposition="outside")
    return fig


def pick_pollutant(df: pd.DataFrame, label: str = "Pollutant", key: str = "pollutant") -> str:
    options = bk.available_pollutants(df)
    return st.selectbox(label, options, key=key,
                        format_func=lambda p: f"{p} - {bk.POLLUTANT_NAMES[p]}")


# --------------------------------------------------------------------------
# PAGE 1: DASHBOARD OVERVIEW
# --------------------------------------------------------------------------
def page_dashboard():
    st.markdown(
        "<div class='hero'><h1>🌫️ India Air Quality Monitoring &amp; Environmental Analysis System</h1>"
        "<p>Upload data or fetch live readings, clean it, analyse it and classify air quality "
        "using CPCB standards.</p></div>", unsafe_allow_html=True)
    source_badge()

    if st.session_state.raw_df is None:
        st.subheader("Get started in 3 steps")
        st.markdown("1. **Load data** - upload a CSV/Excel file, fetch live data from OpenAQ, or use the sample dataset.\n"
                    "2. **Clean it** - open *Data Cleaning* and press the button.\n"
                    "3. **Explore** - statistics, charts, AQI classification and trends.")
        if st.button("📂 Load the sample dataset to try the app", type="primary"):
            set_dataset(bk.load_sample_dataset(), SOURCE_SAMPLE)
            st.rerun()
        return

    df = get_clean_data()
    if df is None:
        return
    df = sidebar_filters(df)
    if df.empty:
        st.warning("No data for the selected filters.")
        return

    pollutants = bk.available_pollutants(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", f"{len(df):,}")
    c2.metric("Cities", df["City"].nunique())
    c3.metric("Pollutants", len(pollutants))
    c4.metric("Period", f"{df['Date'].min():%d %b %Y} - {df['Date'].max():%d %b %Y}")

    daily = cached_daily(df)
    strict = st.session_state.get("strict_cpcb", True)
    classification = cached_classification(daily, strict)
    valid = classification.dropna(subset=["AQI"])

    if valid.empty:
        st.warning("AQI could not be calculated for this data. CPCB needs at least three pollutants "
                   "including PM2.5 or PM10 (you can relax this in the sidebar).")
    else:
        worst, best = valid.iloc[0], valid.iloc[-1]
        m1, m2, m3 = st.columns(3)
        m1.metric("Most polluted city (latest)", worst["City"], f"AQI {worst['AQI']:.0f} - {worst['AQI_Category']}",
                  delta_color="off")
        m2.metric("Cleanest city (latest)", best["City"], f"AQI {best['AQI']:.0f} - {best['AQI_Category']}",
                  delta_color="off")
        m3.metric("Main pollutant (most polluted city)", str(worst["Dominant_Pollutant"]))
        left, right = st.columns([3, 2])
        with left:
            st.plotly_chart(aqi_bar_chart(valid))
        with right:
            means = df[pollutants].mean().round(2).rename("Mean").reset_index()
            means.columns = ["Pollutant", "Mean"]
            fig = px.bar(means, x="Pollutant", y="Mean", color="Pollutant", title="Average concentration (all selected data)")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig)
    st.subheader("Data preview")
    st.dataframe(df.head(200), hide_index=True)


# --------------------------------------------------------------------------
# PAGE 2: DATA UPLOAD
# --------------------------------------------------------------------------
def page_upload():
    st.header("📤 Data Upload")
    source_badge()
    tab_file, tab_url, tab_sample = st.tabs(["Upload CSV / Excel", "Download from URL", "Sample dataset"])

    with tab_file:
        st.markdown("Your file needs a **date column** and at least one pollutant column "
                    "(`PM2.5, PM10, NO2, SO2, CO, O3`). A `City` column is recommended. "
                    "Common names such as *Datetime*, *PM2.5 (ug/m3)* or *Bangalore* are recognised automatically. "
                    "Files with one row per measurement (columns *parameter* and *value*) also work.")
        co_unit = st.selectbox("Unit of CO in your file", ["mg/m3", "ug/m3"],
                               help="CPCB uses mg/m3. If your CO values are in the thousands, choose ug/m3.")
        uploaded = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
        if uploaded is not None and st.button("Load uploaded file", type="primary"):
            try:
                df = bk.load_dataset(uploaded, uploaded.name)
                bk.standardize_columns(df)       # validates the columns early
            except ValueError as exc:
                st.error(str(exc))
            else:
                set_dataset(df, SOURCE_UPLOAD, co_unit)
                st.success(f"Loaded {len(df):,} rows. Now open **🧹 Data Cleaning**.")

    with tab_url:
        st.markdown("Paste a **direct link** to a CSV/Excel file. It is saved in the `data/` folder.")
        st.caption("Suggested sources: *Air Quality Data in India (2015-2020)* on Kaggle (free login, download "
                   "`city_day.csv` manually and upload it above) or the CPCB Central Control Room data portal "
                   "(airquality.cpcb.gov.in/ccr).")
        url = st.text_input("File URL (https://...)")
        co_unit_url = st.selectbox("Unit of CO in this file", ["mg/m3", "ug/m3"], key="co_unit_url")
        if st.button("Download and load"):
            try:
                with st.spinner("Downloading..."):
                    path = bk.download_dataset(url)
                    df = bk.load_dataset(path)
                    bk.standardize_columns(df)
            except ValueError as exc:
                st.error(str(exc))
            else:
                set_dataset(df, SOURCE_URL, co_unit_url)
                st.success(f"Downloaded {path.name} and loaded {len(df):,} rows.")

    with tab_sample:
        st.warning("The sample dataset is **synthetic** (generated for demonstration: 6 cities, 2023-2024). "
                   "It contains deliberate problems - missing values, duplicates, spikes, inconsistent city names - "
                   "so that the cleaning step has something to show. It is NOT live or real data.")
        if st.button("Load sample dataset", type="primary"):
            set_dataset(bk.load_sample_dataset(), SOURCE_SAMPLE)
            st.success("Sample dataset loaded. Now open **🧹 Data Cleaning**.")

    raw = st.session_state.raw_df
    if raw is not None:
        st.markdown("---")
        st.subheader("Raw data preview")
        a, b = st.columns(2)
        a.metric("Rows", f"{len(raw):,}")
        b.metric("Columns", raw.shape[1])
        st.dataframe(raw.head(100), hide_index=True)
        with st.expander("Column types"):
            st.dataframe(raw.dtypes.astype(str).rename("Type").reset_index().rename(columns={"index": "Column"}),
                         hide_index=True)


# --------------------------------------------------------------------------
# PAGE 3: LIVE DATA COLLECTION
# --------------------------------------------------------------------------
def page_live():
    st.header("🌐 Live Data Collection (OpenAQ API)")
    st.info("Everything fetched on this page is **real data from OpenAQ**. Nothing is generated or simulated. "
            "If a station has no recent data, you will see a message instead of invented numbers.")
    with st.expander("How to get a free OpenAQ API key"):
        st.markdown(
            "1. Create a free account at **explore.openaq.org** (Register).\n"
            "2. Open your account/settings page and copy your **API key**.\n"
            "3. Paste it below, **or** save it permanently: copy `.env.example` to `.env` and write "
            "`OPENAQ_API_KEY=your_key_here`, then restart the app.")
    api_key = st.text_input("OpenAQ API key", value=bk.get_api_key(), type="password")
    cities = st.multiselect("Cities", list(bk.INDIAN_CITIES), default=["Delhi", "Mumbai"])
    mode = st.radio("What do you want to fetch?",
                    ["Latest readings (current snapshot)", "Historical daily averages"], horizontal=True)
    max_loc = st.slider("Monitoring stations per city", 1, 5, 2,
                        help="More stations = more API calls. The free plan has rate limits, so keep this low.")
    start_date = end_date = None
    if mode.startswith("Historical"):
        c1, c2 = st.columns(2)
        start_date = c1.date_input("From", date.today() - timedelta(days=30))
        end_date = c2.date_input("To", date.today())
        if start_date > end_date:
            st.error("'From' must be before 'To'.")
        elif (end_date - start_date).days > 90:
            st.warning("Long ranges need many API calls and may hit the rate limit. Try 90 days or less.")

    ready = bool(api_key) and bool(cities) and not (start_date and start_date > end_date)
    if st.button("🌐 Fetch live data", type="primary", disabled=not ready):
        progress = st.progress(0.0, text="Starting...")

        def update(fraction: float, message: str) -> None:
            progress.progress(min(max(fraction, 0.0), 1.0), text=message)

        try:
            if mode.startswith("Latest"):
                long_df, messages = bk.fetch_latest(cities, api_key, max_loc, progress=update)
                st.session_state.live_long = long_df
                st.session_state.live_latest = bk.latest_snapshot(long_df)
                st.session_state.live_messages = messages
            else:
                wide, long_df, messages = bk.fetch_history(cities, api_key, start_date, end_date, max_loc,
                                                           progress=update)
                st.session_state.live_long = long_df
                st.session_state.live_messages = messages
                if not wide.empty:
                    set_dataset(wide, SOURCE_LIVE, "mg/m3")
        except bk.OpenAQError as exc:       # unexpected API problem
            st.error(str(exc))
        progress.empty()

    for message in st.session_state.live_messages:
        st.warning(message)

    snapshot = st.session_state.live_latest
    if snapshot is not None and not snapshot.empty:
        st.subheader("Latest readings (average of the selected stations)")
        st.caption("Values are converted to µg/m³ (CO to mg/m³). These are the stations' most recent readings, "
                   "not 24-hour averages.")
        st.dataframe(snapshot, hide_index=True)
        with st.expander("Station-level readings with original units and timestamps"):
            st.dataframe(st.session_state.live_long, hide_index=True)
        st.info("Open **🌍 Environmental Classification → Live snapshot** to classify these readings.")
    elif mode.startswith("Latest") and st.session_state.live_long is not None and not st.session_state.live_messages:
        st.warning("No recent readings were found for the selected cities.")

    if st.session_state.source == SOURCE_LIVE and st.session_state.raw_df is not None:
        st.success(f"Historical live dataset loaded: {len(st.session_state.raw_df):,} city-days. "
                   "Open **🧹 Data Cleaning** to continue.")
        st.dataframe(st.session_state.raw_df.head(200), hide_index=True)


# --------------------------------------------------------------------------
# PAGE 4: DATA CLEANING
# --------------------------------------------------------------------------
MISSING_OPTIONS = {
    "Interpolation (best for time series)": "interpolate",
    "City mean": "mean",
    "City median": "median",
    "Do not fill": "none",
}
OUTLIER_OPTIONS = {
    "Only detect and count (keep values)": "flag",
    "Cap at IQR limits": "cap",
    "Treat as missing, then fill": "remove",
}


def page_cleaning():
    st.header("🧹 Data Cleaning")
    source_badge()
    if st.session_state.raw_df is None:
        st.info("Load a dataset first (📤 Data Upload or 🌐 Live Data Collection).")
        return
    st.markdown("The cleaner renames columns, converts types, tidies city names, removes duplicates, replaces "
                "impossible values (negative or absurdly high) with missing, handles outliers and fills missing values.")
    c1, c2, c3 = st.columns(3)
    missing_label = c1.selectbox("Missing values", list(MISSING_OPTIONS))
    outlier_label = c2.selectbox("Outliers (IQR rule)", list(OUTLIER_OPTIONS))
    k = c3.slider("IQR multiplier", 1.0, 3.0, 1.5, 0.1,
                  help="Values beyond Q1 - k*IQR or Q3 + k*IQR (per city) are outliers. 1.5 is the standard.")
    if st.button("🧹 Clean data", type="primary"):
        if run_cleaning(MISSING_OPTIONS[missing_label], OUTLIER_OPTIONS[outlier_label], k):
            st.success("Cleaning finished.")

    report, clean = st.session_state.cleaning_report, st.session_state.clean_df
    if report is None or clean is None:
        return
    st.subheader("Cleaning summary")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Rows before", f"{report['rows_initial']:,}")
    m2.metric("Rows after", f"{report['rows_after']:,}")
    m3.metric("Duplicates removed", report["duplicates_removed"])
    m4.metric("Same-day rows merged", report["same_day_rows_merged"])
    m5.metric("Invalid values → missing", report["invalid_values_set_missing"])

    st.subheader("Missing values before and after cleaning")
    table = pd.DataFrame({"Before cleaning": report["missing_before"], "After cleaning": report["missing_after"]})
    left, right = st.columns([2, 3])
    left.dataframe(table)
    long = table.reset_index().melt(id_vars="index", var_name="Stage", value_name="Missing values")
    fig = px.bar(long, x="index", y="Missing values", color="Stage", barmode="group",
                 labels={"index": "Pollutant"}, color_discrete_sequence=["#E53935", "#2E9E4F"])
    right.plotly_chart(fig)
    if sum(report["missing_after"].values()) > 0:
        st.caption("Some values remain missing because that pollutant was never measured in a city "
                   "(no values are invented).")

    st.subheader("Outliers detected (IQR rule, per city)")
    st.dataframe(bk.outlier_summary(clean, k=report["outlier_k"]))
    st.caption(f"Action taken: {report['outlier_action']}.")

    st.subheader("Cleaned dataset")
    st.dataframe(clean.head(300), hide_index=True)
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Download cleaned CSV", bk.dataframe_to_csv_bytes(clean),
                       "cleaned_air_quality.csv", "text/csv")
    d2.download_button("⬇️ Download cleaned Excel", bk.dataframe_to_excel_bytes(clean),
                       "cleaned_air_quality.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# --------------------------------------------------------------------------
# PAGE 5: STATISTICAL ANALYSIS
# --------------------------------------------------------------------------
def page_statistics(df: pd.DataFrame):
    st.header("📊 Statistical Analysis")
    source_badge()
    pollutant = pick_pollutant(df, key="stat_pollutant")
    stats_df = bk.descriptive_statistics(df)
    row = stats_df.loc[pollutant]
    unit = bk.POLLUTANT_UNITS[pollutant]
    st.caption(f"Values in {unit}. Mode = most frequent value after rounding to 0.1.")
    a = st.columns(4)
    a[0].metric("Mean", row["Mean"])
    a[1].metric("Median", row["Median"])
    a[2].metric("Mode", row["Mode"])
    a[3].metric("Std deviation", row["Std Dev"])
    b = st.columns(4)
    b[0].metric("Minimum", row["Min"])
    b[1].metric("Maximum", row["Max"])
    b[2].metric("Variance", row["Variance"])
    b[3].metric("IQR", row["IQR"])

    tab1, tab2, tab3, tab4 = st.tabs(["All pollutants", "Quartiles", "City-wise", "Correlation coefficients"])
    with tab1:
        st.dataframe(stats_df)
    with tab2:
        st.dataframe(stats_df[["Min", "Q1 (25%)", "Q2 (50%)", "Q3 (75%)", "Max", "IQR"]])
    with tab3:
        st.dataframe(bk.city_statistics(df, pollutant), hide_index=True)
    with tab4:
        method = st.selectbox("Method", ["pearson", "spearman", "kendall"], key="stat_corr_method")
        st.dataframe(bk.correlation_matrix(df, method=method))


# --------------------------------------------------------------------------
# PAGE 6: CORRELATION HEATMAP
# --------------------------------------------------------------------------
def page_correlation(df: pd.DataFrame):
    st.header("🔥 Correlation Heatmap")
    source_badge()
    pollutants = bk.available_pollutants(df)
    if len(pollutants) < 2:
        st.info("At least two pollutants are needed for a correlation analysis.")
        return
    method = st.radio("Correlation method", ["pearson", "spearman", "kendall"], horizontal=True)
    corr = bk.correlation_matrix(df, method=method)
    tab_plotly, tab_seaborn = st.tabs(["Interactive (Plotly)", "Static (Seaborn)"])
    with tab_plotly:
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        aspect="auto", title=f"{method.capitalize()} correlation between pollutants")
        st.plotly_chart(fig)
    with tab_seaborn:
        fig2, ax = plt.subplots(figsize=(7, 5.5))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5, ax=ax)
        ax.set_title(f"{method.capitalize()} correlation")
        st.pyplot(fig2)
        plt.close(fig2)
    st.subheader("Strongest relationships")
    st.dataframe(bk.strongest_correlations(corr, 10), hide_index=True)
    st.caption("Correlation shows that pollutants rise and fall together (often from a common source such as "
               "traffic or burning); it does not prove that one causes the other.")


# --------------------------------------------------------------------------
# PAGE 7: POLLUTION VISUALIZATION
# --------------------------------------------------------------------------
def page_visualization(df: pd.DataFrame):
    st.header("📈 Pollution Visualization")
    source_badge()
    pollutants = bk.available_pollutants(df)
    cities = sorted(df["City"].unique())
    tabs = st.tabs(["Line trends", "City bar chart", "Histogram", "Box plot (outliers)",
                    "City-wise comparison", "Daily / Weekly / Monthly"])

    with tabs[0]:
        p = pick_pollutant(df, key="viz_line_p")
        c1, c2 = st.columns([3, 1])
        chosen = c1.multiselect("Cities", cities, default=cities[:5], key="viz_line_c")
        freq = c2.selectbox("Aggregation", list(bk.FREQ_MAP), key="viz_line_f")
        ts = bk.time_series(df, p, freq, chosen)
        if ts.empty:
            st.info("No data for this selection.")
        else:
            fig = px.line(ts, x="Date", y=p, color="City", title=f"{p} ({freq.lower()} mean, {bk.POLLUTANT_UNITS[p]})")
            fig.add_hline(y=bk.NAAQS_LIMITS[p], line_dash="dot", line_color="red",
                          annotation_text="NAAQS limit")
            st.plotly_chart(fig)

    with tabs[1]:
        p = pick_pollutant(df, key="viz_bar_p")
        stat = st.selectbox("Statistic", ["mean", "median", "max", "min"], key="viz_bar_s")
        table = bk.city_comparison(df, [p], stat)[p].sort_values(ascending=False).reset_index()
        fig = px.bar(table, x="City", y=p, color=p, color_continuous_scale="YlOrRd", text=p,
                     title=f"{stat.capitalize()} {p} by city ({bk.POLLUTANT_UNITS[p]})")
        st.plotly_chart(fig)

    with tabs[2]:
        p = pick_pollutant(df, key="viz_hist_p")
        c1, c2 = st.columns(2)
        bins = c1.slider("Number of bins", 10, 100, 40)
        split = c2.checkbox("Colour by city", value=False)
        fig = px.histogram(df, x=p, nbins=bins, color="City" if split else None, marginal="box",
                           barmode="overlay", opacity=0.75, title=f"Distribution of {p}")
        st.plotly_chart(fig)

    with tabs[3]:
        p = pick_pollutant(df, key="viz_box_p")
        fig = px.box(df, x="City", y=p, color="City", points="outliers", title=f"Box plot of {p} by city")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig)
        st.markdown("**Outlier count (IQR rule, per city)**")
        st.dataframe(bk.outlier_summary(df))

    with tabs[4]:
        stat = st.selectbox("Statistic", ["mean", "median", "max"], key="viz_cmp_s")
        table = bk.city_comparison(df, pollutants, stat)
        st.dataframe(table)
        norm = bk.normalize_columns(table)
        fig = px.imshow(norm, text_auto=".2f", color_continuous_scale="YlOrRd", aspect="auto",
                        title="Relative pollution level (0 = lowest city, 1 = highest city, per pollutant)")
        st.plotly_chart(fig)
        picks = st.multiselect("Pollutants for the grouped bar chart", pollutants, default=pollutants[:3])
        if picks:
            long = table[picks].reset_index().melt(id_vars="City", var_name="Pollutant", value_name="Value")
            st.plotly_chart(px.bar(long, x="City", y="Value", color="Pollutant", barmode="group",
                                   title="City-wise pollution comparison (native units)"))

    with tabs[5]:
        p = pick_pollutant(df, key="viz_per_p")
        city = st.selectbox("City", ["All Cities"] + cities, key="viz_per_c")
        for freq in bk.FREQ_MAP:
            series = bk.city_series(df, p, city, freq)
            if series.empty:
                continue
            fig = px.line(series.reset_index(), x="Date", y=p, markers=(freq != "Daily"),
                          title=f"{freq} trend of {p} - {city}")
            st.plotly_chart(fig)


# --------------------------------------------------------------------------
# PAGE 8: ENVIRONMENTAL CLASSIFICATION
# --------------------------------------------------------------------------
def methodology_expander():
    with st.expander("📘 Classification methodology (CPCB National AQI)"):
        st.markdown(
            "**How the AQI is calculated**\n"
            "1. Readings are first converted to **daily values** and standard units "
            "(µg/m³; CO in mg/m³).\n"
            "2. Each pollutant gets a **sub-index (0-500)** by linear interpolation between the CPCB breakpoints:  "
            "`Ip = (IHi − ILo)/(BPHi − BPLo) × (Cp − BPLo) + ILo`.\n"
            "3. The **AQI is the highest sub-index**; that pollutant is the *dominant (responsible) pollutant*.\n"
            "4. **Averaging periods matter:** PM2.5, PM10, NO2 and SO2 use 24-hour averages; CO and O3 use 8-hour "
            "values. With daily data the daily mean is used for CO and O3 as an approximation.\n"
            "5. **CPCB validity rule:** an AQI is reported only when at least **three pollutants** are available and "
            "**one of them is PM2.5 or PM10** (switch off in the sidebar to relax).\n"
            "6. Raw concentrations are never compared directly with the AQI category limits (0-500).")
        st.dataframe(bk.breakpoints_table(), hide_index=True)
        legend = " ".join(f"<span style='background:{bk.CATEGORY_COLORS[c]};color:white;padding:3px 10px;"
                          f"border-radius:8px;margin-right:6px'>{c} ({bk.CATEGORY_RANGES[c]})</span>"
                          for c in bk.CATEGORY_ORDER)
        st.markdown(legend, unsafe_allow_html=True)


def show_city_detail(row: pd.Series, note: str = "") -> None:
    """Coloured card, alerts and recommendations for one classification row."""
    category = row["AQI_Category"]
    colour = bk.CATEGORY_COLORS.get(category, "#9E9E9E")
    aqi_text = "-" if pd.isna(row["AQI"]) else f"{row['AQI']:.0f}"
    st.markdown(
        f"<div class='aqi-card' style='background:{colour}'><p>{row['City']} {note}</p>"
        f"<h2>AQI {aqi_text} - {category}</h2>"
        f"<p>{bk.CATEGORY_HEALTH.get(category, '')}</p></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.metric("Pollutant responsible for highest pollution", str(row["Dominant_Pollutant"]) if pd.notna(
        row["Dominant_Pollutant"]) else "n/a")
    c2.metric("Date of reading", f"{pd.to_datetime(row['Date']):%d %b %Y}")
    st.subheader("🚨 Pollution alerts")
    for alert in bk.generate_alerts(row):
        {"error": st.error, "warning": st.warning, "info": st.info, "success": st.success}[alert["level"]](
            alert["message"])
    if category != "No Data":
        recs = bk.get_recommendations(category, row["Dominant_Pollutant"])
        r1, r2 = st.columns(2)
        with r1:
            st.subheader("🩺 Health advice")
            for item in recs["health"]:
                st.markdown(f"- {item}")
        with r2:
            st.subheader(f"🌱 Reducing {row['Dominant_Pollutant']}")
            for item in recs["action"]:
                st.markdown(f"- {item}")


def page_classification(df: pd.DataFrame):
    st.header("🌍 Environmental Classification")
    source_badge()
    methodology_expander()
    strict = st.session_state.get("strict_cpcb", True)
    tab_data, tab_live = st.tabs(["Your dataset (daily CPCB AQI)", "Live snapshot (indicative)"])

    with tab_data:
        daily = cached_daily(df)
        mode = st.radio("Which day to classify?", ["Latest available day per city", "A specific date"], horizontal=True)
        as_of = None
        if mode == "A specific date":
            d_min, d_max = daily["Date"].min().date(), daily["Date"].max().date()
            as_of = st.date_input("Date", value=d_max, min_value=d_min, max_value=d_max, key="class_date")
        table = cached_classification(daily, strict, as_of)
        if table.empty:
            st.info("No data for that selection.")
        else:
            st.subheader("City-wise environmental classification")
            show_cols = ["City", "Date", "AQI", "AQI_Category", "Dominant_Pollutant"] + [
                p for p in bk.POLLUTANTS if p in table.columns]
            show_classification_table(table[show_cols])
            valid = table.dropna(subset=["AQI"])
            if valid.empty:
                st.warning("AQI could not be computed (CPCB rule: ≥3 pollutants including PM2.5 or PM10).")
            else:
                st.plotly_chart(aqi_bar_chart(valid))
                worst = valid.iloc[0]
                st.info(f"Highest pollution: **{worst['City']}** (AQI {worst['AQI']:.0f}, "
                        f"dominant pollutant **{worst['Dominant_Pollutant']}**).")
                focus = st.selectbox("Show details for city", list(valid["City"]), key="class_focus")
                show_city_detail(valid[valid["City"] == focus].iloc[0])

                history = bk.compute_aqi(daily[daily["City"] == focus], strict).dropna(subset=["AQI"])
                if not history.empty:
                    st.subheader(f"AQI history - {focus}")
                    fig = px.line(history, x="Date", y="AQI")
                    bands = [(0, 50, "Good"), (50, 100, "Satisfactory"), (100, 200, "Moderate"),
                             (200, 300, "Poor"), (300, 400, "Very Poor"), (400, 500, "Severe")]
                    for lo, hi, name in bands:
                        fig.add_hrect(y0=lo, y1=hi, fillcolor=bk.CATEGORY_COLORS[name], opacity=0.15, line_width=0)
                    fig.update_yaxes(range=[0, 500])
                    st.plotly_chart(fig)
                    counts = history["AQI_Category"].value_counts().reindex(bk.CATEGORY_ORDER).dropna().reset_index()
                    counts.columns = ["Category", "Days"]
                    st.plotly_chart(px.pie(counts, names="Category", values="Days", color="Category",
                                           color_discrete_map=bk.CATEGORY_COLORS, title=f"Days per AQI category - {focus}"))

    with tab_live:
        snapshot = st.session_state.live_latest
        if snapshot is None or snapshot.empty:
            st.info("No live snapshot yet. Fetch **Latest readings** in 🌐 Live Data Collection first.")
        else:
            st.warning("These are the most recent instantaneous readings from OpenAQ, not 24-hour averages. "
                       "The result is an **indicative** AQI, not an official CPCB value.")
            live_table = bk.city_classification(snapshot.drop(columns=["Source", "Latest_Reading", "Stations"],
                                                              errors="ignore"), strict)
            show_cols = ["City", "Date", "AQI", "AQI_Category", "Dominant_Pollutant"] + [
                p for p in bk.POLLUTANTS if p in live_table.columns]
            show_classification_table(live_table[show_cols])
            for _, row in live_table.iterrows():
                with st.expander(f"{row['City']} - {row['AQI_Category']}"):
                    show_city_detail(row, "(live, indicative)")


# --------------------------------------------------------------------------
# PAGE 9: TREND ANALYSIS
# --------------------------------------------------------------------------
def page_trends(df: pd.DataFrame):
    st.header("📉 Trend Analysis")
    source_badge()
    cities = sorted(df["City"].unique())
    c1, c2, c3, c4 = st.columns(4)
    pollutant = c1.selectbox("Pollutant", bk.available_pollutants(df), key="trend_p")
    city = c2.selectbox("City", ["All Cities"] + cities, key="trend_c")
    freq = c3.selectbox("Aggregation", list(bk.FREQ_MAP), index=2 if len(df) > 400 else 0, key="trend_f")
    window = c4.slider("Moving average window", 2, 30, 7 if freq == "Daily" else 3,
                       help="Number of periods (days, weeks or months).")

    series = bk.city_series(df, pollutant, city, freq)
    if len(series) < 5:
        st.warning("Not enough data points for a trend analysis. Choose Daily aggregation or a wider date range.")
        return
    info = bk.compute_trend(series)
    frame = bk.moving_average_frame(series, window)

    st.subheader("Trend (linear regression)")
    m = st.columns(5)
    m[0].metric("Direction", info["direction"].split(" (")[0])
    m[1].metric("Change per 30 days", f"{info['slope_per_30d']:+.2f} {bk.POLLUTANT_UNITS[pollutant]}")
    m[2].metric("Change per 30 days (%)", f"{info['pct_per_30d']:+.2f}%")
    m[3].metric("R²", f"{info['r2']:.3f}")
    m[4].metric("p-value", f"{info['p_value']:.4f}")
    st.caption("A trend is called increasing/decreasing only when the regression slope is statistically significant "
               "(p < 0.05). A low R² means the straight line explains little of the day-to-day variation.")

    show_forecast = st.checkbox("Show a 30-day linear projection (scikit-learn)", value=False)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame.index, y=frame["Value"], name=pollutant, mode="lines", opacity=0.6))
    fig.add_trace(go.Scatter(x=frame.index, y=frame["Moving_Avg"], name=f"Moving average ({window})",
                             line=dict(width=3)))
    line = bk.trend_line(series)
    if not line.empty:
        fig.add_trace(go.Scatter(x=line.index, y=line.values, name="Regression trend", line=dict(dash="dash", width=3)))
    if show_forecast:
        forecast, r2 = bk.forecast_trend(bk.city_series(df, pollutant, city, "Daily"), 30)
        if forecast.empty:
            st.info("At least 30 daily points are needed for a projection.")
        else:
            fig.add_trace(go.Scatter(x=forecast["Date"], y=forecast["Forecast"], name="Projection",
                                     line=dict(dash="dot", width=3)))
            st.caption(f"Projection model R² on past data: {r2:.2f}. It extends the trend (and yearly cycle if "
                       "the data span a year); it is indicative only.")
    fig.update_layout(title=f"{freq} {pollutant} - {city}", yaxis_title=bk.POLLUTANT_UNITS[pollutant])
    st.plotly_chart(fig)

    left, right = st.columns(2)
    with left:
        st.subheader("Percentage change")
        pct = frame["Pct_Change"].dropna().reset_index()
        pct.columns = ["Date", "Pct_Change"]
        pct["Direction"] = pct["Pct_Change"].apply(lambda v: "Increase" if v > 0 else "Decrease")
        st.plotly_chart(px.bar(pct, x="Date", y="Pct_Change", color="Direction",
                               color_discrete_map={"Increase": "#E53935", "Decrease": "#2E9E4F"},
                               title="Change from previous period (%)"))
    with right:
        st.subheader("Most polluted periods")
        st.dataframe(bk.most_polluted_periods(series, 8), hide_index=True)
        st.subheader("Seasonal pattern")
        profile = bk.monthly_profile(df, pollutant, city)
        st.plotly_chart(px.bar(profile, x="Month", y="Average", title="Average by calendar month"))

    st.subheader("Historical vs current pollution")
    days = st.slider("'Current' = last N days", 7, 90, 30)
    comparison = bk.historical_vs_current(df, pollutant, city, days)
    if comparison is None:
        st.info("Not enough history to compare.")
    else:
        k = st.columns(4)
        k[0].metric(f"Current ({comparison['current_start']} to {comparison['current_end']})",
                    f"{comparison['current_mean']:.2f}")
        k[1].metric("Historical average (before that)", f"{comparison['historical_mean']:.2f}",
                    f"{comparison['pct_diff']:+.1f}% now vs history", delta_color="inverse")
        if "last_year_mean" in comparison:
            k[2].metric("Same period last year", f"{comparison['last_year_mean']:.2f}",
                        f"{comparison['pct_vs_last_year']:+.1f}% vs last year", delta_color="inverse")
        else:
            k[2].metric("Same period last year", "n/a")
        k[3].metric("Unit", bk.POLLUTANT_UNITS[pollutant])
        st.caption("Compare like with like: pollution is seasonal, so the same-period-last-year comparison is fairer "
                   "than comparing winter with the whole-year average.")

    st.subheader("Trend of all pollutants")
    st.dataframe(bk.trend_summary_table(df, city), hide_index=True)


# --------------------------------------------------------------------------
# PAGE 10: REPORTS AND DOWNLOADS
# --------------------------------------------------------------------------
def page_reports(df: pd.DataFrame):
    st.header("📥 Reports & Downloads")
    source_badge()
    strict = st.session_state.get("strict_cpcb", True)
    stats_df = bk.descriptive_statistics(df)
    corr = bk.correlation_matrix(df) if len(bk.available_pollutants(df)) > 1 else None
    classification = cached_classification(cached_daily(df), strict)
    trends = bk.trend_summary_table(df)
    label = st.session_state.source or "Unknown"
    markdown = bk.build_report_markdown(label, df, st.session_state.cleaning_report, stats_df, corr,
                                        classification, trends, strict)
    html_report = bk.build_report_html(label, df, st.session_state.cleaning_report, stats_df, corr,
                                       classification, trends, strict)
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")

    st.subheader("Cleaned dataset (current filters)")
    a, b = st.columns(2)
    a.download_button("⬇️ CSV", bk.dataframe_to_csv_bytes(df), "cleaned_air_quality.csv", "text/csv")
    b.download_button("⬇️ Excel", bk.dataframe_to_excel_bytes(df), "cleaned_air_quality.xlsx",
                      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.subheader("Analysis report")
    c, d = st.columns(2)
    c.download_button("⬇️ Report (HTML)", html_report, f"air_quality_report_{stamp}.html", "text/html")
    d.download_button("⬇️ Report (Markdown)", markdown, f"air_quality_report_{stamp}.md", "text/markdown")
    if st.button("💾 Save a copy in the reports/ folder"):
        path = bk.save_report(html_report, f"air_quality_report_{stamp}.html")
        bk.save_report(markdown, f"air_quality_report_{stamp}.md")
        st.success(f"Saved to {path.parent}")
    with st.expander("Preview report"):
        st.markdown(markdown)


# --------------------------------------------------------------------------
# MAIN: sidebar navigation and routing
# --------------------------------------------------------------------------
def main():
    st.sidebar.title("🌫️ Air Quality Monitor")
    st.sidebar.caption("India Air Quality Monitoring & Environmental Analysis")
    page = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed")
    st.sidebar.markdown("---")
    st.sidebar.checkbox("Strict CPCB AQI rule (≥3 pollutants incl. PM)", value=True, key="strict_cpcb",
                        help="CPCB reports an AQI only if at least three pollutants are available and one is PM2.5 or PM10.")
    source = st.session_state.source
    st.sidebar.caption(f"Data: {source if source else 'none loaded'}")

    if page == PAGES[0]:
        page_dashboard()
    elif page == PAGES[1]:
        page_upload()
    elif page == PAGES[2]:
        page_live()
    elif page == PAGES[3]:
        page_cleaning()
    else:
        df = get_clean_data()
        if df is None:
            return
        df = sidebar_filters(df)
        if df.empty:
            st.warning("No data for the selected filters. Adjust the cities or date range in the sidebar.")
            return
        if page == PAGES[4]:
            page_statistics(df)
        elif page == PAGES[5]:
            page_correlation(df)
        elif page == PAGES[6]:
            page_visualization(df)
        elif page == PAGES[7]:
            page_classification(df)
        elif page == PAGES[8]:
            page_trends(df)
        elif page == PAGES[9]:
            page_reports(df)


main()
