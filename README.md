# India Air Quality Monitoring and Environmental Analysis System

A Streamlit dashboard (frontend.py) on top of a reusable Python backend (backend.py).
It loads or fetches air-quality data for Indian cities, cleans it, analyses it, classifies it with the
CPCB National AQI method and studies pollution trends.

## Architecture

```
 User ─► frontend.py (Streamlit pages, charts, buttons)
              │  calls functions only
              ▼
         backend.py ── OpenAQ API v3 (live data, needs API key)
              │      ── CSV / Excel files, data/sample_air_quality.csv
              ▼
   pandas · numpy · scipy · scikit-learn  (cleaning, statistics, AQI, trends)
```

| File | Purpose |
|------|---------|
| backend.py | All logic: API requests, loading, cleaning, statistics, correlation, CPCB AQI, alerts, trends, exports |
| frontend.py | 10 pages: Dashboard, Upload, Live data, Cleaning, Statistics, Correlation, Visualization, Classification, Trends, Reports |
| data/sample_air_quality.csv | SYNTHETIC demo data (6 cities, 2023-2024) with deliberate problems for the cleaning demo |
| reports/ | Generated HTML / Markdown reports |

## Install and run (Windows + VS Code)

1. Install Python 3.10, 3.11 or 3.12 from python.org (tick **Add Python to PATH**).
2. In VS Code: File > Open Folder > `air_quality_project`. Open the terminal with Ctrl + ` (PowerShell).
3. Run:

```
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run frontend.py
```

If PowerShell blocks activation: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again.
Choose the interpreter in VS Code: Ctrl+Shift+P > "Python: Select Interpreter" > the one inside `venv`.
The app opens at http://localhost:8501. Stop it with Ctrl + C.

## OpenAQ API key (for live data)

1. Register for free at https://explore.openaq.org and copy your API key from your account page.
2. Copy `.env.example` to `.env` and write `OPENAQ_API_KEY=your_key`, **or** paste the key in the Live Data page.
3. Restart the app after editing `.env`.

The free plan has rate limits (requests per minute/hour). Keep "stations per city" low (1-2).
Live data is always labelled "LIVE" in the app; the sample dataset is always labelled "SAMPLE".
If OpenAQ has no station or no recent data for a city you will see a message, never invented numbers.

## Suggested workflow for a demo

1. Data Upload > Sample dataset > Load.
2. Data Cleaning > Clean data (show missing values before/after).
3. Statistical Analysis, Correlation Heatmap, Pollution Visualization.
4. Environmental Classification (CPCB AQI, alerts, recommendations).
5. Trend Analysis (regression, moving average, projection, historical vs current).
6. Live Data Collection (with your key), then Reports & Downloads.

## Using real datasets

Upload any CSV/Excel with a date column and pollutant columns. Recognised names include
`Date/Datetime`, `City`, `PM2.5`, `PM10`, `NO2`, `SO2`, `CO`, `O3` (units in brackets are ignored).
Good free sources: Kaggle "Air Quality Data in India (2015-2020)" (`city_day.csv`, free login) and the
CPCB Central Control Room portal (airquality.cpcb.gov.in/ccr). If your CO is in µg/m³, choose that on the upload page.

## Classification method (CPCB National AQI)

* Readings become daily values in standard units (µg/m³; CO in mg/m³).
* Sub-index per pollutant = linear interpolation between CPCB breakpoints; AQI = highest sub-index.
* PM2.5, PM10, NO2, SO2: 24-hour averages. CO, O3: 8-hour values (the daily mean is used as an approximation
  for daily data - a documented limitation).
* AQI is reported only if at least 3 pollutants are available, one of them PM2.5 or PM10 (can be relaxed in the sidebar).
* Categories: Good 0-50, Satisfactory 51-100, Moderate 101-200, Poor 201-300, Very Poor 301-400, Severe 401-500.
* The live snapshot uses instantaneous readings, so its AQI is marked "indicative".

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `streamlit` is not recognised | Activate the venv, then `pip install -r requirements.txt` |
| `ModuleNotFoundError` | Same: activate the venv and reinstall requirements |
| API key rejected (HTTP 401) | Re-copy the key; no spaces or quotes |
| Rate limit message | Wait a minute; use fewer cities/stations or a shorter date range |
| Excel upload fails | `pip install openpyxl` |
