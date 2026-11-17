# India Air Quality Monitoring and Environmental Analysis Report

Generated: 2026-11-03 12:45

## 1. Dataset overview

Data source: Live OpenAQ API data

Records: 2841 | Cities: 2 | Period: 2016-02-05 to 2026-11-02 | Pollutants: PM2.5, PM10, NO2, SO2, CO, O3

## 2. Data cleaning

Rows before/after: 2841 / 2841. Duplicates removed: 0. Same-day rows merged: 0. Invalid values set to missing: 1. Missing-value method: interpolate. Outlier action: flag (IQR k=1.5). Outliers detected: {'PM2.5': 218, 'PM10': 160, 'NO2': 54, 'SO2': 70, 'CO': 72, 'O3': 138}

| index | Missing before | Missing after |
| --- | --- | --- |
| PM2.5 | 37 | 0 |
| PM10 | 31 | 0 |
| NO2 | 21 | 0 |
| SO2 | 442 | 0 |
| CO | 175 | 0 |
| O3 | 168 | 0 |

## 3. Descriptive statistics

| index | Count | Mean | Median | Mode | Min | Max | Std Dev | Variance | Q1 (25%) | Q2 (50%) | Q3 (75%) | IQR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PM2.5 | 2841.0 | 97.14 | 58.15 | 36.0 | 1.94 | 985.0 | 120.02 | 14405.96 | 28.0 | 58.15 | 109.3 | 81.3 |
| PM10 | 2841.0 | 164.48 | 118.0 | 107.0 | 0.0 | 1560.0 | 148.41 | 22025.17 | 60.2 | 118.0 | 222.5 | 162.3 |
| NO2 | 2841.0 | 43.11 | 31.84 | 11.4 | 0.33 | 446.0 | 35.9 | 1289.11 | 14.7 | 31.84 | 61.55 | 46.85 |
| SO2 | 2841.0 | 36.02 | 32.2 | 43.6 | 0.0 | 1973.78 | 49.6 | 2460.46 | 14.23 | 32.2 | 43.58 | 29.35 |
| CO | 2841.0 | 0.54 | 0.38 | 0.0 | 0.0 | 12.15 | 0.74 | 0.55 | 0.0 | 0.38 | 0.82 | 0.82 |
| O3 | 2841.0 | 29.89 | 23.29 | 8.8 | 0.0 | 257.8 | 26.3 | 691.78 | 10.06 | 23.29 | 40.95 | 30.9 |

## 4. Correlation matrix (Pearson)

| index | PM2.5 | PM10 | NO2 | SO2 | CO | O3 |
| --- | --- | --- | --- | --- | --- | --- |
| PM2.5 | 1.0 | 0.438 | 0.32 | -0.005 | 0.223 | 0.085 |
| PM10 | 0.438 | 1.0 | 0.559 | 0.091 | 0.322 | 0.256 |
| NO2 | 0.32 | 0.559 | 1.0 | 0.16 | 0.353 | 0.361 |
| SO2 | -0.005 | 0.091 | 0.16 | 1.0 | -0.12 | 0.001 |
| CO | 0.223 | 0.322 | 0.353 | -0.12 | 1.0 | 0.309 |
| O3 | 0.085 | 0.256 | 0.361 | 0.001 | 0.309 | 1.0 |

## 5. Environmental classification

CPCB AQI: sub-index per pollutant by linear interpolation between CPCB breakpoints; AQI is the maximum sub-index; PM2.5/PM10/NO2/SO2 use 24-hour averages, CO/O3 use 8-hour values (daily mean used as approximation for daily data). Strict CPCB rule applied (at least 3 pollutants incl. PM).

| City | Date | AQI | AQI_Category | Dominant_Pollutant |
| --- | --- | --- | --- | --- |
| Delhi | 2026-11-02 | 403.0 | Severe | PM2.5 |
| Mumbai | 2026-11-02 | 70.0 | Satisfactory | PM10 |

## 6. Alerts and recommendations

Most polluted city (latest): Delhi - AQI 403 (Severe), dominant pollutant PM2.5.

Recommendations: Stay indoors and keep windows closed. Avoid all outdoor physical activity. Seek medical advice if you have breathing difficulty, chest pain or persistent cough. Avoid burning waste, crop residue and biomass. Enforce vehicle emission checks and clean fuels. Use covered transport of construction material and water sprinkling.

## 7. Trend analysis (linear regression, all selected cities)

| Pollutant | Trend | Change per 30 days | Change per 30 days (%) | R² | p-value |
| --- | --- | --- | --- | --- | --- |
| PM2.5 | Decreasing | -0.31 | -0.32 | 0.013 | 0.0 |
| PM10 | Decreasing | -0.848 | -0.51 | 0.062 | 0.0 |
| NO2 | Decreasing | -0.243 | -0.57 | 0.084 | 0.0 |
| SO2 | Increasing | 0.163 | 0.52 | 0.044 | 0.0 |
| CO | Decreasing | -0.01 | -1.52 | 0.303 | 0.0 |
| O3 | Decreasing | -0.288 | -0.9 | 0.197 | 0.0 |
