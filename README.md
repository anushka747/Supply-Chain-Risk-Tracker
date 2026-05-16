# 🚦 Supply Chain Disruption Risk Tracker

> **Tools:** Python · Pandas · NumPy · Matplotlib · Statistical Anomaly Detection  
> **Domain:** Supply Chain · Operational Risk · Procurement Analytics  
> **Dataset:** 1,256 weekly supplier records · 8 suppliers · 3 years (2022–2024)

---

## 🎯 Business Problem

Supply chain disruptions cost global businesses over **$1.5 trillion annually**. Yet most procurement teams only react *after* a stockout or delay hits production. This project builds a proactive **supplier risk intelligence system** that:

1. **Detects** lead time anomalies before they cascade into stockouts
2. **Scores** every supplier on a composite risk index across 4 dimensions
3. **Quantifies** the exact financial cost of each disruption event
4. **Recommends** targeted mitigation actions by supplier and region

---

## 📊 Key Findings

| Metric | Value |
|---|---|
| Suppliers tracked | **8 across 6 regions** |
| Overall on-time delivery rate | **63.6%** |
| Total disruption cost (3 years) | **$348,499** |
| Annualised disruption cost | **$116,166** |
| Anomaly events detected | **42** |
| Total stockout events | **33** |

### Supplier Risk Scorecard

| Supplier | Region | Risk Score | Risk Tier | On-Time Rate | Disruption Cost |
|---|---|---|---|---|---|
| AlphaComponents | China | 76.2 | 🔴 Critical | 54.8% | $60,550 |
| EpsilonTextiles | Bangladesh | 71.7 | 🔴 Critical | 64.3% | $81,200 |
| EtaChemicals | China | 71.7 | 🔴 Critical | 56.1% | $72,111 |
| ThetaPlastics | Vietnam | 59.4 | 🟠 High | 53.5% | $73,382 |
| DeltaLogistics | USA | 41.7 | 🟠 High | 63.7% | $17,105 |
| GammaParts | India | 31.9 | 🟡 Medium | 75.2% | $19,929 |
| ZetaElectronics | Taiwan | 31.3 | 🟡 Medium | 74.5% | $12,809 |
| BetaMetals | Germany | 15.8 | 🟢 Low | 66.9% | $11,414 |

### Critical Pattern: Q3 Seasonal Disruption Spike
On-time rates drop to **34–40%** every Q3 for Asia-region suppliers — a recurring seasonal risk that is entirely predictable and preventable.

---

## 📁 Project Structure

```
supply-chain-risk-tracker/
│
├── supply_chain_risk.py        # Full analysis pipeline (run end-to-end)
├── requirements.txt            # Python dependencies
│
├── charts/
│   ├── eda_overview.png        # 6-panel EDA dashboard
│   ├── anomaly_detection.png   # Lead time anomaly detection per supplier
│   └── risk_scorecard.png      # Composite risk scores + financial impact
│
└── README.md
```

---

## 🔍 Analysis Sections

### 1. Data Generation & Feature Engineering
- Simulated 1,256 weekly procurement records across 8 suppliers and 6 global regions
- Modelled seasonal disruptions (Q3 Asia spike), random shock events (port strikes, shortages)
- Engineered: `rolling_lt_4w`, `lead_time_vs_promised`, `anomaly` flag, `disruption_cost_usd`

### 2. Anomaly Detection — μ+2σ Threshold Method
- Computed per-supplier mean and standard deviation of lead times
- Flagged 42 anomaly events where lead time exceeded `mean + 2 standard deviations`
- Plotted rolling 4-week average with anomaly threshold overlay for top-risk suppliers

### 3. Composite Risk Scoring
Risk score (0–100) built from 4 weighted dimensions:
- **Delay score** (30%) — average delay days normalised across suppliers
- **Defect score** (25%) — average defect rate
- **Stockout score** (25%) — stockout event frequency
- **Anomaly score** (20%) — rate of statistical lead time anomalies

### 4. Financial Impact Quantification
- Per-event disruption cost = `delay_days × order_value × 0.3%` + `stockout_flag × 12% order_value`
- Rolled up to supplier, region, category, and quarterly level
- Identified top 5 costliest disruption incidents with full audit trail

### 5. SQL-Style Aggregations
- Region-level risk summary (suppliers, on-time rate, avg delay, total cost)
- Month-over-month disruption cost change (last 6 months)
- Top 5 costliest disruption weeks with supplier and event details

---

## 💡 Risk Mitigation Recommendations

| # | Finding | Recommendation |
|---|---|---|
| 1 | 3 Critical-tier suppliers (AlphaComponents, EpsilonTextiles, EtaChemicals) | Immediately dual-source — reduce single-vendor dependency |
| 2 | Q3 on-time rate drops to 34–40% for Asia suppliers every year | Pre-build 6-week safety stock every May before peak disruption |
| 3 | 42 anomaly events mostly concentrated in 3 suppliers | Deploy μ+2σ weekly automated alert to procurement team |
| 4 | China = highest disruption cost region ($132,661) | Accelerate Vietnam/India supplier development as alternatives |
| 5 | Stockout events cost 12% of order value per incident | Enforce SLA contracts with financial penalties for delays > 5 days |

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/anushka747/supply-chain-risk-tracker.git
cd supply-chain-risk-tracker

# Install dependencies
pip install -r requirements.txt

# Run the full analysis
python supply_chain_risk.py
```

Output: full metrics printed to terminal + 3 chart PNGs generated in project directory.

---

## 🛠 Skills Demonstrated

- **Python:** Pandas, NumPy (data wrangling, feature engineering, rolling statistics)
- **Statistical Analysis:** Anomaly detection using μ+2σ threshold method
- **Data Visualization:** Matplotlib (multi-panel dashboards, time-series plots, scatter, heatmaps, donut charts)
- **Risk Modelling:** Composite index design with weighted scoring across 4 dimensions
- **SQL Thinking:** GroupBy aggregations, window operations, segment-level rollups in Pandas
- **Business Framing:** Disruption cost quantification, seasonal pattern detection, procurement recommendations

---

## 👩‍💻 Author

**Anushka Raj** — Data Analyst  
[LinkedIn](https://linkedin.com/in/anushka-raj747) · [GitHub](https://github.com/anushka747) · anushkar747@gmail.com
