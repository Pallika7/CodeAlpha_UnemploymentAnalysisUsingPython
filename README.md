# India Unemployment Analysis & Predictive Modeling

A condensed data science and predictive analytics project analyzing regional labor market dynamics across India, isolating the socio-economic shocks of COVID-19 lockdowns, and benchmarking predictive machine learning models.

---

## 🎯 Project Objectives & Data Engineering

* **Data Sanitization:** Cleans structural dataset anomalies and transforms raw date records into standardized timelines.
* **Macroeconomic Feature Calculation:** Derives key underlying metrics including total active labor force headcounts, absolute unemployed citizen pools, and working population proxies.
* **Situational Tracking:** Injects a binary lockdown flag to cleanly separate pre-pandemic baselines from the crisis phase.

---

## 📊 Core Analytical Insights

* **Sector Divergence:** Urban labor markets sustained a significantly sharper initial shock than rural zones due to mobility restrictions shutting down service and retail hubs. Rural areas displayed higher natural resilience through outdoor agricultural continuity.
* **Labor Market Breakdown:** The pandemic induced a unique dual collapse where unemployment surged while labor participation dropped dramatically, meaning individuals lost jobs and stopped seeking work simultaneously.
* **Regional Disparity:** Labor market vulnerability is highly non-uniform, with extreme long-term unemployment concentrations in specific states while others maintained strong structural resilience.

---

## 🤖 Predictive Modeling & Performance

The project evaluates two machine learning paradigms to forecast highly volatile economic shifts:

1. **Baseline Model (Random Forest):** Predicts rates using static variables, treating monthly records as entirely isolated snapshots.
2. **Sequential Model (Gradient Boosting):** Integrates a 1-month historical lag feature to capture temporal economic momentum.

### Benchmark Metrics

| Model Configuration | $R^2$ Score | Mean Absolute Error (MAE) |
| :--- | :---: | :---: |
| **Baseline (Random Forest)** | 52.56% | 5.08% |
| **Sequential (Gradient Boosting)** | **58.11%** | **4.41%** |

The sequential architecture significantly minimized error margins, proving that short-term economic momentum is highly predictive. Feature importance rankings show that the situational lockdown flag acts as the dominant driver of market volatility, followed by regional identity and labor participation rates.

---

## 🛠️ Tech Stack & Tools

* **Language:** Python
* **Data Libraries:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn, Power BI *(optional, remove if not used)*
* **Machine Learning:** Scikit-Learn, LightGBM / XGBoost

---
