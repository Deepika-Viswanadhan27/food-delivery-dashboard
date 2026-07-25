# Food Delivery Operations Dashboard

A Python recreation of a Power BI report (`food_order.pbix`) analyzing 1,000
food delivery orders from New Delhi — built with pandas, numpy, matplotlib,
seaborn, scikit-learn, and statistics, with a live interactive version in
Streamlit + Plotly.

## Files

| File | What it is |
|---|---|
| `food_delivery_dashboard.py` | Static analysis script — cleans data, runs descriptive stats, K-Means clustering, a regression check, and saves `food_delivery_dashboard.png` |
| `interactive_dashboard.py` | Live, filterable web dashboard (Streamlit + Plotly) |
| `food_orders_new_delhi.csv` | Source data (Order ID, Customer ID, Order/Delivery timestamps, fees, payment method, discounts) |
| `requirements.txt` | Python dependencies |

## Run the static version

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python3 food_delivery_dashboard.py
```

This prints descriptive statistics and clustering/regression results to the
console, and saves `food_delivery_dashboard.png` in the same folder.

## Run the interactive version locally

```bash
pip install -r requirements.txt
streamlit run interactive_dashboard.py
```

Opens automatically at `http://localhost:8501`. Use the sidebar to filter by
date range, payment method, and discount type — every chart, KPI card, and
the customer segmentation update live.

## Deploy it publicly (free)

1. Push this folder to a GitHub repository (must include
   `interactive_dashboard.py`, `food_orders_new_delhi.csv`, and
   `requirements.txt`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
3. Click **New app** → pick the repo/branch → set the main file to
   `interactive_dashboard.py` → **Deploy**.
4. You'll get a public URL anyone can open in a browser — no local Python
   install needed on their end.

Notes:
- The free tier is public by default — don't put sensitive real data in the
  CSV if you swap in your own dataset.
- Free-tier apps sleep after inactivity and take a few seconds to wake on
  the next visit.
- Every GitHub push auto-redeploys the app.

## What's in the analysis

- **Cleaning & features** — parsed timestamps, filled missing discount
  values, derived delivery duration, net revenue, and refund flag.
- **Descriptive statistics** — mean/median/mode/stdev/variance for order
  value, delivery fee, commission, processing fee, refunds, and delivery
  duration.
- **K-Means clustering** — segments customers into Budget / Mid-tier /
  Premium groups based on spend and order frequency.
- **Linear regression** — tests whether fee structure predicts order value
  (result: no meaningful linear relationship in this dataset — an honest
  finding, not a modeling error).
- **Visuals** — recreations of all 13 original Power BI visuals (KPI cards,
  donut, bar, pie, scatter, line, ribbon, trend) plus bonus panels
  (correlation heatmap, segmentation charts, distribution plots).
