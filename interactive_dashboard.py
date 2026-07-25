"""
FOOD DELIVERY — INTERACTIVE DASHBOARD (Streamlit + Plotly)
Recreates the Power BI report (food_order.pbix) as a live, filterable web app.

SETUP (run once):
    pip install streamlit plotly pandas numpy scikit-learn

RUN:
    streamlit run interactive_dashboard.py
(make sure food_orders_new_delhi.csv is in the same folder)
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(page_title="Food Delivery Dashboard", layout="wide")

# =====================================================================
# LOAD + CLEAN (cached so filters don't reprocess the raw file each time)
# =====================================================================
@st.cache_data
def load_data():
    df = pd.read_csv('food_orders_new_delhi.csv')
    df['Order Date and Time'] = pd.to_datetime(df['Order Date and Time'])
    df['Delivery Date and Time'] = pd.to_datetime(df['Delivery Date and Time'])
    df['Discounts and Offers'] = df['Discounts and Offers'].fillna('No Discount')
    df['Delivery Duration (min)'] = (
        (df['Delivery Date and Time'] - df['Order Date and Time']).dt.total_seconds() / 60
    )
    df['Order Day'] = df['Order Date and Time'].dt.date
    df['Is Refunded'] = df['Refunds/Chargebacks'] > 0
    return df

df_raw = load_data()

# =====================================================================
# SIDEBAR FILTERS — this is the interactivity Power BI slicers give you
# =====================================================================
st.sidebar.header("Filters")

min_d, max_d = df_raw['Order Day'].min(), df_raw['Order Day'].max()
date_range = st.sidebar.date_input("Order date range", (min_d, max_d), min_value=min_d, max_value=max_d)

payment_methods = st.sidebar.multiselect(
    "Payment Method", options=sorted(df_raw['Payment Method'].unique()),
    default=sorted(df_raw['Payment Method'].unique())
)
discount_types = st.sidebar.multiselect(
    "Discounts and Offers", options=sorted(df_raw['Discounts and Offers'].unique()),
    default=sorted(df_raw['Discounts and Offers'].unique())
)

# Apply filters
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_d, end_d = date_range
else:
    start_d, end_d = min_d, max_d

df = df_raw[
    (df_raw['Order Day'] >= start_d) & (df_raw['Order Day'] <= end_d) &
    (df_raw['Payment Method'].isin(payment_methods)) &
    (df_raw['Discounts and Offers'].isin(discount_types))
]

st.title("🍽️ Food Delivery Operations Dashboard — New Delhi")
st.caption(f"Showing {len(df):,} of {len(df_raw):,} orders based on current filters")

if df.empty:
    st.warning("No orders match the current filters — widen your selection in the sidebar.")
    st.stop()

# =====================================================================
# KPI CARDS
# =====================================================================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Delivery Fee", f"{df['Delivery Fee'].sum():,.0f}")
c2.metric("Total Order Value", f"{df['Order Value'].sum():,.0f}")
c3.metric("Total Processing Fee", f"{df['Payment Processing Fee'].sum():,.0f}")
c4.metric("Total Refunds/Chargebacks", f"{df['Refunds/Chargebacks'].sum():,.0f}")

st.divider()

# =====================================================================
# ROW 1: Donut (top customers) | Bar (fee by discount) | Scatter | Pie
# =====================================================================
row1 = st.columns(4)

with row1[0]:
    top_cust = df['Customer ID'].value_counts().head(10).reset_index()
    top_cust.columns = ['Customer ID', 'Orders']
    fig = px.pie(top_cust, names='Customer ID', values='Orders', hole=0.5,
                 title="Order Count — Top 10 Customers")
    st.plotly_chart(fig, use_container_width=True)

with row1[1]:
    disc = df.groupby('Discounts and Offers')['Delivery Fee'].sum().sort_values().reset_index()
    fig = px.bar(disc, x='Delivery Fee', y='Discounts and Offers', orientation='h',
                 title="Total Delivery Fee by Discount Type", color='Delivery Fee',
                 color_continuous_scale='Teal')
    st.plotly_chart(fig, use_container_width=True)

with row1[2]:
    fig = px.scatter(df, x='Delivery Fee', y='Order Value', color='Payment Method',
                      title="Order Value vs Delivery Fee", opacity=0.6)
    st.plotly_chart(fig, use_container_width=True)

with row1[3]:
    fig = px.pie(disc, names='Discounts and Offers', values='Delivery Fee',
                 title="Delivery Fee Share by Discount")
    st.plotly_chart(fig, use_container_width=True)

# =====================================================================
# ROW 2: Line (processing fee by method) | Time trend | Ribbon | Heatmap
# =====================================================================
row2 = st.columns(4)

with row2[0]:
    pm = df.groupby('Payment Method')['Payment Processing Fee'].sum().sort_values().reset_index()
    fig = px.line(pm, x='Payment Method', y='Payment Processing Fee', markers=True,
                   title="Processing Fee by Payment Method")
    st.plotly_chart(fig, use_container_width=True)

with row2[1]:
    daily = df.groupby('Order Day')['Delivery Fee'].sum().reset_index()
    fig = px.area(daily, x='Order Day', y='Delivery Fee', title="Delivery Fee Trend Over Time")
    st.plotly_chart(fig, use_container_width=True)

with row2[2]:
    ribbon = df.groupby('Payment Method')[['Order Value', 'Delivery Fee']].sum().reset_index()
    ribbon_melt = ribbon.melt(id_vars='Payment Method', var_name='Metric', value_name='Total')
    fig = px.bar(ribbon_melt, x='Payment Method', y='Total', color='Metric', barmode='group',
                 title="Order Value & Delivery Fee by Method")
    st.plotly_chart(fig, use_container_width=True)

with row2[3]:
    metrics = ['Order Value', 'Delivery Fee', 'Commission Fee', 'Payment Processing Fee',
               'Refunds/Chargebacks', 'Delivery Duration (min)']
    corr = df[metrics].corr()
    fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r', title="Correlation Matrix")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# =====================================================================
# ROW 3: Machine learning — customer segments + regression
# =====================================================================
st.subheader("Customer Segmentation (K-Means) & Predictive Check")

cust = df.groupby('Customer ID').agg(
    total_order_value=('Order Value', 'sum'),
    avg_order_value=('Order Value', 'mean'),
    total_delivery_fee=('Delivery Fee', 'sum'),
    order_count=('Order ID', 'count'),
).reset_index()

if len(cust) >= 3:
    feats = ['total_order_value', 'avg_order_value', 'total_delivery_fee', 'order_count']
    X = StandardScaler().fit_transform(cust[feats])
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    cust['segment'] = km.fit_predict(X)
    seg_rank = cust.groupby('segment')['avg_order_value'].mean().sort_values().index
    seg_map = {seg_rank[0]: 'Budget', seg_rank[1]: 'Mid-tier', seg_rank[2]: 'Premium'}
    cust['segment_label'] = cust['segment'].map(seg_map)

    row3 = st.columns(2)
    with row3[0]:
        seg_counts = cust['segment_label'].value_counts().reset_index()
        seg_counts.columns = ['Segment', 'Customers']
        fig = px.bar(seg_counts, x='Segment', y='Customers', color='Segment',
                     title="Customer Segments", color_discrete_map={
                         'Budget': '#2E86AB', 'Mid-tier': '#F18F01', 'Premium': '#C73E1D'})
        st.plotly_chart(fig, use_container_width=True)

    with row3[1]:
        fig = px.scatter(cust, x='avg_order_value', y='total_delivery_fee', color='segment_label',
                          title="Segments: Avg Order Value vs Delivery Fee", color_discrete_map={
                              'Budget': '#2E86AB', 'Mid-tier': '#F18F01', 'Premium': '#C73E1D'})
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Not enough distinct customers in the current filter to run clustering.")

# Regression check
reg_features = ['Delivery Fee', 'Commission Fee', 'Payment Processing Fee', 'Delivery Duration (min)']
if len(df) >= 20:
    X_train, X_test, y_train, y_test = train_test_split(
        df[reg_features], df['Order Value'], test_size=0.2, random_state=42)
    lr = LinearRegression().fit(X_train, y_train)
    preds = lr.predict(X_test)
    st.caption(
        f"Regression check — predicting Order Value from fee structure: "
        f"R² = {r2_score(y_test, preds):.3f}, MAE = {mean_absolute_error(y_test, preds):.2f} "
        f"(low R² means fees don't meaningfully predict order value in this data)."
    )

st.divider()
with st.expander("View filtered raw data"):
    st.dataframe(df, use_container_width=True)
