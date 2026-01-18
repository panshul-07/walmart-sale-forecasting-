import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Walmart Demand Forecasting",
    layout="wide"
)

st.title("Walmart Demand Forecasting")

# ---------------- DATA LOADING ----------------
@st.cache_data
def load_data():
    """
    Uses REAL CSV only.
    Works locally and on Streamlit Cloud.
    CSV is NOT tracked by Git.
    """

    CSV_PATH = "store_history.csv"

    if not os.path.exists(CSV_PATH):
        st.error(
            "store_history.csv not found.\n\n"
            "• Locally: place CSV in project root\n"
            "• Streamlit Cloud: upload CSV via file manager"
        )
        st.stop()

    df = pd.read_csv(CSV_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


df = load_data()

# ---------------- UI CONTROLS ----------------
st.subheader("Input Parameters & Units")
col1, col2 = st.columns(2)

with col1:
    store_id = st.number_input(
        "Store ID",
        min_value=int(df["Store"].min()),
        max_value=int(df["Store"].max()),
        value=1
    )
    holiday = st.toggle("Holiday Week? (+15% impact)")
    temperature = st.slider("Temperature (°F)", 20, 120, 70)

with col2:
    fuel_price = st.slider("Fuel Price ($/gal)", 2.0, 5.0, 3.5, 0.1)
    cpi = st.slider("Consumer Price Index (CPI)", 200, 300, 220)
    unemployment = st.slider("Unemployment Rate (%)", 3.0, 15.0, 7.0, 0.1)

# ---------------- PREDICTION LOGIC ----------------
store_df = df[df["Store"] == store_id].sort_values("Date")
base_sales = store_df["Weekly_Sales"].mean()

def temp_factor(t):
    return np.exp(-((t - 70) ** 2) / (2 * 40 ** 2))

def predict_sales(t, f, c, u, h):
    return (
        base_sales
        * (1.15 if h else 1.0)
        * temp_factor(t)
        * (1 - (f - 3.5) * 0.05)
        * (1 + (c - 220) * 0.002)
        * (1 - (u - 7) * 0.04)
    )

predicted_sales = predict_sales(
    temperature, fuel_price, cpi, unemployment, holiday
)

# ---------------- METRICS ----------------
st.divider()
m1, m2 = st.columns([2, 1])

with m1:
    st.metric(
        "Predicted Weekly Sales (USD)",
        f"${predicted_sales:,.2f}",
        f"{(predicted_sales - base_sales) / base_sales * 100:.1f}% vs avg"
    )

with m2:
    st.write(f"**Avg Sales:** ${base_sales:,.2f}")
    st.write(f"**Records:** {len(store_df)}")

# ---------------- TREND PLOT ----------------
st.subheader("Recent Sales Trend")
recent = store_df.tail(20)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(recent["Date"], recent["Weekly_Sales"], marker="o", label="Actual")
ax.axhline(predicted_sales, color="red", linestyle="--", label="Prediction")
ax.yaxis.set_major_formatter('${x:,.0f}')
ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig, use_container_width=True)

# ---------------- SENSITIVITY ANALYSIS ----------------
st.divider()
st.subheader("Sensitivity Analysis")

cols = st.columns(2)

def plot_sensitivity(label, x_range, func, current):
    fig, ax = plt.subplots(figsize=(8, 4))
    y = [func(x) for x in x_range]
    ax.plot(x_range, y)
    ax.axvline(current, color="red", linestyle="--")
    ax.set_title(label)
    ax.yaxis.set_major_formatter('${x:,.0f}')
    ax.grid(alpha=0.3)
    st.pyplot(fig, use_container_width=True)

with cols[0]:
    plot_sensitivity(
        "Temperature Impact",
        np.linspace(20, 120, 50),
        lambda x: predict_sales(x, fuel_price, cpi, unemployment, holiday),
        temperature
    )

with cols[1]:
    plot_sensitivity(
        "Fuel Price Impact",
        np.linspace(2, 5, 50),
        lambda x: predict_sales(temperature, x, cpi, unemployment, holiday),
        fuel_price
    )

st.caption("Walmart Demand Forecasting | Real data, production-ready")
