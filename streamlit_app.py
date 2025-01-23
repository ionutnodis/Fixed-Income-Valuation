

"""""
Cascade Research - Fixed Income Securities Valuation and Risk Metrics with Hull-White Model for Short Rates Simulation

This application demonstrates how to value fixed income securities using Python and Streamlit. 
We will calculate the bond price, Macaulay duration, modified duration, convexity, and provide a recommendation based on the market price.
We will also simulate hundreds of short rates paths using the Hull-White model and plot the results with an average trend line using Plotly.
Furthermore, we provide an option to save the simulated rates to a CSV file and download it.

author: @ionutnodis  / Cascade Research ~ Fixed Income Team

"""


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton
import plotly.express as px
import plotly.graph_objects as go
from fredapi import fred
import os

# Application Header
st.title("Cascade Research")
st.header("Fixed Income Securities Valuation and Risk Metrics")

# Sidebar Input Parameters
st.sidebar.header("Input Parameters")
face_value = st.sidebar.number_input("Face Value ($)", min_value=0, value=1000, step=100)
coupon_rate = st.sidebar.number_input("Coupon Rate (%)", min_value=0.0, value=5.0, step=0.10) / 100
maturity = st.sidebar.number_input("Years to Maturity", min_value=1, value=10, step=1)
market_rate = st.sidebar.number_input("Market Yield Rate (%)", min_value=0.0, value=4.0, step=0.10) / 100
market_price = st.sidebar.number_input("Market Price ($)", min_value=0, value=950, step=10)
coupon_frequency = st.sidebar.selectbox("Coupon Frequency", [1, 2, 4, 6, 12], index=1)
credit_rating = st.sidebar.selectbox("Credit Rating", ["Aaa", "Aa1", "Aa2", "Aa3", "A1", "A2", "A3", "Baa1", "Baa2", "Baa3", "Ba1", "Ba2", "Ba3", "B1", "B2", "B3", "Caa1", "Caa2", "Caa3", "Ca", "C"], index=10)

# Bond Pricing Functions
def bond_price(face_value, coupon_rate, maturity, market_rate, coupon_frequency=2):
    periods = maturity * coupon_frequency
    coupon = (face_value * coupon_rate) / coupon_frequency
    discount_rate = market_rate / coupon_frequency
    cash_flows = [coupon] * (periods - 1) + [coupon + face_value]
    return sum([cf / (1 + discount_rate)**(t + 1) for t, cf in enumerate(cash_flows)])

def macaulay_duration(face_value, coupon_rate, maturity, market_rate, coupon_frequency=2):
    periods = maturity * coupon_frequency
    coupon = (face_value * coupon_rate) / coupon_frequency
    discount_rate = market_rate / coupon_frequency
    cash_flows = [coupon] * (periods - 1) + [coupon + face_value]
    weights = [(t + 1) * cf / (1 + discount_rate)**(t + 1) for t, cf in enumerate(cash_flows)]
    bond_price_value = bond_price(face_value, coupon_rate, maturity, market_rate, coupon_frequency)
    return sum(weights) / bond_price_value

def modified_duration(macaulay_duration, market_rate, coupon_frequency=2):
    return macaulay_duration / (1 + market_rate / coupon_frequency)

def convexity(face_value, coupon_rate, maturity, market_rate, coupon_frequency=2):
    periods = maturity * coupon_frequency
    coupon = (face_value * coupon_rate) / coupon_frequency
    discount_rate = market_rate / coupon_frequency
    cash_flows = [coupon] * (periods - 1) + [coupon + face_value]
    convex = sum([(t + 1) * (t + 2) * cf / (1 + discount_rate)**(t + 2) for t, cf in enumerate(cash_flows)])
    bond_price_value = bond_price(face_value, coupon_rate, maturity, market_rate, coupon_frequency)
    return convex / bond_price_value

def adjust_discount_rate(market_rate, credit_rating):
    rating_adjustment = {
        "Aaa": 0.0, "Aa1": 0.001, "Aa2": 0.002, "Aa3": 0.003, "A1": 0.004, "A2": 0.005, "A3": 0.006,
        "Baa1": 0.008, "Baa2": 0.01, "Baa3": 0.012, "Ba1": 0.02, "Ba2": 0.03, "Ba3": 0.04,
        "B1": 0.05, "B2": 0.06, "B3": 0.07, "Caa1": 0.08, "Caa2": 0.09, "Caa3": 0.1, "Ca": 0.12, "C": 0.15
    }
    return market_rate + rating_adjustment.get(credit_rating, 0.0)

def compute_bond_valuation(face_value, coupon_rate, maturity, market_rate, market_price, credit_rating, coupon_frequency=2):
    adjusted_rate = adjust_discount_rate(market_rate, credit_rating)
    bond_price_value = bond_price(face_value, coupon_rate, maturity, adjusted_rate, coupon_frequency)
    mac_duration = macaulay_duration(face_value, coupon_rate, maturity, adjusted_rate, coupon_frequency)
    mod_duration = modified_duration(mac_duration, adjusted_rate, coupon_frequency)
    convexity_value = convexity(face_value, coupon_rate, maturity, adjusted_rate, coupon_frequency)
    price_change = -mod_duration * 0.01 * bond_price_value + 0.5 * convexity_value * (0.01)**2 * bond_price_value

    recommendation = "Buy" if bond_price_value < market_price - 10 else "Sell" if bond_price_value > market_price + 10 else "Hold"

    return {
        "bond_price": bond_price_value,
        "macaulay_duration": mac_duration,
        "modified_duration": mod_duration,
        "convexity": convexity_value,
        "price_change": price_change,
        "recommendation": recommendation
    }

# Calculations
valuation = compute_bond_valuation(face_value, coupon_rate, maturity, market_rate, market_price, credit_rating, coupon_frequency)

# Display Results in Columns
st.header("Bond Valuation Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Bond Price", f"${valuation['bond_price']:.2f}")
col2.metric("Macaulay Duration", f"{valuation['macaulay_duration']:.2f} years")
col3.metric("Modified Duration", f"{valuation['modified_duration']:.2f} years")

col4, col5 = st.columns(2)
col4.metric("Convexity", f"{valuation['convexity']:.2f}")
col5.metric("Recommendation", valuation['recommendation'])

# Yield vs. Price Plot
st.header("Yield vs. Price Sensitivity")
yields = np.linspace(market_rate - 0.01, market_rate + 0.01, 100)
prices = [bond_price(face_value, coupon_rate, maturity, y, coupon_frequency) for y in yields]
fig = px.line(x=yields * 100, y=prices, labels={"x": "Yield (%)", "y": "Price ($)"}, title="Bond Price Sensitivity to Yield Changes")
fig.add_vline(x=market_rate * 100, line_dash="dash", line_color="red", annotation_text="Market Rate")
st.plotly_chart(fig)


# Streamlit header
st.title("Hull-White Model for Short Rates")

# Initialize FRED API
# API_KEY = "eb7110d142550e5c11ff1d58efcface1"
fred = fred.Fred(api_key='eb7110d142550e5c11ff1d58efcface1')

# Access API key from environment variable
# FRED_API_KEY = os.getenv('eb7110d142550e5c11ff1d58efcface1')
# fred = fred.Fred(api_key_file=FRED_API_KEY)

# Fetch data from FRED
@st.cache_data
def fetch_fred_data():
    short_rate_data = fred.get_series("FEDFUNDS")
    one_year_yields = fred.get_series("GS1")
    ten_year_yields = fred.get_series("GS10")
    data = pd.DataFrame({
        "FEDFUNDS": short_rate_data,
        "GS1": one_year_yields,
        "GS10": ten_year_yields
    }).dropna()
    data.index = pd.to_datetime(data.index)
    data.index.name = "Date"
    return data

# Fetch and display data
st.subheader("Real Time FRED Data")
data = fetch_fred_data()
st.write(data)

# Model parameters
st.subheader("Hull-White Model Parameters")
r0 = st.number_input("Current Short Rate (r0)", value=data["FEDFUNDS"].iloc[-1] / 100, format="%.4f")
theta = st.number_input("Mean Reversion Level (theta)", value=(data["GS1"].iloc[-1] + data["GS10"].iloc[-1]) / 200, format="%.4f")
k = st.slider("Mean Reversion Speed (k)", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
rate_changes = data["FEDFUNDS"].pct_change().dropna()
sigma = st.number_input("Volatility (sigma)", value=rate_changes.std(), format="%.4f")

# Hull-White model simulation function
def hull_white(r0, theta, k, sigma, T, num_steps=1000):
    dt = T / num_steps
    rates = [r0]
    for _ in range(num_steps):
        r = rates[-1]
        dr = k * (theta - r) * dt + sigma * np.random.normal() * np.sqrt(dt)
        rates.append(r + dr)
    return rates

# Simulation parameters
st.subheader("Simulation Parameters")
T = st.slider("Time Horizon (Years)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
n = st.slider("Number of Steps", min_value=100, max_value=1000, value=252, step=50)
num_simulations = st.slider("Number of Simulations", min_value=10, max_value=500, value=100, step=10)

# Run simulations
st.subheader("Running Simulations")
simulations = [hull_white(r0, theta, k, sigma, T, n) for _ in range(num_simulations)]

# Save to CSV
simulated_rates = pd.DataFrame(simulations).T
simulated_rates.index.name = "Time"
csv_filename = "simulated_rates.csv"
simulated_rates.to_csv(csv_filename)
st.success(f"Simulated rates saved to {csv_filename}")

# Add a download button for the CSV file
st.subheader("Download Simulation Results")
csv_data = simulated_rates.to_csv(index=True)  # Convert DataFrame to CSV format
st.download_button(
    label="Download CSV",
    data=csv_data,
    file_name="simulated_rates.csv",
    mime="text/csv",
)

# Plot simulations using Plotly
st.subheader("Simulation Results")

fig = go.Figure()

# Time vector
time = np.linspace(0, T, n + 1)

# Add traces for each simulation
for i in range(num_simulations):
    fig.add_trace(go.Scatter(x=time, y=simulations[i], mode='lines',
                             line=dict(color='blue', width=1), opacity=0.5))

# Calculate the average of the simulations at each time step
average_simulation = np.mean(simulations, axis=0)

# Add the average line to the figure
fig.add_trace(go.Scatter(
    x=time,
    y=average_simulation,
    mode='lines',
    line=dict(color='red', width=2),
    name='Average'
))

# Update plot layout
fig.update_layout(
    title="Simulated Short Rates Using Hull-White Model",
    xaxis_title="Time (Years)",
    yaxis_title="Rate (%)",
    showlegend=True,
    template="plotly_white",
    width=800,
    height=600
)

# Display the plot in Streamlit
st.plotly_chart(fig)



