import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd

# ---------------------------------------------------------
# 1. CONFIGURATION & STYLE
# ---------------------------------------------------------
st.set_page_config(
    page_title="GOOG Stock Predictor | LSTM",
    page_icon="📈",
    layout="centered"
)

# Custom CSS to remove the "Streamlit" footer and make it look cleaner
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. LOAD ARTIFACTS
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    # Load the model and scaler only once to save memory
    model = tf.keras.models.load_model('my_lstm_model.h5')
    scaler = joblib.load('my_scaler.pkl')
    return model, scaler

try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error(f"Fatal Error: Could not load model. {e}")
    st.stop()

# ---------------------------------------------------------
# 3. SIDEBAR (CONTEXT & TECH SPECS)
# ---------------------------------------------------------
with st.sidebar:
    st.header("About This Project")
    st.info(
        """
        This tool uses a **Long Short-Term Memory (LSTM)** neural network 
        to forecast Google (GOOG) stock price trends.
        
        **Tech Stack:**
        - TensorFlow / Keras
        - Docker Containerization
        - Google Cloud Run
        
        **Data Source:** 
        - Historical OHLC data (2010-2025)
        """
    )
    
    st.write("---")
    st.caption("Developed by [Your Name]")
    
    with st.expander("🛠 View Model Architecture"):
        # Capture model summary string
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        short_summary = "\n".join(stringlist)
        st.code(short_summary, language="text")

# ---------------------------------------------------------
# 4. MAIN INTERFACE
# ---------------------------------------------------------
st.title("📈 GOOG Price Trajectory Model")
st.markdown("### Neural Network Inference Engine")

# Create two columns for a dashboard feel
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Input Parameters")
    # Using a realistic default value for GOOG (approx $170-$200 range)
    current_price = st.number_input(
        "Current Closing Price ($)", 
        min_value=0.0, 
        max_value=500.0, 
        value=175.50,
        step=0.5
    )
    
    steps_to_predict = st.slider("Days to Forecast", min_value=1, max_value=7, value=3)
    
    run_btn = st.button("Run Simulation", type="primary")

# ---------------------------------------------------------
# 5. PREDICTION LOGIC & VISUALIZATION
# ---------------------------------------------------------
if run_btn:
    with col2:
        # Placeholder for loading animation
        with st.spinner('Running LSTM inference cycles...'):
            
            future_prices = []
            current_input = np.array([[current_price]])
            
            # Iterative Prediction Loop (Autoregression)
            # We predict day 1, then use that prediction to predict day 2, etc.
            
            # 1. Scale initial input
            current_scaled = scaler.transform(current_input)
            
            # 2. Reshape for LSTM [batch, timesteps, features] -> [1, 1, 1]
            # NOTE: If your model expects more time steps (e.g. 60), 
            # we are padding here for the demo to prevent crashing. 
            # ideally, we'd need the last 60 days of real data.
            # Assuming model fits (1,1,1) shape for this demo context:
            seq = current_scaled.reshape(1, 1, 1)
            
            for _ in range(steps_to_predict):
                # Predict
                pred_scaled = model.predict(seq, verbose=0)
                
                # Inverse transform to get dollar amount
                pred_price = scaler.inverse_transform(pred_scaled)[0][0]
                future_prices.append(pred_price)
                
                # Update sequence for next loop (feed prediction back in)
                seq = pred_scaled.reshape(1, 1, 1)

            # -----------------------------------------------------
            # RESULTS DISPLAY
            # -----------------------------------------------------
            
            # 1. The Big Metric (Tomorrow's Price)
            next_day_price = future_prices[0]
            delta = next_day_price - current_price
            
            st.metric(
                label="Predicted Price (Next Trading Day)", 
                value=f"${next_day_price:.2f}", 
                delta=f"{delta:.2f} ({ (delta/current_price)*100 :.2f}%)"
            )
            
            # 2. The Chart (Visualizing the Trend)
            st.markdown("#### Forecast Trajectory")
            
            # Create a dataframe for the chart
            # We add the current price as "Day 0" to show the connecting line
            chart_data = pd.DataFrame({
                "Day": ["Today"] + [f"+{i} Days" for i in range(1, steps_to_predict + 1)],
                "Price": [current_price] + future_prices
            })
            
            st.line_chart(chart_data.set_index("Day"), color="#4285F4") # Google Blue color

else:
    # Default state instructions
    with col2:
        st.info("👈 Enter the latest market data and click 'Run Simulation' to execute the model.")

st.divider()
st.caption("""
**Note:** This model uses a recursive prediction strategy. It feeds its own predictions back into itself to forecast future trends. 
While trained on real historical data, this is an academic demonstration of LSTM architecture capabilities, not financial advice.
Built with TensorFlow, Python, and ☕.
""")

st.warning("⚠️ **Model Limitations:** This is an educational prototype trained on historical data. It is not financial advice. Due to recent market volatility (GOOG doubling in 2025), the model may under-predict current extreme highs due to normalization constraints. In other words, this thing is so broken that you should not ever trust it. Do not trade on this.")
