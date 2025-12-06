import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
import yfinance as yf  

# ---------------------------------------------------------
# 1. CONFIGURATION & STYLE
# ---------------------------------------------------------
st.set_page_config(
    page_title="GOOG Stock Predictor | LSTM",
    page_icon="📈",
    layout="centered"
)

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
    try:
        model = tf.keras.models.load_model('my_lstm_model.h5')
        scaler = joblib.load('my_scaler.pkl')
        return model, scaler
    except Exception as e:
        return None, None

model, scaler = load_artifacts()

if model is None or scaler is None:
    st.error("Fatal Error: Could not load model or scaler. Check your .h5 and .pkl files.")
    st.stop()

# ---------------------------------------------------------
# 3. SIDEBAR
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
        - yfinance API (Real-time Data)
        """
    )
    st.write("---")
    st.caption("Developed by Duy NC")

# ---------------------------------------------------------
# 4. MAIN INTERFACE
# ---------------------------------------------------------
st.title("📈 GOOG Price Trajectory Model")
st.markdown("### Neural Network Inference Engine")

col1, col2 = st.columns([1, 2])

try:
    live_data = yf.Ticker("GOOG").history(period="1d")
    default_price = float(live_data['Close'].iloc[-1])
except:
    default_price = 175.00 # Fallback if API fail

with col1:
    st.subheader("Input Parameters")
    current_price = st.number_input(
        "Current Closing Price ($)", 
        min_value=0.0, 
        max_value=5000.0, 
        value=default_price,
        step=0.5
    )
    
    steps_to_predict = st.slider("Days to Forecast", min_value=1, max_value=7, value=3)
    run_btn = st.button("Run Simulation", type="primary")

# ---------------------------------------------------------
# 5. PREDICTION LOGIC
# ---------------------------------------------------------
if run_btn:
    with col2:
        with st.spinner('Fetching live market context & running LSTM...'):           
            try:
                stock_data = yf.Ticker("GOOG")
                hist = stock_data.history(period="6mo")              
                last_99_days = hist['Close'].values[-99:]              
                input_sequence = np.append(last_99_days, current_price)
                input_sequence = input_sequence.reshape(-1, 1)
                
            except Exception as e:
                st.error(f"API Error: Could not fetch context data. {e}")
                st.stop()
            current_seq_scaled = scaler.transform(input_sequence)
            current_seq_scaled = current_seq_scaled.reshape(1, 100, 1)
        
            future_prices = []
            
            for _ in range(steps_to_predict):
                pred_scaled = model.predict(current_seq_scaled, verbose=0)
                pred_price = scaler.inverse_transform(pred_scaled)[0][0]
                future_prices.append(pred_price)
                new_step = pred_scaled.reshape(1, 1, 1)
                current_seq_scaled = np.append(current_seq_scaled[:, 1:, :], new_step, axis=1)

            next_day_price = future_prices[0]
            delta = next_day_price - current_price
            
            st.metric(
                label="Predicted Price (Next Trading Day)", 
                value=f"${next_day_price:.2f}", 
                delta=f"{delta:.2f} ({ (delta/current_price)*100 :.2f}%)"
            )
            
            st.markdown("#### Forecast Trajectory")
            days_labels = ["0 (Today)"] + [f"{i} (+{i} Day)" for i in range(1, steps_to_predict + 1)]
            
            chart_data = pd.DataFrame({
                "Day": days_labels,
                "Price": [current_price] + future_prices
            })
            
            st.line_chart(chart_data.set_index("Day"), color="#4285F4")

else:
    with col2:
        st.info("👈 Enter the latest price (or use the live default) and click Run.")

st.divider()
st.caption("""
**Note:** This model uses a recursive prediction strategy. It feeds its own predictions back into itself to forecast future trends. 
While trained on real historical data, this is an academic demonstration of LSTM architecture capabilities, not financial advice.
Built with TensorFlow, Python, and ☕.
""")
