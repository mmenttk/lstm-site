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
            
            # --- STEP A: GET CONTEXT DATA (Robust Version) ---
            stock_data_values = []
            
            # 1. Try Live API
            try:
                stock = yf.Ticker("GOOG")
                hist = stock.history(period="6mo")
                
                if len(hist) < 99:
                    raise ValueError("API returned insufficient data")
                
                stock_data_values = hist['Close'].values
                # st.success("Connected to Live Market Data") # Optional feedback
                
            except Exception as e:
            # 2. Fallback to CSV if API fails (The Cloud Fix)
                # st.warning("Live feed unstable. Switching to cached data.") # Optional feedback
                try:
                    df = pd.read_csv("goog_fallback.csv")
                    stock_data_values = df['Close'].values
                except:
                    st.error("Critical: Both Live API and Backup Data failed.")
                    st.stop()

            # Now we guarantee we have data. Take the last 99 points.
            last_99_days = stock_data_values[-99:]
            
            # Check length again just to be safe
            if len(last_99_days) != 99:
                st.error(f"Data Error: Expected 99 historical days, got {len(last_99_days)}.")
                st.stop()

            # Append your user input 'current_price' to make it 100
            input_sequence = np.append(last_99_days, current_price)
            
            # Reshape for the scaler (100 rows, 1 column)
            input_sequence = input_sequence.reshape(-1, 1)
        
            # --- STEP B: SCALE & RESHAPE ---
            # Scale the data using the loaded scaler
            current_seq_scaled = scaler.transform(input_sequence)
            
            # Reshape for LSTM: (1 sample, 100 timesteps, 1 feature)
            current_seq_scaled = current_seq_scaled.reshape(1, 100, 1)
            
            # --- STEP C: PREDICT LOOP ---
            future_prices = []
            
            for _ in range(steps_to_predict):
                # 1. Predict next step
                pred_scaled = model.predict(current_seq_scaled, verbose=0)
                
                # 2. Inverse transform to get dollars
                pred_price = scaler.inverse_transform(pred_scaled)[0][0]
                future_prices.append(pred_price)
                
                # 3. Update the sequence (Sliding Window)
                # Remove first element, add new prediction at the end
                new_step = pred_scaled.reshape(1, 1, 1)
                current_seq_scaled = np.append(current_seq_scaled[:, 1:, :], new_step, axis=1)

            # --- STEP D: VISUALIZATION ---
            next_day_price = future_prices[0]
            delta = next_day_price - current_price
            
            st.metric(
                label="Predicted Price (Next Trading Day)", 
                value=f"${next_day_price:.2f}", 
                delta=f"{delta:.2f} ({ (delta/current_price)*100 :.2f}%)"
            )
            
            st.markdown("#### Forecast Trajectory")
            
            chart_data = pd.DataFrame({
                "Day": ["Today"] + [f"+{i} Day(s)" for i in range(1, steps_to_predict + 1)],
                "Price": [current_price] + future_prices
            })
            
            st.line_chart(chart_data.set_index("Day"), color="#4285F4")


st.divider()
st.caption("""
**Note:** This model uses a recursive prediction strategy. It feeds its own predictions back into itself to forecast future trends. 
While trained on real historical data, this is an academic demonstration of LSTM architecture capabilities, not financial advice.
Built with TensorFlow, Python, and ☕.
""")

