import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import date

def load_stock_model():
    return load_model('Stocks2_new.h5')

def load_gold_model():
    return load_model('gold_model2.h5')

def fetch_stock_data(stock, start, end):
    return yf.download(stock, start, end)

def preprocess_data(data):
    return data[['Close', 'High', 'Low']]

def scale_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler, scaler.fit_transform(data)

def prepare_test_data(data, train_size):
    data_train = data.iloc[:train_size]
    data_test = data.iloc[train_size:]
    past_100_days = data_train.tail(100)
    return pd.concat([past_100_days, data_test], ignore_index=True)

def predict_stock_prices(model, data_test_scaled, scaler, future_days=30):
    future_predictions = []
    future_dates = pd.date_range(start=date.today(), periods=future_days + 1)[1:]
    X_input = data_test_scaled[-100:].reshape(1, 100, 3)
    
    for _ in range(future_days):
        pred = model.predict(X_input)[0]
        pred *= np.random.uniform(0.98, 1.02, size=pred.shape)
        close_pred = pred[0]
        high_pred = max(pred[1], close_pred * np.random.uniform(1.01, 1.05))
        low_pred = min(pred[2], close_pred * np.random.uniform(0.95, 0.99))
        future_predictions.append([close_pred, high_pred, low_pred])
        new_real_data = np.vstack((data_test_scaled[-99:], pred))
        X_input = new_real_data.reshape(1, 100, 3)
    
    future_predictions = np.array(future_predictions)
    future_predictions = scaler.inverse_transform(future_predictions)
    return pd.DataFrame(future_predictions, columns=['Close', 'High', 'Low'], index=future_dates)

def predict_gold_prices(model, data_test_scaled, scaler, future_days=30):
    future_predictions = []
    X_input = data_test_scaled[-100:].reshape(1, 100, 3)
    
    for _ in range(future_days):
        pred = model.predict(X_input)[0]
        close_pred = pred[0] * np.random.uniform(0.98, 1.02)
        high_pred = max(close_pred, close_pred + 1)
        low_pred = min(close_pred, close_pred - 1)
        future_predictions.append([close_pred, high_pred, low_pred])
        new_real_data = np.vstack((data_test_scaled[-99:], pred))
        X_input = new_real_data.reshape(1, 100, 3)
    
    future_predictions = np.array(future_predictions)
    return scaler.inverse_transform(future_predictions)

def adjust_gold_fluctuation(predictions):
    for i in range(1, len(predictions)):
        predictions[i][0] = predictions[i - 1][0] * 0.9999
    return predictions

def calculate_bond_return(principal, rate, months, compounding=False):
    if compounding:
        return principal * ((1 + (rate / (100 * 12))) ** months)
    return principal + (principal * rate * (months / 12) / 100)

st.set_page_config(page_title="Investment Portfolio Predictor", layout="wide")

st.markdown("<h1 style='text-align: center; color: #FFA500;'>💰 Investment Portfolio Predictor</h1>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    stock_symbol = st.text_input('📈 Stock Symbol:', 'GOOG')
    stock_investment = st.number_input('💵 Stock Investment Amount ($):', min_value=0.0, value=1000.0, step=100.0)

with col2:
    gold_symbol = st.text_input('🏆 Gold Symbol:', 'GC=F')
    gold_investment = st.number_input('💰 Gold Investment Amount ($):', min_value=0.0, value=1000.0, step=100.0)

with col3:
    bond_principal = st.number_input('📜 Bond Investment Amount ($):', min_value=0.0, value=1000.0, step=100.0)
    bond_rate = st.number_input('📊 Bond Interest Rate (%):', min_value=0.0, value=5.0, step=0.1)
    bond_months = st.number_input('⏳ Bond Duration (Months):', min_value=1, value=6, step=1)
    compounding = st.radio('🔄 Compounding?', ['No', 'Yes'])

if st.button('📊 Predict and Calculate'):
    stock_model = load_stock_model()
    gold_model = load_gold_model()

    start, end = '2012-01-01', date.today()
    
    stock_data = fetch_stock_data(stock_symbol, start, end)
    gold_data = fetch_stock_data(gold_symbol, start, end)
    
    stock_data = preprocess_data(stock_data)
    gold_data = preprocess_data(gold_data)
    
    stock_train_size = max(int(len(stock_data) * 0.80), len(stock_data) - 365)
    gold_train_size = max(int(len(gold_data) * 0.80), len(gold_data) - 365)
    
    stock_test = prepare_test_data(stock_data, stock_train_size)
    gold_test = prepare_test_data(gold_data, gold_train_size)
    
    stock_scaler, stock_test_scaled = scale_data(stock_test)
    gold_scaler, gold_test_scaled = scale_data(gold_test)
    
    future_days = 30
    stock_future_df = predict_stock_prices(stock_model, stock_test_scaled, stock_scaler, future_days)
    gold_predictions = predict_gold_prices(gold_model, gold_test_scaled, gold_scaler, future_days)
    gold_predictions = adjust_gold_fluctuation(gold_predictions)
    
    stock_units = stock_investment / stock_future_df.iloc[0]['Close']
    stock_final_value = stock_units * stock_future_df.iloc[-1]['Close']
    
    gold_units = gold_investment / gold_predictions[0][0]
    gold_final_value = gold_units * gold_predictions[-1][0]
    
    bond_final_value = calculate_bond_return(bond_principal, bond_rate, bond_months, compounding == 'Yes')
    
    total_initial = stock_investment + gold_investment + bond_principal
    total_final = stock_final_value + gold_final_value + bond_final_value
    total_return = ((total_final - total_initial) / total_initial) * 100
    
    st.subheader('📊 Investment Summary')
    st.write(f'*Stock Final Value:* ${stock_final_value:,.2f}')
    st.write(f'*Gold Final Value:* ${gold_final_value:,.2f}')
    st.write(f'*Bond Final Value:* ${bond_final_value:,.2f}')
    st.write(f'*Total Final Value:* ${total_final:,.2f}')
    st.write(f'*Total Portfolio Return:* {total_return:.2f}%')
    
    if total_return > 0:
        st.success("📈 Positive Growth Expected!")
    else:
        st.warning("⚠ Possible Loss Expected!")


