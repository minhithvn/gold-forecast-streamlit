import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import datetime
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import plotly.graph_objects as go

# =======================================
# 🔧 TÍNH RSI & MACD
# =======================================
def tinh_RSI(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    RS = gain / loss
    RSI = 100 - (100 / (1 + RS))
    data['RSI'] = RSI
    return data

def tinh_MACD(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    data['MACD'] = short_ema - long_ema
    data['Tín hiệu_MACD'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()
    return data

# =======================================
# 📊 TẢI DỮ LIỆU
# =======================================
def tai_du_lieu(symbol, months=6):
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=30 * months)
    data = yf.download(symbol, start=start, end=end)
    data = tinh_RSI(tinh_MACD(data))
    return data

# =======================================
# 🧠 HUẤN LUYỆN MÔ HÌNH
# =======================================
def train_model(data, symbol):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']])

    x_train, y_train = [], []
    for i in range(60, len(scaled_data)):
        x_train.append(scaled_data[i-60:i])
        y_train.append(scaled_data[i])

    x_train, y_train = np.array(x_train), np.array(y_train)

    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1)

    # Lưu mô hình và scaler
    model.save(f"{symbol}_model.h5")
    joblib.dump(scaler, f"{symbol}_scaler.pkl")

    return model, scaler

# =======================================
# 🔮 DỰ ĐOÁN N NGÀY TIẾP THEO
# =======================================
def du_doan(model, scaler, data, so_ngay):
    last_60 = data[['Close']].tail(60).values
    scaled_last_60 = scaler.transform(last_60)

    X_input = np.array([scaled_last_60])
    predictions = []

    for _ in range(so_ngay):
        pred = model.predict(X_input)[0][0]
        predictions.append(pred)
        scaled_last_60 = np.append(scaled_last_60[1:], [[pred]], axis=0)
        X_input = np.array([scaled_last_60])

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    dates = pd.date_range(start=data.index[-1] + datetime.timedelta(days=1), periods=so_ngay)
    df_pred = pd.DataFrame({'Ngày': dates, 'Giá dự đoán (USD/oz)': predictions.flatten()})
    return df_pred

# =======================================
# 🖥️ GIAO DIỆN STREAMLIT
# =======================================
st.title("📊 DỰ ĐOÁN GIÁ HÀNG HÓA (PRO DASHBOARD)")
st.write("Công cụ dự đoán xu hướng giá Vàng / Bạc / Dầu trong tương lai bằng mô hình học sâu (LSTM).")

symbol = st.selectbox("🔹 Chọn loại hàng hóa:", ["GC=F (Vàng)", "SI=F (Bạc)", "CL=F (Dầu thô)"])
symbol_code = symbol.split(" ")[0]
so_ngay_du_doan = st.slider("📆 Chọn số ngày muốn dự đoán:", 7, 30, 7)
tuy_chon = st.radio("⚙️ Chọn chế độ:", ["Huấn luyện lại mô hình", "Chỉ dự đoán (dùng mô hình đã lưu)"])

if st.button("🚀 Chạy phân tích & dự đoán"):
    data = tai_du_lieu(symbol_code)
    st.subheader(f"📈 Dữ liệu {symbol}: 6 tháng gần nhất")
    st.dataframe(data.tail())

    if tuy_chon == "Huấn luyện lại mô hình":
        model, scaler = train_model(data, symbol_code)
        st.success("✅ Đã huấn luyện mô hình mới!")
    else:
        if os.path.exists(f"{symbol_code}_model.h5") and os.path.exists(f"{symbol_code}_scaler.pkl"):
            model = load_model(f"{symbol_code}_model.h5")
            scaler = joblib.load(f"{symbol_code}_scaler.pkl")
            st.success("✅ Đã tải mô hình đã lưu sẵn!")
        else:
            st.error("❌ Chưa có mô hình lưu sẵn. Vui lòng huấn luyện trước.")
            st.stop()

    df_pred = du_doan(model, scaler, data, so_ngay_du_doan)
    st.subheader(f"📅 Kết quả dự đoán giá {so_ngay_du_doan} ngày tới")
    st.dataframe(df_pred)

    # Lấy giá hiện tại và giá trung bình dự đoán, ép kiểu thành float an toàn
    try:
        gia_hien_tai = float(data['Close'].iloc[-1])
    except Exception:
        gia_hien_tai = None

    try:
        gia_trung_binh_du_doan = float(df_pred['Giá dự đoán (USD/oz)'].mean())
    except Exception:
        gia_trung_binh_du_doan = None

    # Kiểm tra hợp lệ trước khi so sánh
    if (gia_hien_tai is None) or (gia_trung_binh_du_doan is None) or np.isnan(gia_hien_tai) or np.isnan(
            gia_trung_binh_du_doan):
        xu_huong = "Không xác định"
        chenhlech = None
    else:
        xu_huong = "📈 TĂNG" if (gia_trung_binh_du_doan > gia_hien_tai) else "📉 GIẢM"
        chenhlech = round(((gia_trung_binh_du_doan - gia_hien_tai) / gia_hien_tai) * 100, 2)

    # Hiển thị an toàn
    st.markdown(f"""
    ### 💡 Phân tích xu hướng dự đoán
    - **Giá hiện tại:** {gia_hien_tai if gia_hien_tai is not None else 'Không có dữ liệu'}  
    - **Giá trung bình {so_ngay_du_doan} ngày tới:** {gia_trung_binh_du_doan if gia_trung_binh_du_doan is not None else 'Không có dữ liệu'}  
    - **Chênh lệch:** {str(chenhlech) + '%' if chenhlech is not None else 'Không xác định'}  
    - **Dự báo xu hướng:** {xu_huong}
    """)

    st.markdown(f"""
    ### 💡 Phân tích xu hướng dự đoán
    - **Giá hiện tại:** {gia_hien_tai:.2f} USD/oz  
    - **Giá trung bình {so_ngay_du_doan} ngày tới:** {gia_trung_binh_du_doan:.2f} USD/oz  
    - **Thay đổi ước tính:** {chenhlech}%  
    - **Xu hướng dự báo:** {xu_huong}
    """)

    # Biểu đồ nến + đường dự đoán
    st.subheader("📊 Biểu đồ giá & Dự đoán")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=data.index[-90:], open=data['Open'][-90:], high=data['High'][-90:],
        low=data['Low'][-90:], close=data['Close'][-90:],
        name="Giá thực tế"
    ))
    fig.add_trace(go.Scatter(
        x=df_pred['Ngày'], y=df_pred['Giá dự đoán (USD/oz)'],
        mode='lines+markers', name="Giá dự đoán", line=dict(color='red', width=2)
    ))
    fig.update_layout(xaxis_rangeslider_visible=False, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # Phân tích kỹ thuật
    st.subheader("📉 Phân tích kỹ thuật (RSI & MACD)")
    data_recent = data.tail(100)

    # RSI
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=data_recent.index, y=data_recent['RSI'], mode='lines', name='RSI', line=dict(color='orange')))
    fig2.add_hline(y=70, line_dash="dash", line_color="red")
    fig2.add_hline(y=30, line_dash="dash", line_color="green")
    fig2.update_layout(title="Chỉ báo RSI (Sức mạnh tương đối)")
    st.plotly_chart(fig2, use_container_width=True)

    rsi_value = data['RSI'].iloc[-1]
    if rsi_value > 70:
        giai_thich_rsi = "RSI > 70 → **Quá mua**, có thể sắp giảm."
    elif rsi_value < 30:
        giai_thich_rsi = "RSI < 30 → **Quá bán**, có thể phục hồi."
    else:
        giai_thich_rsi = "RSI trong vùng 30–70 → **Ổn định**."
    st.info(f"👉 {giai_thich_rsi} (RSI hiện tại: {rsi_value:.2f})")

    # MACD
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=data_recent.index, y=data_recent['MACD'], mode='lines', name='MACD', line=dict(color='purple')))
    fig3.add_trace(go.Scatter(x=data_recent.index, y=data_recent['Tín hiệu_MACD'], mode='lines', name='Đường tín hiệu', line=dict(color='black', dash='dash')))
    fig3.update_layout(title="Chỉ báo MACD (Hội tụ – Phân kỳ trung bình động)")
    st.plotly_chart(fig3, use_container_width=True)

    macd_val = data['MACD'].iloc[-1]
    signal_val = data['Tín hiệu_MACD'].iloc[-1]
    if macd_val > signal_val:
        giai_thich_macd = "MACD cắt **lên trên** đường tín hiệu → xu hướng **tăng**."
    else:
        giai_thich_macd = "MACD cắt **xuống dưới** đường tín hiệu → xu hướng **giảm**."
    st.info(f"👉 {giai_thich_macd} (MACD: {macd_val:.4f}, Tín hiệu: {signal_val:.4f})")

    # Nút tải xuống CSV
    csv = df_pred.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Tải xuống kết quả CSV", csv, "du_doan_gia.csv", "text/csv")

st.caption("© 2025 Dự đoán giá hàng hóa - Phiên bản PRO Dashboard (Việt hóa đầy đủ)")
from streamlit_autorefresh import st_autorefresh

# Refresh mỗi 10 phút = 600,000 milliseconds
st_autorefresh(interval=10 * 60 * 1000, key="gold_refresh")

st.title("📈 Dự đoán giá vàng - Cập nhật tự động mỗi 10 phút")
