import streamlit as st
import numpy as np
import pandas as pd
import pickle
from PIL import Image
import tensorflow as tf
import io

# ---------------- UI Styling ----------------
st.set_page_config(page_title="Prediksi Harga Mobil Toyota", layout="wide")
st.markdown("""
    <style>
    body {
        background-color: #0a0f3c;
        color: #FFD700;
    }
    .stApp {
        background-color: #0a0f3c;
        color: #FFD700;
    }
    .stButton>button {
        background-color: #FFD700;
        color: #0a0f3c;
        font-weight: bold;
    }
    .stSelectbox>div, .stNumberInput>div {
        background-color: #1c1f4a;
        color: #FFD700;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- Load Model & Data ----------------
model = pickle.load(open('prediksi_hargamobil.sav', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))

with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)
with open('metrics.pkl', 'rb') as f:
    metrics = pickle.load(f)
with open('X_test.pkl', 'rb') as f:
    X_test = pickle.load(f)
with open('y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)

# CNN model untuk deteksi kondisi visual
cnn_model = tf.keras.models.load_model('cnn_model.h5')
class_labels = ['Bagus dan Mulus', 'Layak Pakai', 'perlu perbaikan']

# ---------------- Fungsi Deteksi Visual ----------------
def detect_condition(image_data):
    img = Image.open(image_data).convert('RGB')
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = cnn_model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]
    return predicted_class

# ---------------- Fungsi Format Harga ----------------
def format_price(number):
    return "{:,.0f}".format(number).replace(",", ".")

# ---------------- UI Aplikasi ----------------
st.title('🚗 Prediksi Harga Mobil Toyota Bekas + Deteksi Kondisi Visual')

st.image('mobil.png', use_column_width=True)

with st.container():
    car_models = sorted(list(set(pd.read_csv('toyota.csv')['model'].unique())))
    selected_model = st.selectbox('Model Mobil', car_models)

    transmissions = ['Manual', 'Automatic', 'Semi-Auto']
    selected_transmission = st.selectbox('Transmisi', transmissions)

    fuel_types = ['Petrol', 'Diesel', 'Hybrid', 'Other']
    selected_fuel_type = st.selectbox('Jenis Bahan Bakar', fuel_types)

    col1, col2 = st.columns(2)
    with col1:
        year = st.number_input('Tahun Produksi', min_value=2001, max_value=2024, step=1)
    with col2:
        mileage = st.number_input('Jarak Tempuh (KM)', min_value=0)

# Kamera untuk input gambar mobil
st.subheader("📷 Ambil Gambar Mobil untuk Deteksi Kondisi Visual")
image_file = st.camera_input("Gunakan kamera belakang untuk menangkap gambar mobil")

# ---------------- Tombol Prediksi ----------------
if st.button('🔍 Prediksi Harga & Kondisi Mobil'):
    if year == 0 or mileage == 0 or image_file is None:
        st.warning('Mohon lengkapi semua data input dan ambil gambar mobil!')
    else:
        with st.spinner('Memproses prediksi...'):
            # --- Prediksi Harga Mobil ---
            try:
                cat_features = pd.DataFrame({
                    'model': [selected_model],
                    'transmission': [selected_transmission],
                    'fuelType': [selected_fuel_type]
                })
                encoded_cats = encoder.transform(cat_features)
                num_features = np.array([[year, mileage]])
                X_pred = np.hstack((num_features, encoded_cats))
                prediction = model.predict(X_pred)[0]
                prediction_rupiah = prediction * 19500
                st.success('✅ Prediksi Selesai!')

                st.subheader('💰 Hasil Prediksi Harga')
                st.write('**Prediksi Harga Mobil (IDR):**', f"🟡 Rp {format_price(prediction_rupiah)}")
                st.write(f"📊 MAE: {metrics['mae']:.2f}")
                st.write(f"📊 MAPE: {metrics['mape']:.2f}%")
                st.write(f"📊 Akurasi Model: {metrics['accuracy']:.2f}%")
            except Exception as e:
                st.error(f'❌ Gagal memproses prediksi harga: {str(e)}')

            # --- Prediksi Kondisi Mobil dari Gambar ---
            try:
                condition_result = detect_condition(image_file)
                st.subheader("🔍 Hasil Deteksi Kondisi Visual Mobil")
                st.image(image_file, caption="Gambar Mobil", use_column_width=True)
                st.write(f"🛠️ **Kondisi Mobil Terdeteksi:** {condition_result}")
            except Exception as e:
                st.error(f'❌ Gagal mendeteksi kondisi mobil: {str(e)}')
