import pickle
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

# Load model, encoder, feature names, metrics, and test data
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

st.title('Prediksi Harga Mobil Toyota')
st.image('mobil.png', use_column_width=True)

# Input fields
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

def format_price(number):
    return "{:,.0f}".format(number).replace(",", ".")

if st.button('Prediksi Harga'):
    if year == 0 or mileage == 0:
        st.warning('Mohon lengkapi semua data input!')
    else:
        with st.spinner('Memproses prediksi...'):
            try:
                # Prepare categorical features
                categorical_features = pd.DataFrame({
                    'model': [selected_model],
                    'transmission': [selected_transmission],
                    'fuelType': [selected_fuel_type]
                })
                
                # Encode categorical features
                encoded_categorical = encoder.transform(categorical_features)
                
                # Prepare numerical features
                numerical_features = np.array([[year, mileage]])
                
                # Combine features
                prediction_input = np.hstack((numerical_features, encoded_categorical))
                
                # Make prediction
                prediction = model.predict(prediction_input)[0]
                
                # Convert to Rupiah
                prediction_rupiah = prediction * 19500
                
                # Display prediction
                st.success('Prediksi Selesai!')
                st.write('Prediksi Harga Mobil (IDR):', f"Rp {format_price(prediction_rupiah)}")
                
                # Display precomputed metrics
                st.write(f"Mean Absolute Error (MAE): {metrics['mae']:.2f}")
                st.write(f"Mean Absolute Percentage Error (MAPE): {metrics['mape']:.2f}%")
                st.write(f"Akurasi Model: {metrics['accuracy']:.2f}%")
                
            except Exception as e:
                st.error(f'Terjadi kesalahan: {str(e)}')
