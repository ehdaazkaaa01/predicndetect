# Aplikasi-Prediksi-Harga-Mobil-main
 Aplikasi Prediksi Harga Mobil Bekas Toyota Menggunakan XGBoost

 **Tujuan**
 
Tujuan dari proyek pembuatan sistem ini adalah yang memprediksi harga penjualan mobil 
bekas terutama merek Toyota berdasarkan kondisi kendaraan yang ingin dijual. Dengan adanya 
sistem ini diharapkan konsumen dapat dengan mudah menentukan harga jual mobil bekasnya cepat 
dan akurat.  

**Data**

Data untuk sistem prediksi harga mobil bekas diambil dari situs auto2000.co.id, dengan total 6738 baris dan 9 kolom. Kolom prediktor meliputi model, tahun, harga, transmisi, jenis bahan bakar, pajak, efisiensi bahan bakar, dan jarak tempuh, sementara kolom target adalah harga penjualan. Sistem ini bertujuan memprediksi harga jual mobil bekas merek Toyota.

![image](https://github.com/user-attachments/assets/1061f99a-7cfd-4402-a744-9e505462ea7e)

**Metode**

Proyek ini menggunakan XGBoost untuk memprediksi harga mobil bekas Toyota, dengan optimasi parameter melalui 5-fold cross-validation. Evaluasi dilakukan menggunakan R² Score, MAE, dan MAPE, menghasilkan prediksi yang akurat dan andal.

**Hasil**

Dataset menghasilkan akurasi 92.03%, MAE 878.27, R² 95.93%, dan MAPE 7.97%. Setelah itu, dibuat tampilan HTML untuk menyajikan hasil tersebut, termasuk potongan kode dan gambar yang menunjukkan metrik akurasi, MAE, MAPE, dan R².

![image](https://github.com/user-attachments/assets/bcd76226-20f7-4886-8162-f78f9dd4cafc)

![image](https://github.com/user-attachments/assets/f987a7bc-92ac-4fdb-b63b-6821276fb1b7)


