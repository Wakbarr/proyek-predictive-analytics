## Laporan Proyek Machine Learning – Akbar Widianto

**Email:** wdntoakbar@gmail.com  
**ID Dicoding:** wakbarr  

---

## 1. Domain Proyek  
Dalam dunia keuangan digital, cryptocurrency seperti Ethereum (ETH) sangat populer namun fluktuatif. Pergerakan harga yang cepat membuat investor sulit memprediksi arah pasar. Proyek ini memanfaatkan **predictive analytics** dengan machine learning untuk mengidentifikasi pola dari data historis dan memprediksi harga penutupan Ethereum 30 hari ke depan, sehingga membantu perencanaan strategi investasi.

---

## 2. Business Understanding  

### 2.1 Problem Statements  
1. Bagaimana memprediksi harga penutupan Ethereum 30 hari ke depan berdasarkan data historis?  
2. Model machine learning mana yang paling akurat untuk prediksi harga Ethereum?

### 2.2 Goals  
- Mengembangkan model ML yang memprediksi harga penutupan ETH 30 hari mendatang dengan akurasi tinggi.  
- Membandingkan performa tiga model regresi: KNN, Random Forest, dan AdaBoost.

### 2.3 Solution Statement  
1. Lakukan **Exploratory Data Analysis (EDA)** untuk memahami distribusi dan korelasi fitur.  
2. Bangun tiga model regresi: KNN, Random Forest, AdaBoost.  
3. Evaluasi model dengan **MSE** dan **R² Score** untuk memilih model terbaik.

---

## 3. Data Understanding  

- **Sumber dataset**: Data historis harga Ethereum harian (≥ 500 baris).  
- **Kolom relevan**:  
  - `Date` (tanggal)  
  - `Open`, `High`, `Low`, `Close` (harga)  

- **Statistik awal**:  
  - Tidak ada missing values setelah pembersihan.  
  - Outliers dihapus menggunakan metode IQR.

---

## 4. Exploratory Data Analysis (EDA)  

### 4.1 Univariate Analysis  
- **Distribusi Harga**: Histogram `Close` cenderung skewed ke kanan (periode harga tinggi).

### 4.2 Multivariate Analysis  
- **Korelasi Antar Fitur**:  
  - Open, High, Low, Close sangat berkorelasi (> 0.9).  
  - Tambahan fitur `OHLC_Average` (rata-rata Open/High/Low/Close).

---

## 5. Data Preparation  

1. **Pembersihan Data**  
   - Hapus kolom: `SNo`, `Name`, `Symbol`, `Volume`, `Marketcap`.  
   - Hapus baris dengan NaN.  

2. **Fitur & Target**  
   - Buat `OHLC_Average` = rata-rata (`Open`, `High`, `Low`, `Close`).  
   - Target `Price_After_Month` = `Close` yang digeser 30 hari ke depan.  

3. **Outlier Removal**  
   - Deteksi dan hapus menggunakan **Interquartile Range (IQR)**.

4. **Normalisasi**  
   - StandardScaler pada semua fitur.

5. **Split Dataset**  
   - 80% data latih, 20% data uji (random_state=42).

---

## 6. Modeling  

### 6.1 Pendekatan  
Regresi—karena target adalah variabel kontinu (harga).

### 6.2 Model yang Dibandingkan  
| Model           | Parameter Utama                       |
| --------------- | ------------------------------------- |
| **KNN**         | `n_neighbors=10`                      |
| **RandomForest**| `n_estimators=50`, `max_depth=16`     |
| **AdaBoost**    | `n_estimators=50`, `learning_rate=0.05`|

### 6.3 Metrik Evaluasi  
- **Mean Squared Error (MSE)**  
- **R² Score**

---

## 7. Evaluation  

| Model           | MSE       | R² Score  |
| --------------- | --------- | --------- |
| **KNN**         | [6100.06] | [0.8259]  |
| **RandomForest**| [7163.01] | [0.7955]  |
| **AdaBoost**    | [6203.98] | [0.8229]  |

### 7.1 Visualisasi  
- Plot perbandingan **harga aktual vs prediksi**  

- ![image](https://github.com/user-attachments/assets/3709799b-913b-48e9-8f2d-28a23fa6bf02)


## 8. Hasil & Kesimpulan  

- **Model Terbaik**: [sebutkan model dengan MSE terendah & R² tertinggi].  
- Prediksi harga Ethereum 30 hari ke depan menggunakan model terpilih:  
  ![image](https://github.com/user-attachments/assets/6a189f6f-862d-4528-88f2-dc903009fde7)


**Kesimpulan**: Model ML dapat membantu prediksi harga ETH, namun pasar crypto bersifat volatile—hasil prediksi harus dipakai sebagai referensi & bukan kepastian.

---
