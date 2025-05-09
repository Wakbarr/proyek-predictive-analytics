# Laporan Proyek Machine Learning - Akbar Widianto

## Domain Proyek

Harga cryptocurrency, khususnya Ethereum, dikenal sangat fluktuatif dan dipengaruhi oleh berbagai faktor pasar yang kompleks. Investor dan trader sering kali kesulitan memprediksi pergerakan harga untuk membuat keputusan yang tepat. Analisis tradisional cenderung reaktif, di mana keputusan diambil setelah perubahan harga terjadi. Oleh karena itu, pendekatan prediktif berbasis machine learning diperlukan untuk memberikan wawasan proaktif tentang potensi pergerakan harga di masa depan.

Proyek ini bertujuan membangun model machine learning untuk memprediksi harga penutupan Ethereum 30 hari ke depan berdasarkan data historis. Tiga model yang digunakan adalah K-Nearest Neighbors (KNN), Random Forest, dan AdaBoost, dengan perbandingan performa untuk menemukan algoritma terbaik. Dataset yang digunakan adalah data historis Ethereum dari Kaggle, tersedia di [https://www.kaggle.com/datasets/akbarwidianto/ethereum-historical-data](https://www.kaggle.com/datasets/akbarwidianto/ethereum-historical-data).

## Business Understanding

### Problem Statements

* Bagaimana cara memprediksi harga penutupan Ethereum 30 hari ke depan berdasarkan data historis?
* Model machine learning mana yang paling akurat dalam memprediksi harga Ethereum?
* Fitur apa saja yang paling berpengaruh dalam memprediksi harga Ethereum?

## Goals

* Membangun model machine learning untuk memprediksi harga penutupan Ethereum 30 hari ke depan.
* Membandingkan performa KNN, Random Forest, dan AdaBoost untuk menentukan model terbaik.
* Mengidentifikasi fitur yang paling berpengaruh dalam prediksi harga Ethereum.

## Solution Statement

* Melakukan Exploratory Data Analysis (EDA) untuk memahami karakteristik data dan hubungan antar fitur.
* Menerapkan KNN, Random Forest, dan AdaBoost untuk memprediksi harga Ethereum.
* Mengevaluasi performa model dengan metrik Mean Squared Error (MSE) dan R² Score untuk menentukan model terbaik.

## Data Understanding

Dataset yang digunakan adalah data historis Ethereum dari Kaggle ([https://www.kaggle.com/datasets/akbarwidianto/ethereum-historical-data](https://www.kaggle.com/datasets/akbarwidianto/ethereum-historical-data)), dengan 2160 baris dan 9 kolom. Berikut adalah deskripsi fitur:

| No | Kolom     | Tipe Data | Deskripsi                                  |
| -- | --------- | --------- | ------------------------------------------ |
| 1  | SNo       | int64     | Nomor urut data, sebagai penanda baris     |
| 2  | Name      | object    | Nama koin kripto, dalam hal ini "Ethereum" |
| 3  | Symbol    | object    | Simbol koin, "ETH"                         |
| 4  | High      | float64   | Harga tertinggi pada hari tersebut         |
| 5  | Low       | float64   | Harga terendah pada hari tersebut          |
| 6  | Open      | float64   | Harga pembukaan pada hari tersebut         |
| 7  | Close     | float64   | Harga penutupan pada hari tersebut         |
| 8  | Volume    | float64   | Volume transaksi pada hari tersebut        |
| 9  | Marketcap | float64   | Kapitalisasi pasar pada hari tersebut      |

**Catatan:** Kolom Date tidak ada dalam dataset asli tetapi ditambahkan sebagai indeks saat memuat data dengan `pd.read_csv()` dan `set_index('Date')`.

### Kondisi Data

* **Missing Values:** Tidak ada nilai kosong pada dataset awal.
* **Duplikasi Data:** Tidak ada baris duplikat.
* **Outlier:** Terdapat outlier pada kolom harga (Open, High, Low, Close), yang ditangani pada tahap Data Preparation.

## Univariate Analysis

* **Distribusi Harga Penutupan (Close):** Variasi harga tinggi, distribusi multimodal, mencerminkan volatilitas pasar cryptocurrency.
* **Distribusi Volume:** Variasi signifikan, dengan lonjakan pada periode tertentu.

## Multivariate Analysis

* **Korelasi Antar Fitur:** Kolom harga (Open, High, Low, Close) memiliki korelasi mendekati 1, menunjukkan pergerakan seragam.
* **Hubungan dengan Volume:** Korelasi lemah antara Volume dan harga, menunjukkan pengaruh terbatas dalam jangka pendek.

## Data Preparation

Berikut tahapan pemrosesan data secara runtut:

1. **Pembersihan Data**

   * Menghapus Kolom Tidak Relevan: Kolom SNo, Name, Symbol, Volume, dan Marketcap dihapus karena tidak relevan untuk prediksi.
   * Penanganan Outlier: Outlier dihapus menggunakan metode Interquartile Range (IQR).

2. **Pembuatan Fitur**

   * OHLC\_Average: Rata-rata dari Open, High, Low, dan Close untuk menangkap tren harian.
   * Price\_After\_Month: Target prediksi, dibuat dengan menggeser Close 30 hari ke depan.

3. **Penghapusan Nilai Kosong**

   * Baris dengan nilai kosong (akibat penggeseran Close) dihapus dengan `dropna()`.

4. **Pemisahan Fitur dan Target**

   * Fitur (X): Semua kolom kecuali Price\_After\_Month.
   * Target (y): Price\_After\_Month.

5. **Normalisasi Data**

   * Fitur dinormalisasi dengan StandardScaler untuk skala seragam.

6. **Pembagian Data**

   * Data dibagi: 80% latih, 20% uji, dengan `random_state=42`.

## Modeling

Tiga model machine learning diterapkan:

* **K-Nearest Neighbors (KNN):** n\_neighbors=10.
* **Random Forest:** n\_estimators=50, max\_depth=16, random\_state=42.
* **AdaBoost:** n\_estimators=50, learning\_rate=0.05, random\_state=42.

Setiap model dilatih dan dievaluasi.

## Evaluation

### Metrik Evaluasi

* **Mean Squared Error (MSE)**
* **R² Score**

### Hasil Evaluasi

* **KNN:** MSE = 6100.06, R² = 0.8259
* **Random Forest:** MSE = 7163.01, R² = 0.7955
* **AdaBoost:** MSE = 6203.98, R² = 0.8229

**Analisis:** KNN memiliki MSE terendah dan R² tertinggi, menunjukkan performa terbaik.

## Hubungan dengan Business Understanding

* Prediksi harga 30 hari ke depan tercapai dengan **KNN** sebagai model terbaik.
* Fitur harga (Open, High, Low, Close) paling berpengaruh.
* Investor dapat merencanakan strategi proaktif, meski volatilitas pasar tetap tantangan.

## Visualisasi

* **Prediksi vs Aktual:** Plot 100 data pertama menunjukkan Random Forest mengikuti tren dengan baik.
![image](https://github.com/user-attachments/assets/775c6a77-014d-4537-91ae-4a6d06d161d2)


* **Prediksi 30 Hari:** Visualisasi harga prediksi dengan KNN menunjukkan proyeksi realistis.
![image](https://github.com/user-attachments/assets/5f976074-5d16-4772-a49a-4ba1a826ff0a)


## Kesimpulan

* **KNN** adalah model terbaik untuk memprediksi harga Ethereum 30 hari ke depan.
* Fitur harga (Open, High, Low, Close) sangat berpengaruh.
* Prediksi ini membantu investor, meski pasar sangat fluktuatif.

## Referensi

* Dicoding: [https://www.dicoding.com/academies/319-machine-learning-terapan](https://www.dicoding.com/academies/319-machine-learning-terapan)
* Kaggle Dataset: [https://www.kaggle.com/datasets/akbarwidianto/ethereum-historical-data](https://www.kaggle.com/datasets/akbarwidianto/ethereum-historical-data)
