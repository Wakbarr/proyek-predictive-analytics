# Laporan Proyek Machine Learning - Akbar Widianto

## Domain Proyek

Harga cryptocurrency, khususnya Ethereum, sangat fluktuatif dan dipengaruhi oleh berbagai faktor pasar yang kompleks. Para investor dan trader sering kali kesulitan dalam memprediksi pergerakan harga untuk pengambilan keputusan yang tepat. Pendekatan tradisional dalam analisis harga cenderung reaktif, di mana keputusan diambil setelah perubahan harga terjadi. Oleh karena itu, diperlukan solusi prediktif yang dapat memberikan wawasan lebih awal tentang potensi pergerakan harga di masa depan.

Dengan memanfaatkan teknik machine learning, proyek ini bertujuan untuk membangun model yang dapat memprediksi harga Ethereum 30 hari ke depan berdasarkan data historis. Pendekatan ini memungkinkan analisis yang lebih proaktif, membantu investor dalam merencanakan strategi investasi yang lebih baik. Proyek ini menerapkan tiga model machine learning: K-Nearest Neighbors (KNN), Random Forest, dan AdaBoost, untuk membandingkan performa dan menemukan algoritma terbaik dalam memprediksi harga Ethereum.

Dataset yang digunakan adalah data historis Ethereum yang mencakup berbagai metrik harga harian, seperti harga pembukaan, tertinggi, terendah, dan penutupan. Dataset ini bersumber dari Kaggle dan dapat diakses melalui tautan tersebut.

## Business Understanding

### Problem Statements

* Bagaimana cara memprediksi harga penutupan Ethereum 30 hari ke depan berdasarkan data historis?
* Model machine learning mana yang paling akurat dalam memprediksi harga Ethereum?
* Faktor apa saja yang paling berpengaruh dalam memprediksi harga Ethereum?

### Goals

Tujuan yang ingin dicapai dalam proyek ini adalah:

* Membangun model machine learning yang dapat memprediksi harga penutupan Ethereum 30 hari ke depan.
* Membandingkan performa tiga model machine learning (KNN, Random Forest, dan AdaBoost) untuk menentukan model terbaik.
* Mengidentifikasi fitur yang paling berpengaruh dalam memprediksi harga Ethereum.

### Solution Statement

Untuk mencapai tujuan tersebut, langkah-langkah yang akan dilakukan adalah:

1. Melakukan Exploratory Data Analysis (EDA) untuk memahami karakteristik data dan hubungan antar fitur.
2. Menerapkan tiga model machine learning: K-Nearest Neighbors (KNN), Random Forest, dan AdaBoost untuk memprediksi harga Ethereum.
3. Mengevaluasi performa model menggunakan metrik Mean Squared Error (MSE) dan R² Score untuk menentukan model terbaik.

## Data Understanding

Dataset yang digunakan adalah data historis Ethereum yang mencakup periode harian dengan kolom-kolom sebagai berikut:

| No | Kolom     | Tipe Data | Deskripsi                             |
| -- | --------- | --------- | ------------------------------------- |
| 1  | Date      | object    | Tanggal pencatatan data               |
| 2  | Open      | float64   | Harga pembukaan pada hari tersebut    |
| 3  | High      | float64   | Harga tertinggi pada hari tersebut    |
| 4  | Low       | float64   | Harga terendah pada hari tersebut     |
| 5  | Close     | float64   | Harga penutupan pada hari tersebut    |
| 6  | Volume    | float64   | Volume transaksi pada hari tersebut   |
| 7  | Marketcap | float64   | Kapitalisasi pasar pada hari tersebut |

Dataset ini terdiri dari lebih dari 500 sampel data, memenuhi syarat minimum yang ditentukan. Data ini bersih dan tidak mengandung missing values atau duplikasi.

## Univariate Analysis EDA

* **Distribusi Harga Penutupan (Close):** Harga penutupan Ethereum menunjukkan variasi yang signifikan, mencerminkan volatilitas pasar cryptocurrency.
* **Distribusi Volume:** Volume transaksi juga bervariasi, dengan beberapa puncak yang menunjukkan aktivitas perdagangan yang tinggi pada periode tertentu.

## Multivariate Analysis EDA

* **Korelasi Antar Fitur:** Terdapat korelasi kuat antara fitur harga (Open, High, Low, Close), yang menunjukkan bahwa fitur-fitur ini saling berkaitan erat.
* **Hubungan dengan Volume:** Volume transaksi memiliki korelasi yang lebih lemah dengan harga, menunjukkan bahwa volume mungkin tidak secara langsung mempengaruhi pergerakan harga dalam jangka pendek.

## Data Preparation

### Pembersihan Data

* Menghapus kolom yang tidak relevan: SNo, Name, Symbol, Volume, Marketcap.
* Menghapus baris dengan nilai kosong (jika ada).
* Mendeteksi dan menghapus outlier menggunakan metode Interquartile Range (IQR).

### Pembuatan Fitur

* Menambahkan fitur **OHLC\_Average**, yaitu rata-rata dari Open, High, Low, dan Close, untuk menangkap tren harga harian.
* Menentukan target prediksi **Price\_After\_Month**, yaitu harga penutupan 30 hari ke depan, dengan menggeser data kolom Close.

### Normalisasi Data

* Menggunakan StandardScaler untuk menormalisasi fitur agar memiliki skala yang seragam.

### Pembagian Data

* Membagi dataset menjadi 80% data latih dan 20% data uji menggunakan `train_test_split` dari library sklearn.

## Modeling

Pada proyek ini, tiga model machine learning diterapkan untuk memprediksi harga Ethereum:

1. **K-Nearest Neighbors (KNN):** Model berbasis jarak yang sederhana dan efektif untuk data terstruktur.
2. **Random Forest:** Model ensemble yang kuat untuk menangani hubungan non-linear dalam data.
3. **AdaBoost:** Model boosting yang meningkatkan akurasi dengan menggabungkan prediksi dari beberapa estimator lemah.

Setiap model dilatih dengan data latih dan dievaluasi dengan data uji. Parameter yang digunakan untuk masing-masing model adalah:

* **KNN:** `n_neighbors=10`
* **Random Forest:** `n_estimators=50, max_depth=16, random_state=42`
* **AdaBoost:** `n_estimators=50, learning_rate=0.05, random_state=42`

## Evaluation

### Metrik Evaluasi

* **Mean Squared Error (MSE):** Mengukur rata-rata kuadrat error antara prediksi dan nilai aktual.
* **R² Score:** Mengukur seberapa baik model menjelaskan variabilitas data.
![image](https://github.com/user-attachments/assets/b925cbe9-1e71-4cf1-a9a9-5caf1715fc2a)


### Hasil Evaluasi

* **KNN:** MSE = \[6100.06], R² = \[0.8259]
* **Random Forest:** MSE = \[7163.01], R² = \[0.7955]
* **AdaBoost:** MSE = \[6203.98], R² = \[0.8229]

### Visualisasi
![image](https://github.com/user-attachments/assets/47ab3610-43c8-4322-8bbb-5ab69bce5668)


* Plot prediksi vs data aktual untuk 100 data pertama menunjukkan bahwa model Random Forest mampu mengikuti tren harga dengan baik.
  ![image](https://github.com/user-attachments/assets/89cfae3d-303c-4fd0-87b6-0ea690ffe9e6)

  

## Kesimpulan

Berdasarkan analisis dan evaluasi yang dilakukan, dapat disimpulkan bahwa:

* Model **Random Forest** adalah model terbaik untuk memprediksi harga Ethereum 30 hari ke depan berdasarkan data historis.
* Fitur harga (Open, High, Low, Close) memiliki pengaruh signifikan dalam memprediksi harga masa depan.
* Prediksi harga Ethereum dapat membantu investor dalam merencanakan strategi investasi, meskipun tetap perlu diingat bahwa pasar cryptocurrency sangat fluktuatif.

Proyek ini berhasil mengimplementasikan pendekatan machine learning untuk analisis prediktif dalam domain keuangan, khususnya untuk aset cryptocurrency seperti Ethereum.

## Referensi

* Dicoding. Diakses pada \[tanggal] dari [https://www.dicoding.com/academies/319-machine-learning-terapan](https://www.dicoding.com/academies/319-machine-learning-terapan)
* \[Sumber lain yang relevan]
