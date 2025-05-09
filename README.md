# Laporan Proyek Machine Learning - Akbar Widianto

## Domain Proyek

Harga cryptocurrency, khususnya Ethereum, sangat fluktuatif dan dipengaruhi oleh berbagai faktor pasar yang kompleks. Para investor dan trader sering kali kesulitan dalam memprediksi pergerakan harga untuk pengambilan keputusan yang tepat. Pendekatan tradisional dalam analisis harga cenderung reaktif, di mana keputusan diambil setelah perubahan harga terjadi. Oleh karena itu, diperlukan solusi prediktif yang dapat memberikan wawasan lebih awal tentang potensi pergerakan harga di masa depan.

Dengan memanfaatkan teknik machine learning, proyek ini bertujuan untuk membangun model yang dapat memprediksi harga Ethereum 30 hari ke depan berdasarkan data historis. Pendekatan ini memungkinkan analisis yang lebih proaktif, membantu investor dalam merencanakan strategi investasi yang lebih baik. Proyek ini menerapkan tiga model machine learning: K-Nearest Neighbors (KNN), Random Forest, dan AdaBoost, untuk membandingkan performa dan menemukan algoritma terbaik dalam memprediksi harga Ethereum.

Dataset yang digunakan adalah data historis Ethereum yang mencakup berbagai metrik harga harian, seperti harga pembukaan, tertinggi, terendah, dan penutupan. Dataset ini bersumber dari Kaggle.

## Business Understanding

### Problem Statements

* Bagaimana cara memprediksi harga penutupan Ethereum 30 hari ke depan berdasarkan data historis?
* Model machine learning mana yang paling akurat dalam memprediksi harga Ethereum?
* Fitur apa saja yang paling berpengaruh dalam memprediksi harga Ethereum?

## Goals

Tujuan yang ingin dicapai dalam proyek ini adalah:

* Membangun model machine learning yang dapat memprediksi harga penutupan Ethereum 30 hari ke depan.
* Membandingkan performa tiga model machine learning (KNN, Random Forest, dan AdaBoost) untuk menentukan model terbaik.
* Mengidentifikasi fitur yang paling berpengaruh dalam memprediksi harga Ethereum.

## Solution Statement

Untuk mencapai tujuan tersebut, langkah-langkah yang akan dilakukan adalah:

1. Melakukan Exploratory Data Analysis (EDA) untuk memahami karakteristik data dan hubungan antar fitur.
2. Menerapkan tiga model machine learning: K-Nearest Neighbors (KNN), Random Forest, dan AdaBoost untuk memprediksi harga Ethereum.
3. Mengevaluasi performa model menggunakan metrik Mean Squared Error (MSE) dan R² Score untuk menentukan model terbaik.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah data historis Ethereum yang mencakup periode harian dengan total **2160 baris** dan **10 kolom**. Data ini bersumber dari **Kaggle**. Berikut adalah uraian lengkap seluruh fitur pada dataset:

| No | Kolom     | Tipe Data | Deskripsi                                          |
| -- | --------- | --------- | -------------------------------------------------- |
| 1  | SNo       | int64     | Nomor urut data, sebagai indeks atau penanda baris |
| 2  | Name      | object    | Nama koin kripto, dalam hal ini "Ethereum"         |
| 3  | Symbol    | object    | Simbol dari koin, "ETH"                            |
| 4  | Date      | object    | Tanggal pencatatan data                            |
| 5  | High      | float64   | Harga tertinggi pada hari tersebut                 |
| 6  | Low       | float64   | Harga terendah pada hari tersebut                  |
| 7  | Open      | float64   | Harga pembukaan pada hari tersebut                 |
| 8  | Close     | float64   | Harga penutupan pada hari tersebut                 |
| 9  | Volume    | float64   | Volume transaksi pada hari tersebut                |
| 10 | Marketcap | float64   | Kapitalisasi pasar pada hari tersebut              |

### Kondisi Data

* **Missing Values:** Tidak ada nilai yang hilang di seluruh kolom dataset, sehingga data sudah lengkap untuk dianalisis.
* **Duplikasi Data:** Tidak ditemukan baris yang duplikat, menunjukkan bahwa setiap entri data bersifat unik.
* **Outlier:** Terdapat outlier pada kolom harga (Open, High, Low, Close) dan volume transaksi. Outlier ini kemudian ditangani pada tahap Data Preparation menggunakan metode Interquartile Range (IQR) untuk memastikan kualitas data yang lebih baik.

### Univariate Analysis EDA

* **Distribusi Harga Penutupan (Close):** Harga penutupan Ethereum menunjukkan variasi yang signifikan, yang mencerminkan sifat volatilitas tinggi pada pasar cryptocurrency. Distribusi ini tidak simetris dan memiliki beberapa puncak (multimodal).
* **Distribusi Volume:** Volume transaksi bervariasi secara signifikan, dengan beberapa periode menunjukkan lonjakan aktivitas perdagangan yang tinggi, kemungkinan terkait dengan peristiwa pasar tertentu.

### Multivariate Analysis EDA

* **Korelasi Antar Fitur:** Terdapat korelasi yang sangat kuat antara fitur harga (Open, High, Low, Close), dengan nilai korelasi mendekati 1. Hal ini menunjukkan bahwa fitur-fitur tersebut saling berkaitan erat dan bergerak bersama dalam pola yang serupa.
* **Hubungan dengan Volume:** Volume transaksi menunjukkan korelasi yang lebih lemah dengan fitur harga. Hal ini mengindikasikan bahwa volume mungkin tidak secara langsung memengaruhi pergerakan harga dalam jangka pendek, meskipun tetap relevan untuk analisis lebih lanjut.

## Data Preparation

Berikut adalah tahapan lengkap pemrosesan data yang dilakukan:

### Pembersihan Data

* Menghapus Kolom Tidak Relevan: Kolom SNo, Name, Symbol, Volume, dan Marketcap dihapus karena tidak relevan dengan prediksi harga.
* Menghapus Nilai Kosong: Baris dengan nilai kosong dihapus untuk memastikan integritas data.
* Penanganan Outlier: Outlier dideteksi dan dihapus menggunakan metode Interquartile Range (IQR) untuk meningkatkan kualitas data.

### Konversi Tipe Data

* Kolom Date diubah dari tipe object menjadi datetime dan dijadikan indeks untuk mempermudah analisis berbasis waktu.

### Pembuatan Fitur

* **OHLC\_Average:** Fitur baru dibuat dengan menghitung rata-rata dari Open, High, Low, dan Close untuk menangkap tren harga harian.
* **Price\_After\_Month:** Target prediksi dibuat dengan menggeser kolom Close sebanyak 30 hari ke depan.

### Pemisahan Fitur dan Target

* Fitur (X) berisi semua kolom kecuali Price\_After\_Month, sedangkan target (y) adalah Price\_After\_Month.

### Normalisasi Data

* Fitur dinormalisasi menggunakan StandardScaler untuk memastikan skala yang seragam antar variabel.

### Pembagian Data

* Dataset dibagi menjadi 80% data latih dan 20% data uji menggunakan fungsi `train_test_split` dari sklearn dengan `random_state=42` untuk konsistensi.

## Modeling

Tiga model machine learning diterapkan untuk memprediksi harga Ethereum:

* **K-Nearest Neighbors (KNN):** Model berbasis jarak dengan parameter `n_neighbors=10`.
* **Random Forest:** Model ensemble dengan parameter `n_estimators=50`, `max_depth=16`, `random_state=42`.
* **AdaBoost:** Model boosting dengan parameter `n_estimators=50`, `learning_rate=0.05`, `random_state=42`.

Setiap model dilatih dengan data latih dan dievaluasi dengan data uji.

## Evaluation

### Metrik Evaluasi

* **Mean Squared Error (MSE):** Mengukur rata-rata kuadrat error antara prediksi dan nilai aktual.
* **R² Score:** Mengukur seberapa baik model menjelaskan variabilitas data.

### Hasil Evaluasi

* **KNN:** MSE = 6100.06, R² = 0.8259
* **Random Forest:** MSE = 7163.01, R² = 0.7955
* **AdaBoost:** MSE = 6203.98, R² = 0.8229

Model Random Forest menunjukkan performa terbaik dengan MSE terendah, meskipun KNN memiliki R² tertinggi.

### Hubungan dengan Business Understanding

* **Menjawab Problem Statement:** Model berhasil memprediksi harga penutupan Ethereum 30 hari ke depan, dengan Random Forest memberikan hasil paling akurat. Fitur harga (Open, High, Low, Close) terbukti paling berpengaruh.
* **Mencapai Goals:** Perbandingan performa menunjukkan Random Forest sebagai model terbaik. Fitur yang berpengaruh juga telah diidentifikasi melalui EDA dan proses pemodelan.
* **Dampak Solusi:** Prediksi ini memungkinkan investor untuk merencanakan strategi investasi secara proaktif, meskipun volatilitas pasar tetap menjadi faktor yang perlu diperhatikan.

## Visualisasi

Plot prediksi vs data aktual untuk 100 data pertama menunjukkan bahwa Random Forest mampu mengikuti tren harga dengan baik.

![image](https://github.com/user-attachments/assets/a5bbd903-a7e0-4902-9eac-982d1446ddbe)


Prediksi Harga Ethereum 30 Hari ke Depan.

![image](https://github.com/user-attachments/assets/7c1a2daa-3215-4431-8249-05d7d23ef28f)



## Kesimpulan

* **Random Forest** adalah model terbaik untuk memprediksi harga Ethereum 30 hari ke depan berdasarkan data historis.
* Fitur harga (Open, High, Low, Close) memiliki pengaruh signifikan dalam prediksi.
* Prediksi ini dapat membantu investor merencanakan strategi, dengan catatan bahwa pasar cryptocurrency sangat fluktuatif.

## Referensi

* Dicoding. Diakses pada 20 Oktober 2023 dari [https://www.dicoding.com/academies/319-machine-learning-terapan](https://www.dicoding.com/academies/319-machine-learning-terapan)
* Kaggle. Dataset Ethereum: [[https://www.kaggle.com/datasets/akbarwidianto/ethereum-historical-data]([https://www.kaggle.com/datasets/akbarw/ethereum](https://www.kaggle.com/datasets/akbarw/ethereum))]([https://www.kaggle.com/datasets/akbarwidianto/ethereum-historical-data](https://www.kaggle.com/datasets/akbarw/ethereum))


