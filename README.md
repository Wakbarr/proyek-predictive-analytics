# Laporan Proyek Machine Learning - Akbar Widianto

## Domain Proyek
Harga cryptocurrency, khususnya Ethereum, dikenal sangat fluktuatif dan dipengaruhi oleh berbagai faktor pasar yang kompleks, seperti sentimen pasar, volume transaksi, dan kebijakan ekonomi global. Investor dan trader sering kali kesulitan memprediksi pergerakan harga untuk membuat keputusan yang tepat. Analisis tradisional cenderung reaktif, di mana keputusan diambil setelah perubahan harga terjadi, sehingga kurang efektif dalam pasar yang dinamis. Oleh karena itu, pendekatan prediktif berbasis machine learning diperlukan untuk memberikan wawasan proaktif tentang potensi pergerakan harga di masa depan.

Proyek ini bertujuan membangun model machine learning untuk memprediksi harga penutupan Ethereum 30 hari ke depan berdasarkan data historis. Tiga model yang digunakan adalah K-Nearest Neighbors (KNN), Random Forest, dan AdaBoost, dengan perbandingan performa untuk menemukan algoritma terbaik. Dataset yang digunakan adalah data historis Ethereum dari Kaggle, tersedia di  
https://www.kaggle.com/datasets/akbarwidianto/ethereum-historical-data.

**Referensi:**  
- Dicoding: https://www.dicoding.com/academies/319-machine-learning-terapan  
- Kaggle Dataset: https://www.kaggle.com/datasets/akbarwidianto/ethereum-historical-data  

---

## Business Understanding

### Problem Statements
- Bagaimana cara memprediksi harga penutupan Ethereum 30 hari ke depan berdasarkan data historis?  
- Model machine learning mana yang paling akurat dalam memprediksi harga Ethereum?  
- Fitur apa saja yang paling berpengaruh dalam memprediksi harga Ethereum?  

### Goals
- Membangun model machine learning untuk memprediksi harga penutupan Ethereum 30 hari ke depan.  
- Membandingkan performa KNN, Random Forest, dan AdaBoost untuk menentukan model terbaik.  
- Mengidentifikasi fitur yang paling berpengaruh dalam prediksi harga Ethereum.  

### Solution Statement
- Melakukan Exploratory Data Analysis (EDA) untuk memahami karakteristik data dan hubungan antar fitur.  
- Menerapkan tiga algoritma (KNN, Random Forest, dan AdaBoost) untuk memprediksi harga Ethereum, kemudian memilih model terbaik berdasarkan metrik evaluasi.  
- Mengevaluasi performa model dengan metrik Mean Squared Error (MSE) dan R² Score untuk memastikan solusi yang terukur dan sesuai dengan tujuan.  

---

## Data Understanding
Dataset yang digunakan adalah data historis Ethereum dari Kaggle (https://www.kaggle.com/datasets/akbarwidianto/ethereum-historical-data), dengan **2160 baris** dan **9 kolom**. Berikut adalah deskripsi fitur:

| No | Kolom     | Tipe Data | Deskripsi                                                      |
|----|-----------|-----------|----------------------------------------------------------------|
| 1  | SNo       | int64     | Nomor urut data, sebagai penanda baris                         |
| 2  | Name      | object    | Nama koin kripto, dalam hal ini "Ethereum"                     |
| 3  | Symbol    | object    | Simbol koin, "ETH"                                             |
| 4  | High      | float64   | Harga tertinggi pada hari tersebut                             |
| 5  | Low       | float64   | Harga terendah pada hari tersebut                              |
| 6  | Open      | float64   | Harga pembukaan pada hari tersebut                             |
| 7  | Close     | float64   | Harga penutupan pada hari tersebut                             |
| 8  | Volume    | float64   | Volume transaksi pada hari tersebut                            |
| 9  | Marketcap | float64   | Kapitalisasi pasar pada hari tersebut                          |

**Catatan:** Kolom Date tidak ada dalam dataset asli tetapi ditambahkan sebagai indeks saat memuat data dengan `pd.read_csv()` dan `set_index('Date')`.

### Kondisi Data
- **Missing Values:** Tidak ada nilai kosong pada dataset awal.  
- **Duplikasi Data:** Tidak ada baris duplikat.  
- **Outlier:** Terdapat outlier pada kolom harga (Open, High, Low, Close), yang ditangani pada tahap Data Preparation.

---

## Univariate Analysis
- **Distribusi Harga Penutupan (Close):** Variasi harga tinggi, distribusi multimodal, mencerminkan volatilitas pasar cryptocurrency.  
- **Distribusi Volume:** Variasi signifikan, dengan lonjakan pada periode tertentu, menunjukkan aktivitas perdagangan yang tidak konsisten.

---

## Multivariate Analysis
- **Korelasi Antar Fitur:** Kolom harga (Open, High, Low, Close) memiliki korelasi mendekati 1, menunjukkan pergerakan seragam dalam satu hari.  
- **Hubungan dengan Volume:** Korelasi lemah antara Volume dan harga, menunjukkan pengaruh terbatas dalam jangka pendek.

---

## Data Preparation
Berikut adalah tahapan data preparation yang dilakukan secara runtut sesuai dengan notebook:

1. **Memuat dan Mengurutkan Data**  
   - Dataset dimuat dari file CSV dan diurutkan berdasarkan kolom Date menggunakan `sort_values(by='Date')`.  
   - Kolom Date dijadikan indeks dengan `set_index('Date')`.  
   **Alasan:** Memastikan data diurutkan kronologis untuk analisis deret waktu.

2. **Pembersihan Data**  
   - Menghapus Kolom Tidak Relevan: SNo, Name, Symbol, Volume, dan Marketcap dihapus karena tidak relevan untuk prediksi.  
   **Alasan:** Mengurangi noise dan fokus pada fitur yang berpengaruh terhadap prediksi.

3. **Pembuatan Fitur**  
   - `OHLC_Average`: Rata-rata dari kolom Open, High, Low, dan Close dibuat dengan `mean(axis=1)`.  
   - `Price_After_Month`: Target prediksi dibuat dengan menggeser Close 30 hari ke depan menggunakan `shift(-30)`.  
   **Alasan:** OHLC_Average meringkas informasi harga harian, sementara Price_After_Month menentukan target prediksi 30 hari ke depan.

4. **Penghapusan Nilai Kosong**  
   - Baris dengan nilai kosong (akibat penggeseran Close) dihapus dengan `dropna()`.  
   **Alasan:** Memastikan data lengkap untuk pelatihan model.

5. **Penanganan Outlier**  
   - Outlier pada kolom harga dan OHLC_Average dideteksi dan dihapus menggunakan metode Interquartile Range (IQR).  
   **Alasan:** Mencegah bias pada model akibat nilai ekstrem yang tidak representatif.

6. **Pemisahan Fitur dan Target**  
   - Fitur (X): Semua kolom kecuali Price_After_Month.  
   - Target (y): Kolom Price_After_Month.  
   **Alasan:** Memisahkan variabel independen dan dependen untuk pelatihan model.

7. **Normalisasi Data**  
   - Fitur dinormalisasi menggunakan StandardScaler.  
   **Alasan:** Penting untuk algoritma seperti KNN yang sensitif terhadap skala data.

8. **Pembagian Data**  
   - Data dibagi menjadi 80% data latih dan 20% data uji dengan `train_test_split(..., random_state=42)`.  
   **Alasan:** Memungkinkan evaluasi model yang objektif dan reproduktif.

---

## Modeling
Tiga model machine learning diterapkan untuk memprediksi harga Ethereum:

- **K-Nearest Neighbors (KNN)**  
  - Cara Kerja: Mencari 10 tetangga terdekat (Euclidean) dan mengambil rata-rata target.  
  - Parameter: `n_neighbors=10`.  
  - Kelebihan: Mudah diimplementasikan, interpretasi intuitif.  
  - Kekurangan: Sensitif terhadap skala dan outlier; lambat pada dataset besar.

- **Random Forest**  
  - Cara Kerja: Ensemble dari 50 decision trees; prediksi akhir diambil rata-rata.  
  - Parameter: `n_estimators=50, max_depth=16, random_state=42`.  
  - Kelebihan: Mengurangi overfitting, robust terhadap outlier.  
  - Kekurangan: Kompleks, membutuhkan sumber daya komputasi lebih.

- **AdaBoost**  
  - Cara Kerja: Menggabungkan weak learners secara berurutan dengan pembobotan.  
  - Parameter: `n_estimators=50, learning_rate=0.05, random_state=42`.  
  - Kelebihan: Meningkatkan akurasi model dasar.  
  - Kekurangan: Sensitif terhadap noise dan outlier.

Ketiga model dievaluasi, dan model terbaik dipilih berdasarkan metrik evaluasi.

---

## Evaluation

### Metrik Evaluasi
- **Mean Squared Error (MSE):**  
  \`\`\`  
  MSE = (1/n) * Σ(y_actual - y_predicted)²  
  \`\`\`  
  Semakin rendah MSE, semakin baik model.

- **R² Score:** Proporsi varians yang dijelaskan model (0–1, semakin dekat ke 1 semakin baik).

### Hasil Evaluasi
- **KNN:** MSE = 6100.06, R² = 0.8259  
- **Random Forest:** MSE = 7163.01, R² = 0.7955  
- **AdaBoost:** MSE = 6203.98, R² = 0.8229  

**Analisis:** KNN memiliki MSE terendah dan R² tertinggi, menunjukkan performa terbaik dalam memprediksi harga Ethereum.

---

## Hubungan dengan Business Understanding
- Prediksi harga 30 hari ke depan tercapai dengan **KNN** sebagai model terbaik.  
- Fitur harga (Open, High, Low, Close) yang diringkas dalam OHLC_Average paling berpengaruh.  
- Investor dapat merencanakan strategi proaktif berdasarkan prediksi ini, meski volatilitas pasar tetap menjadi tantangan.

---

## Visualisasi
- **Prediksi vs Aktual:** Plot 100 data pertama menunjukkan Random Forest mengikuti tren aktual dengan baik.

![image](https://github.com/user-attachments/assets/f94ef90d-b0f1-4c08-9802-ceb8e7f2ad18)

  
- **Prediksi 30 Hari:** Visualisasi prediksi harga dengan KNN menunjukkan proyeksi realistis.

![image](https://github.com/user-attachments/assets/3813cebd-aec0-475a-a9f6-745f800ab199)


---

## Kesimpulan
- **KNN** adalah model terbaik untuk memprediksi harga Ethereum 30 hari ke depan.  
- Fitur harga (Open, High, Low, Close) sangat berpengaruh dalam prediksi.  
- Prediksi ini dapat membantu investor membuat keputusan proaktif, meskipun pasar cryptocurrency sangat fluktuatif.

---

## Referensi
- Dicoding: https://www.dicoding.com/academies/319-machine-learning-terapan  
- Kaggle Dataset: https://www.kaggle.com/datasets/akbarwidianto/ethereum-historical-data  
