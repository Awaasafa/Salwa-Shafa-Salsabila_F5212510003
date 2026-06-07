# 🌿 Analisis dan Prediksi Indeks Kualitas Lingkungan Hidup (IKLH) Kota Palu Berbasis AI

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

---

## 📋 Deskripsi Proyek

Proyek ini bertujuan untuk menganalisis **parameter operasional kebersihan lapangan serta dampaknya terhadap kualitas lingkungan** di 8 kecamatan Kota Palu menggunakan metode pemodelan Sains Data. Tahapan ini berfokus pada pra-pemrosesan data dan pemisahan data (*Data Splitting*) murni guna mempersiapkan pengembangan model prediksi **Regresi Linear Berganda** dan **Support Vector Classification (SVC)** yang objektif.

---

## 📂 Struktur Proyek

```
📁 Salwa-Shafa-Salsabila_F5212510003/
├── README.md
├── 📄 PROJECT_SALWA_SHAFA.py
└── 📊 Data Set IKLH.csv
```

---

## 📊 Dataset

| Keterangan | Detail |
|---|---|
| Sumber | Dinas Lingkungan Hidup (DLH) Kota Palu |
| Periode | 2019 – 2024 |
| Cakupan | 8 Kecamatan di Kota Palu (Mantikulore, Palu Barat, Palu Selatan, Palu Timur, Palu Utara, Tatanga, Tawaeli, Ulujadi) |
| Total Data | 48 baris |

### Variabel yang Digunakan

| Variabel | Nama | Tipe | Keterangan |
|---|---|---|---|
| **X1** | TPS | Numerik | Jumlah Tempat Penampungan Sementara sampah |
| **X2** | Armada PickUp | Numerik | Jumlah unit kendaraan pickup pengangkut sampah |
| **X3** | Izin LH | Numerik | Jumlah dokumen izin lingkungan hidup yang terbit |
| **X4** | Pengaduan | Numerik | Jumlah laporan kasus pencemaran dari warga |
| **X5** | Bank Sampah | Numerik | Jumlah lokasi bank sampah aktif kelolaan warga |
| **X6** | Truk Besar | Numerik | Jumlah unit armada truk sampah berkapasitas besar |
| **X7** | IKA Kota | Numerik | Indeks Kualitas Air |
| **X8** | IKU Kota | Numerik | Indeks Kualitas Udara |
| **Y** | Target Label SVC | Kategori | Status kualitas lingkungan hidup (`RENDAH`, `SEDANG`, `TINGGI`) |

---

## ⚙️ Alur Pemrosesan Data

```
  Dataset CSV
       │
       ▼
 Load Dataset
 └── 48 data × 11 kolom
       │
       ▼
 Definisi Fitur & Target
 ├── X = [X1 sampai X8]
 └── y = Target Y (Label SVC)
       │
       ▼
 Split Data (Random Split)
 └── 70% Training | 15% Validasi | 15% Testing
       │
       ▼
 Cetak Shape
 └── Menampilkan ukuran dimensi data ke terminal
```

---

## ✂️ Split Data

Dataset dibagi secara proporsional menjadi 3 subset menggunakan metode **random split** dengan rasio **70 / 15 / 15** untuk menghindari bias (*data leakage*).

| Subset | Jumlah Data | Proporsi | Kegunaan |
|---|---|---|---|
| Training | 33 baris | 68.75% | Melatih algoritma model Sains Data |
| Validasi | 7 baris | 14.58% | Orientasi uji coba & tuning parameter model |
| Testing | 8 baris | 16.67% | Uji performa dan keandalan akhir model |

```python
# Langkah 1: Pisahkan 70% Training dan 30% Sementara (Temp)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# Langkah 2: Bagi sisa 30% Sementara menjadi Validasi dan Testing rata dua (50:50)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)
```

---

## 🛠️ Cara Menjalankan

### 1. Clone Repository

```bash
git clone https://github.com/Awaasafa/Salwa-Shafa-Salsabila_F5212510003
cd Salwa-Shafa-Salsabila_F5212510003
```

### 2. Install Dependencies

```bash
pip install pandas scikit-learn
```

### 3. Jalankan Program

```bash
python PROJECT_SALWA_SHAFA.py
```

### Output Terminal

```
=== MENAMPILKAN DATASET MURNI KOTA PALU ===
Total Baris Data Asli Berhasil Dimuat: 48 baris

=== PROSES PEMBAGIAN DATASET (SPLIT DATA) ===
--------------------------------------------------
X_train shape :  (33, 8)
X_val shape   :  (7, 8)
X_test shape  :  (8, 8)
--------------------------------------------------
y_train shape :  (33,)
y_val shape   :  (7,)
y_test shape  :  (8,)
--------------------------------------------------
[SUKSES] Pembagian data otomatis rasio 70:15:15 berhasil!
```

---

## 📦 Requirements

```
pandas>=1.5.0
scikit-learn>=1.1.0
```

---

## 👤 Author

| Keterangan | Detail |
|---|---|
| **Nama** | Salwa Shafa Salsabila |
| **NIM** | F5212510003 |
| **Universitas** | Universitas Tadulako |
| **Mata Kuliah** | Statistik dan Probabilitas |
| **Tahun** | 2026 |

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan akademik. Bebas digunakan dengan mencantumkan sumber.