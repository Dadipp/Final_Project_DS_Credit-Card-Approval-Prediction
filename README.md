# 🏦 Credit Card Approval Prediction

Proyek ini bertujuan untuk memprediksi persetujuan aplikasi kartu kredit berdasarkan data historis, termasuk histori pembayaran dan data finansial pemohon. Model ini dibangun menggunakan **Python**, **Scikit-learn**, dan di-deploy menggunakan **Streamlit**.

## 🚀 Demo Aplikasi
Kunjungi aplikasi Streamlit:  
👉 [Credit Card Approval Prediction](https://finalprojectdscredit-card-approval-prediction-szkuw6h6i4wxupem.streamlit.app)

---

## 📌 Fitur
- **Prediksi Persetujuan**: Menggunakan model XGBoost untuk memprediksi apakah aplikasi kartu kredit akan disetujui berdasarkan input pengguna.
- **Visualisasi Data**: Menampilkan Exploratory Data Analysis (EDA) dengan histogram, bar chart, dan pie chart untuk memberikan insight.
- **Penjelasan Prediksi**: Menggunakan **SHAP values** untuk menginterpretasikan fitur yang paling memengaruhi hasil prediksi.
- **Pengunggahan Dataset**: Mendukung analisis dengan mengunggah dataset baru untuk evaluasi lebih lanjut.
- **Navigasi Interaktif**: Menu yang mencakup Project Background, Business Objective, Data Understanding, EDA, Feature Correlation, Preprocessing, Evaluation, Prediction, dan Conclusion.

---

## 🛠️ Teknologi yang Digunakan
- **Python**
- **Streamlit**
- **Pandas & NumPy**
- **Matplotlib & Seaborn**
- **Scikit-learn**
- **SHAP**
- **Pickle**

---

## 📋 Struktur Proyek
- **Project Background**: Penjelasan tujuan proyek untuk menganalisis kelayakan kredit.
- **Business Objective**: Tujuan utama dan spesifik, termasuk evaluasi model XGBoost.
- **Data Understanding**: Pratinjau dataset dan statistik dasar.
- **EDA**: Analisis eksploratif dengan visualisasi distribusi usia, tipe pendapatan, dan status persetujuan.
- **Feature Correlation**: Heatmap untuk mengidentifikasi korelasi antar fitur.
- **Preprocessing**: Proses konversi target, One-Hot Encoding, pembagian data, standarisasi, penghapusan fitur, dan SMOTE.
- **Evaluation**: Evaluasi model dengan metrik AUC, Precision, Recall, dan F1-Score, serta SHAP untuk feature importance.
- **Prediction**: Form input untuk memprediksi kelayakan kredit secara interaktif.
- **Conclusion**: Ringkasan hasil dan rekomendasi untuk implementasi lebih lanjut.
