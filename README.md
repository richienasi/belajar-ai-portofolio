# belajar-ai-portofolio
# 👁️ Ocular Disease AI Classification Web App

Aplikasi web berbasis Artificial Intelligence (AI) dan Machine Learning untuk melakukan deteksi dini serta analisis kondisi/penyakit mata (seperti Katarak, Glaukoma, dll.) dari citra medis mata.

![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)

---

## 📌 Fitur Utama
- **Upload Citra Medis:** Pengguna dapat mengunggah foto/citra medis mata.
- **Prediksi Otomatis AI:** Menghasilkan kelas diagnosis (misal: Glaukoma, Katarak, Normal) beserta *Confidence Score*.
- **Rekomendasi Medis Otomatis:** Memberikan deskripsi indikasi awal dan saran penanganan klinis secara langsung.
- **Antarmuka Interaktif:** Dibuat menggunakan Streamlit agar ringkas dan mudah digunakan.

---

## 🛠️ Teknologi & Tools
- **Language:** Python
- **Framework Web:** Streamlit
- **Deep Learning Framework:** TensorFlow / Keras
- **Library Tambahan:** NumPy, Pandas, Pillow, OpenCV

---

## 📁 Struktur Repositori
```text
.
├── app.py                      # Berkas utama untuk menjalankan aplikasi Streamlit
├── model_penyakit_mata_final.h5 # Model Deep Learning (HDF5) yang telah dilatih
├── Untitled0.ipynb             # Notebook eksperimen, pemrosesan data, & pelatihan model
├── requirements.txt            # Daftar pustaka/dependensi Python
└── README.md                   # Dokumentasi proyek
⚠️ Disclaimer Medis
Aplikasi ini dibangun untuk tujuan edukasi, riset, dan eksperimen portofolio pengembangan model AI. Hasil analisis dari aplikasi ini bukanlah diagnosis medis resmi dan tidak menggantikan pemeriksaan profesional oleh dokter spesialis mata (oftalmologis).
