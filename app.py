import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Konfigurasi Halaman Web
st.set_page_config(page_title="Sistem Deteksi Penyakit Mata AI", layout="centered")
st.title("👁️ Sistem Analisis Medis & Klasifikasi Penyakit Mata")
st.write("Unggah foto klinis mata untuk melakukan deteksi otomatis berbasis Deep Learning.")

# 2. Fungsi Memuat Model yang Aman dari Error Versi Keras (Compile=False)
@st.cache_resource
def load_my_model():
    model_file = "model_penyakit_mata_final.h5"
    # compile=False menjamin model sukses dimuat meski versi TensorFlow di cloud berbeda dengan Colab
    model = tf.keras.models.load_model(model_file, compile=False)
    return model

# Panggil fungsi pemuatan model
try:
    model = load_my_model()
    # Urutan kelas kategori sesuai dengan dataset mata kita
    CLASS_NAMES = ['Cataracts', 'Uveitis', 'Crossed Eyes', 'Bulging Eyes', 'Glaucoma']
except Exception as e:
    st.error(f"Gagal memuat model AI. Pastikan file model_penyakit_mata_final.h5 sudah diunggah di GitHub. Error: {e}")
    st.stop()

# 3. Fitur Unggah Gambar Oleh Pengguna / Perekrut Kerja
uploaded_file = st.file_uploader("Pilih gambar mata (Format: JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Menampilkan gambar ke layar web
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang Diunggah", use_container_width=True)
    
    st.info("🔄 Sedang menganalisis karakteristik citra medis...")
    
    # 4. Pra-pemrosesan gambar agar pas dengan input MobileNetV2 (224x224)
    img_resized = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)  # Menambah dimensi batch menjadi (1, 224, 224, 3)
    
    # 5. Melakukan Eksekusi Prediksi Langsung
    predictions = model.predict(img_array)
    
    # Ambil indeks dengan probabilitas tertinggi
    highest_class_idx = np.argmax(predictions[0])
    result_disease = CLASS_NAMES[highest_class_idx]
    
    # Hitung estimasi skor keyakinan secara manual (Aman dari perbedaan fungsi Keras)
    exp_preds = np.exp(predictions[0] - np.max(predictions[0]))
    probabilities = exp_preds / exp_preds.sum()
    confidence_pct = 100 * probabilities[highest_class_idx]
    
    # 6. Menampilkan Hasil Utama Analisis AI
    st.success(f"**Hasil Diagnosis AI:** Terdeteksi indikasi klinis **{result_disease}**")
    st.metric(label="Tingkat Keyakinan Model (Confidence Score)", value=f"{confidence_pct:.2f}%")
    
    # 7. Penerapan Rekomendasi Medis Pintar (Simulasi Integrasi LLM / Anthropic)
    st.subheader("🤖 Rekomendasi Medis Otomatis (Asisten AI Terintegrasi)")
    st.write(f"Berikut adalah deskripsi analisis mengenai indikasi awal *{result_disease}*:")
    
    if result_disease == 'Cataracts':
        st.warning("⚠️ **Rekomendasi Klinis:** Kondisi ini menunjukkan adanya kekeruhan pada lensa mata. Disarankan untuk menjadwalkan pemeriksaan slit-lamp dengan dokter spesialis mata (Oftalmolog) guna mengevaluasi tingkat ketebalan katarak dan opsi tindakan bedah rawat jalan.")
    elif result_disease == 'Glaucoma':
        st.warning("⚠️ **Rekomendasi Klinis:** Indikasi kerusakan saraf optik akibat tekanan intraokular yang tinggi. Memerlukan penanganan segera melalui tes tonometri dan lapang pandang untuk mencegah penurunan penglihatan lebih lanjut.")
    else:
        st.warning("⚠️ **Rekomendasi Klinis:** Terdeteksi anomali pada struktur atau jaringan mata. Diperlukan pemeriksaan fisik komprehensif oleh tenaga medis profesional untuk penegakan diagnosis yang akurat.")
