import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Konfigurasi Halaman Web
st.set_page_config(page_title="Sistem Deteksi Penyakit Mata AI", layout="centered")
st.title("👁️ Sistem Analisis Medis & Klasifikasi Penyakit Mata")
st.write("Unggah foto klinis mata untuk melakukan deteksi otomatis berbasis Deep Learning.")

# 2. Memuat Model Asli Secara Sempurna dengan Custom Objects
@st.cache_resource
def load_final_model():
    model_file = "model_penyakit_mata_final.h5"
    
    # Daftarkan fungsi pemrosesan bawaan Google yang ikut tersimpan di dalam file .h5
    custom_objects = {
        'preprocess_input': tf.keras.applications.mobilenet_v2.preprocess_input
    }
    
    # Memuat model asli secara utuh tanpa membongkar arsitekturnya
    return tf.keras.models.load_model(model_file, custom_objects=custom_objects, compile=False)

try:
    model = load_final_model()
    CLASS_NAMES = ['Cataracts', 'Uveitis', 'Crossed Eyes', 'Bulging Eyes', 'Glaucoma']
except Exception as e:
    st.error(f"Gagal memuat sistem AI. Log Error: {e}")
    st.stop()

# 3. Fitur Unggah Gambar
uploaded_file = st.file_uploader("Pilih gambar mata (Format: JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png", "avif"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang Diunggah", use_container_width=True)
    
    st.info("🔄 Sedang menganalisis karakteristik citra medis...")
    
    # 4. Pra-pemrosesan Gambar Menyesuaikan Dimensi TensorFlow
    img_resized = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) # Hasil: (1, 224, 224, 3)
    
    # 5. Eksekusi Prediksi Asli dari Model
    predictions = model.predict(img_array)
    raw_scores = predictions.flatten()
    
    # Perhitungan Softmax untuk mengubah output menjadi persentase probabilitas (0% - 100%)
    exp_scores = np.exp(raw_scores - np.max(raw_scores))
    probabilities = exp_scores / np.sum(exp_scores)
    
    highest_class_idx = np.argmax(probabilities)
    result_disease = CLASS_NAMES[highest_class_idx]
    confidence_pct = 100 * probabilities[highest_class_idx]
    
    # 6. Menampilkan Hasil Utama Analisis AI
    st.success(f"**Hasil Diagnosis AI:** Terdeteksi indikasi klinis **{result_disease}**")
    st.metric(label="Tingkat Keyakinan Model (Confidence Score)", value=f"{confidence_pct:.2f}%")
    
    # 7. Rekomendasi Medis Otomatis
    st.subheader("🤖 Rekomendasi Medis Otomatis (Asisten AI Terintegrasi)")
    st.write(f"Berikut adalah deskripsi analisis mengenai indikasi awal *{result_disease}*:")
    
    if result_disease == 'Cataracts':
        st.warning("⚠️ **Rekomendasi Klinis:** Kondisi ini menunjukkan adanya kekeruhan pada lensa mata. Disarankan untuk menjadwalkan pemeriksaan slit-lamp dengan dokter spesialis mata (Oftalmolog) guna mengevaluasi tingkat ketebalan katarak dan opsi tindakan bedah rawat jalan.")
    elif result_disease == 'Glaucoma':
        st.warning("⚠️ **Rekomendasi Klinis:** Indikasi kerusakan saraf optik akibat tekanan intraokular yang tinggi. Memerlukan penanganan segera melalui tes tonometri dan lapang pandang untuk mencegah penurunan penglihatan lebih lanjut.")
    else:
        st.warning("⚠️ **Rekomendasi Klinis:** Terdeteksi anomali pada struktur atau jaringan mata. Diperlukan pemeriksaan fisik komprehensif oleh tenaga medis profesional untuk penegakan diagnosis yang akurat.")
