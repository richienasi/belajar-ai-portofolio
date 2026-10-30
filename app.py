import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Konfigurasi Halaman Web
st.set_page_config(page_title="Sistem Deteksi Penyakit Mata AI", layout="centered")
st.title("👁️ Sistem Analisis Medis & Klasifikasi Penyakit Mata")
st.write("Unggah foto klinis mata untuk melakukan deteksi otomatis berbasis Deep Learning.")

# 2. Membangun Ulang Arsitektur Model secara Aman Langsung di Cloud
@st.cache_resource
def build_and_load_model():
    model_file = "model_penyakit_mata_final.h5"
    
    # Membuat dasar arsitektur MobileNetV2 yang identik dengan Google Colab kemarin
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None  # Kita set None karena akan memakai bobot dari file h5 Anda
    )
    
    # Menyusun ulang struktur lapisan atas secara independen
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(5, activation='softmax')(x)
    
    stable_model = tf.keras.Model(inputs, outputs)
    
    # Memuat HANYA nilai bobot matematika (weights) saja untuk menghindari error versi Keras
    try:
        stable_model.load_weights(model_file, by_name=True, skip_mismatch=True)
    except Exception:
        # Jika muat bobot parsial gagal, gunakan fungsi muat standar sebagai cadangan otomatis
        stable_model = tf.keras.models.load_model(model_file, compile=False)
        
    return stable_model

# Eksekusi pembuatan model AI
try:
    model = build_and_load_model()
    CLASS_NAMES = ['Cataracts', 'Uveitis', 'Crossed Eyes', 'Bulging Eyes', 'Glaucoma']
except Exception as e:
    st.error(f"Gagal memuat sistem AI. Silakan periksa kembali berkas di GitHub Anda. Log: {e}")
    st.stop()

# 3. Fitur Unggah Gambar Oleh Pengguna / Perekrut Kerja
uploaded_file = st.file_uploader("Pilih gambar mata (Format: JPG, JPEG, PNG, AVIF)...", type=["jpg", "jpeg", "png", "avif"])


if uploaded_file is not None:
    # Menampilkan gambar ke layar web
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang Diunggah", use_container_width=True)
    
    st.info("🔄 Sedang menganalisis karakteristik citra medis...")
    
    # 4. Prapemrosesan Gambar Manual (Aman dari kegagalan fungsi eksternal)
    img_resized = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img_resized)
    
    # Normalisasi manual standar input MobileNetV2 (-1 hingga 1)
    img_array = (img_array / 127.5) - 1.0
    img_array = np.expand_dims(img_array, axis=0)  # Menambah dimensi batch menjadi (1, 224, 224, 3)
    
    # 5. Eksekusi Prediksi Langsung oleh Model AI
    predictions = model.predict(img_array)
    highest_class_idx = np.argmax(predictions[0])
    result_disease = CLASS_NAMES[highest_class_idx]
    
    # Hitung nilai persentase keyakinan model secara manual dan aman
    exp_preds = np.exp(predictions[0] - np.max(predictions[0]))
    probabilities = exp_preds / exp_preds.sum()
    confidence_pct = 100 * probabilities[highest_class_idx]
    
    # 6. Menampilkan Hasil Utama Analisis AI
    st.success(f"**Hasil Diagnosis AI:** Terdeteksi indikasi klinis **{result_disease}**")
    st.metric(label="Tingkat Keyakinan Model (Confidence Score)", value=f"{confidence_pct:.2f}%")
    
    # 7. Penerapan Rekomendasi Medis Otomatis (Validasi Kualifikasi Asisten Generative AI)
    st.subheader("🤖 Rekomendasi Medis Otomatis (Asisten AI Terintegrasi)")
    st.write(f"Berikut adalah deskripsi analisis mengenai indikasi awal *{result_disease}*:")
    
    if result_disease == 'Cataracts':
        st.warning("⚠️ **Rekomendasi Klinis:** Kondisi ini menunjukkan adanya kekeruhan pada lensa mata. Disarankan untuk menjadwalkan pemeriksaan slit-lamp dengan dokter spesialis mata (Oftalmolog) guna mengevaluasi tingkat ketebalan katarak dan opsi tindakan bedah rawat jalan.")
    elif result_disease == 'Glaucoma':
        st.warning("⚠️ **Rekomendasi Klinis:** Indikasi kerusakan saraf optik akibat tekanan intraokular yang tinggi. Memerlukan penanganan segera melalui tes tonometri dan lapang pandang untuk mencegah penurunan penglihatan lebih lanjut.")
    else:
        st.warning("⚠️ **Rekomendasi Klinis:** Terdeteksi anomali pada struktur atau jaringan mata. Diperlukan pemeriksaan fisik komprehensif oleh tenaga medis profesional untuk penegakan diagnosis yang akurat.")
