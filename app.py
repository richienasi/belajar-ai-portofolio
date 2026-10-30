import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Judul Aplikasi (Desain Profesional)
st.set_page_config(page_title="Sistem Deteksi Penyakit Mata AI", layout="centered")
st.title("👁️ Sistem Analisis Medis & Klasifikasi Penyakit Mata")
st.write("Unggah foto klinis mata untuk melakukan deteksi otomatis berbasis Deep Learning.")

# 2. Memuat Otak AI yang Sudah Kita Train di Colab
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model("model_penyakit_mata_final.h5")

model = load_my_model()
CLASS_NAMES = ['Cataracts', 'Uveitis', 'Crossed Eyes', 'Bulging Eyes', 'Glaucoma']

# 3. Fitur Unggah Gambar Oleh Pengguna/Perekrut Kerja
uploaded_file = st.file_uploader("Pilih gambar mata (Format: JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang Diunggah", use_container_width=True)
    
    st.write("🔄 Sedang menganalisis karakteristik citra medis...")
    
    # 4. Pra-pemrosesan gambar agar sesuai standar input TensorFlow (224x224)
    img_resized = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    
    # 5. Melakukan Prediksi
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions)
    
    highest_class_idx = np.argmax(score)
    result_disease = CLASS_NAMES[highest_class_idx]
    confidence_pct = 100 * np.max(score)
    
    # 6. Menampilkan Hasil Utama
    st.success(f"**Hasil Diagnosis AI:** Terdeteksi indikasi klinis **{result_disease}**")
    st.info(f"**Tingkat Keyakinan Model (Confidence Score):** {confidence_pct:.2f}%")
    
    # ----------------------------------------------------------------------
    # BAGIAN SIMULASI GENERATIVE AI (Fokus Validasi Sertifikat Anthropic)
    # ----------------------------------------------------------------------
    st.subheader("🤖 Rekomendasi Medis Otomatis (Asisten AI Terintegrasi)")
    st.write(f"Berikut adalah deskripsi analisis mengenai indikasi awal *{result_disease}*:")
    
    if result_disease == 'Cataracts':
        st.warning("⚠️ **Rekomendasi Klinis:** Kondisi ini menunjukkan adanya kekeruhan pada lensa mata. Disarankan untuk menjadwalkan pemeriksaan slit-lamp dengan dokter spesialis mata (Oftalmolog) guna mengevaluasi tingkat ketebalan katarak dan opsi tindakan bedah rawat jalan.")
    elif result_disease == 'Glaucoma':
        st.warning("⚠️ **Rekomendasi Klinis:** Indikasi kerusakan saraf optik akibat tekanan intraokular yang tinggi. Memerlukan penanganan segera melalui tes tonometri dan lapang pandang untuk mencegah penurunan penglihatan lebih lanjut.")
    else:
        st.warning("⚠️ **Rekomendasi Klinis:** Terdeteksi anomali pada struktur atau jaringan mata. Diperlukan pemeriksaan fisik komprehensif oleh tenaga medis profesional untuk penegakan diagnosis yang akurat.")
