import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pickle
from PIL import Image
import plotly.express as px
import pandas as pd

# Konfigurasi halaman dengan tata letak lebar dan tema kustom
st.set_page_config(page_title="Klasifikasi Jenis Kendaraan", layout="wide", initial_sidebar_state="expanded")

# CSS kustom untuk tampilan modern
st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .stButton>button { background-color: #4f46e5; color: white; border-radius: 8px; }
    .stFileUploader { border: 2px dashed #e2e8f0; padding: 20px; border-radius: 10px; }
    .prediction-box { background-color: #e0f2fe; padding: 20px; border-radius: 10px; }
    .sidebar .sidebar-content { background-color: #ffffff; border-right: 1px solid #e2e8f0; }
    h1 { color: #1e3a8a; }
    h3 { color: #3b82f6; }
    </style>
""", unsafe_allow_html=True)

# Memuat model dan indeks kelas
@st.cache_resource
def load_resources():
    model = load_model('best_model.keras')
    with open('class_indices.pkl', 'rb') as f:
        class_indices = pickle.load(f)
    return model, class_indices

try:
    model, class_indices = load_resources()
    class_labels = {v: k for k, v in class_indices.items()}
except Exception as e:
    st.error(f"Error saat memuat model atau indeks kelas: {str(e)}")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("📤 Unggah Gambar Kendaraan")
    uploaded_file = st.file_uploader("Pilih gambar (JPG, JPEG, PNG)", type=['jpg', 'jpeg', 'png'], help="Unggah gambar kendaraan yang jelas untuk klasifikasi.")
    
    st.markdown("---")
    st.subheader("ℹ️ Tentang Aplikasi")
    st.info("""
    Aplikasi ini mengklasifikasikan jenis kendaraan berdasarkan gambar yang diunggah. Jenis kendaraan yang didukung:
    - Bus
    - SUV
    - Family Sedan
    - Fire Engine
    - Heavy Truck
    - Jeep
    - Truck
    - Taxi
    - Racing Car
    - Minibus
    """)

# Konten utama
st.title("🚗 Klasifikasi Jenis Kendaraan Menggunakan MobileNet V2")
st.markdown("Unggah gambar kendaraan untuk mengklasifikasikan jenisnya dengan model AI kami. Pastikan gambar jelas untuk prediksi yang akurat.")

if uploaded_file is not None:
    try:
        # Menampilkan gambar yang diunggah
        col1, col2 = st.columns([2, 1])
        with col1:
            img = Image.open(uploaded_file).convert('RGB')
            st.image(img, caption="Gambar yang Diunggah", use_container_width=True)
        
        # Proses dengan bilah kemajuan
        with col2:
            st.subheader("Hasil Prediksi")
            with st.container():
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Pra-pemrosesan
                status_text.text("Mengubah ukuran gambar...")
                progress_bar.progress(25)
                img_resized = img.resize((224, 224))
                
                status_text.text("Mengonversi ke array...")
                progress_bar.progress(50)
                img_array = image.img_to_array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Prediksi
                status_text.text("Menjalankan prediksi model...")
                progress_bar.progress(75)
                prediction = model.predict(img_array)
                
                progress_bar.progress(100)
                status_text.text("Prediksi selesai!")
                
                # Mendapatkan 3 prediksi teratas
                top_indices = np.argsort(prediction[0])[-3:][::-1]
                top_labels = [class_labels[idx] for idx in top_indices]
                top_confidences = [prediction[0][idx] * 100 for idx in top_indices]
                
                # Menampilkan hasil
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.write(f"**Jenis Kendaraan:** {top_labels[0]}")
                st.write(f"**Tingkat Kepercayaan:** {top_confidences[0]:.2f}%")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Grafik batang untuk tingkat kepercayaan
                df = pd.DataFrame({"Jenis Kendaraan": top_labels, "Tingkat Kepercayaan (%)": top_confidences})
                fig = px.bar(df, x="Tingkat Kepercayaan (%)", y="Jenis Kendaraan", orientation='h', 
                            color="Tingkat Kepercayaan (%)", color_continuous_scale="Blues",
                            title="3 Prediksi Teratas")
                st.plotly_chart(fig, use_container_width=True)
                
    except Exception as e:
        st.error(f"Error saat memproses gambar: {str(e)}")
else:
    st.info("Silakan unggah gambar untuk memulai klasifikasi.")