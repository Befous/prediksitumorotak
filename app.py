import streamlit as st
import tensorflow as tf
import random
from PIL import Image, ImageOps
import numpy as np
import warnings

# Hide deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Page configuration
st.set_page_config(
    page_title="Deteksi Tumor Otak",
    page_icon=":brain:",
    initial_sidebar_state='auto'
)

# Hide Streamlit's default menu and footer
hide_streamlit_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('./prediksi_tumor_otak2.h5')
        return model
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        return None

model = load_model()

def import_and_predict(image_data, model):
    try:
        size = (512, 512)
        image = ImageOps.fit(image_data, size, Image.LANCZOS)
        img = np.asarray(image)
        img = img / 255.0  # Normalization
        img_reshape = np.expand_dims(img, axis=0)
        
        st.write("Image reshaped:", img_reshape.shape)
        
        prediction = model.predict(img_reshape)
        
        st.write("Prediction result:", prediction)
        
        return prediction
    except tf.errors.ResourceExhaustedError as e:
        st.error(f"Resource exhausted error: {e}")
    except tf.errors.InvalidArgumentError as e:
        st.error(f"Invalid argument error: {e}")
    except Exception as e:
        st.error(f"General error in import_and_predict: {e}")
    return None

def prediction_cls(prediction, class_names):
    return class_names[np.argmax(prediction)]

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Add navigation
st.sidebar.title("Navigasi")
option = st.sidebar.radio("Pilih halaman:", ["Beranda", "Prediksi", "About Us"])

if option == "Beranda":
    st.title("Selamat Datang di Aplikasi Deteksi Tumor Otak")
    st.write("""
        Aplikasi ini dirancang untuk membantu mendeteksi jenis tumor otak berdasarkan gambar MRI.
        Pilih halaman Prediksi untuk memulai diagnosis. Output dari predilsi ada 4: Glioma, Meningioma, Pituitary, dan Notumor
    """)
elif option == "Prediksi":
    st.title("Deteksi Tumor Otak")

    file = st.file_uploader("Unggah gambar MRI Anda", type=["jpg", "png"])

    if file is None:
        st.text("Silakan unggah file gambar")
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        
        if model is not None:
            try:
                predictions = import_and_predict(image, model)
                if predictions is not None:
                    accuracy = np.max(predictions)
                    accuracy_percent = accuracy * 100
                    st.sidebar.error(f"Accuracy : {accuracy_percent:.2f}%")

                    detected_class = prediction_cls(predictions, class_names)
                    result_string = f"Detected Disease : {detected_class}"
                    
                    if detected_class == 'notumor':
                        st.balloons()
                        st.sidebar.success(result_string)
                    else:
                        st.sidebar.warning(result_string)
                        st.markdown("## Remedy")
                        
                        if detected_class == 'glioma':
                            st.info("Glioma adalah pertumbuhan sel yang dimulai di otak atau sumsum tulang belakang. Sel-sel dalam glioma terlihat mirip dengan sel-sel otak sehat yang disebut sel glial. Sel glial mengelilingi sel saraf dan membantunya berfungsi.")
                        
                        elif detected_class == 'meningioma':
                            st.info("Meningioma adalah tumor yang terbentuk di meninges, yaitu selaput pelindung otak dan saraf tulang belakang. Tumor ini dapat membesar sehingga menekan otak dan saraf, serta dapat menimbulkan gejala yang parah.")
                        
                        elif detected_class == 'pituitary':
                            st.info("Pituitary adalah tumor otak yang mulai tumbuh di kelenjar hipofisis. Kebanyakan tumor hipofisis bersifat non-kanker (jinak). Tumor jinak kelenjar pituitari juga disebut adenoma hipofisis. Kelenjar pituitari adalah kelenjar kecil yang terletak di sebuah lubang, tepat di belakang mata.")
                else:
                    st.error("Prediction failed.")
            except Exception as e:
                st.error(f"Error during prediction: {e}")

elif option == "About Us":
    st.title("Tentang Kami")
    st.write("""
        Aplikasi prediksi tumor otak ini dibangun oleh 2 mahasiswa Universitas Logistik dan Bisnis Internasional, Ibrohim Mubarok dengan NPM 1214081 dan Aulia Maharani dengan NPM 1214079
    """)
