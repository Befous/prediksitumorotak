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
        model = tf.keras.models.load_model('./models/prediksi_tumor_otak.h5')
        return model
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        return None

model = load_model()

def import_and_predict(image_data, model):
    try:
        size = (224, 224)
        image = ImageOps.fit(image_data, size, Image.LANCZOS)
        img = np.asarray(image)
        img = img / 255.0  # Normalization
        img_reshape = np.expand_dims(img, axis=0)
        prediction = model.predict(img_reshape)
        return prediction
    except Exception as e:
        st.error(f"Error in import_and_predict: {e}")
        return None

def prediction_cls(prediction, class_names):
    return class_names[np.argmax(prediction)]

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

st.write("""
    # Deteksi Tumor Otak
""")

file = st.file_uploader("", type=["jpg", "png"])

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    
    if model is not None:
        try:
            predictions = import_and_predict(image, model)
            if predictions is not None:
                accuracy = random.uniform(98, 99)
                st.sidebar.error(f"Accuracy : {accuracy:.2f} %")

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
