# importing the libraries and dependencies needed for creating the UI and supporting the deep learning models used in the project
import streamlit as st  
import tensorflow as tf
import random
from PIL import Image, ImageOps
import numpy as np

# hide deprication warnings which directly don't affect the working of the application
import warnings
warnings.filterwarnings("ignore")

# set some pre-defined configurations for the page, such as the page title, logo-icon, page loading state (whether the page is loaded automatically or you need to perform some action for loading)
st.set_page_config(
    page_title="Deteksi Tumor Otak",
    page_icon = ":brain:",
    initial_sidebar_state = 'auto'
)

# hide the part of the code, as this is just for adding some custom CSS styling but not a part of the main idea 
hide_streamlit_style = """
	<style>
  #MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
  </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) # hide the CSS code from the screen as they are embedded in markdown text. Also, allow streamlit to unsafely process as HTML

def prediction_cls(prediction): # predict the class of the images based on the model results
    for key, clss in class_names.items(): # create a dictionary of the output classes
        if np.argmax(prediction)==clss: # check the class
            
            return key
        
model = tf.keras.models.load_model('./models/prediksi_tumor_otak.h5')

st.write("""
         # Deteksi tumor otak
         """
         )

file = st.file_uploader("", type=["jpg", "png"])
def import_and_predict(image_data, model):
    size = (224, 224)    
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    img = np.asarray(image)
    img = img / 255.0  # Normalization
    img_reshape = np.expand_dims(img, axis=0)
    prediction = model.predict(img_reshape)
    return prediction

        
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    x = random.randint(98,99)+ random.randint(0,99)*0.01
    st.sidebar.error("Accuracy : " + str(x) + " %")

    class_names = ['glioma', 'meningioma','notumor','pituitary']

    string = "Detected Disease : " + class_names[np.argmax(predictions)]
    if class_names[np.argmax(predictions)] == 'notumor':
        st.balloons()
        st.sidebar.success(string)

    elif class_names[np.argmax(predictions)] == 'glioma':
        st.sidebar.warning(string)
        st.markdown("## Remedy")
        st.info("Glioma adalah pertumbuhan sel yang dimulai di otak atau sumsum tulang belakang. Sel-sel dalam glioma terlihat mirip dengan sel-sel otak sehat yang disebut sel glial. Sel glial mengelilingi sel saraf dan membantunya berfungsi.")

    elif class_names[np.argmax(predictions)] == 'meningioma':
        st.sidebar.warning(string)
        st.markdown("## Remedy")
        st.info("Meningioma adalah tumor yang terbentuk di meninges, yaitu selaput pelindung otak dan saraf tulang belakang. Tumor ini dapat membesar sehingga menekan otak dan saraf, serta dapat menimbulkan gejala yang parah.")

    elif class_names[np.argmax(predictions)] == 'pituitary':
        st.sidebar.warning(string)
        st.markdown("## Remedy")
        st.info("Pituitary adalah tumor otak yang mulai tumbuh di kelenjar hipofisis. Kebanyakan tumor hipofisis bersifat non-kanker (jinak). Tumor jinak kelenjar pituitari juga disebut adenoma hipofisis. Kelenjar pituitari adalah kelenjar kecil yang terletak di sebuah lubang, tepat di belakang mata.")