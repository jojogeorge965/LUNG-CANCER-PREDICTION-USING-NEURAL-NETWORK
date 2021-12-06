import streamlit as st 
from PIL import Image
#from classify import predict
import glob



    
    
    

    
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.mobilenet import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image

import cv2
from PIL import Image, ImageOps

import numpy as np

import tensorflow as tf



class_names=['Benign cases', 'Malignant cases', 'Normal cases']

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('mobilenet_97_model.hdf5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

def import_and_predict(image1): 
    
    
    

    # convert the image pixels to a numpy array
    image = img_to_array(image1)
    
    figure_size = 9



    new_image_gauss = cv2.GaussianBlur(image, (figure_size, figure_size),0)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    

    
    
    

    # prepare the image for the VGG model
    image = preprocess_input(image)
    # predict the probability across all output classes

    # convert the probabilities to class labels
    
    prediction = model.predict(image)


    return prediction 


#st.title("LUNG CANCER PREDICTION")


image = Image.open('LOGO.jpg')
st.image(image, use_column_width=True)




uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    image=image.resize((224,224),Image.ANTIALIAS)


    
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    if st.button("Predict"):
        predictions = import_and_predict(image)
        score = tf.nn.softmax(predictions[0])
        pr=class_names[np.argmax(score)]
        st.write("Patient is having:",pr)
    if st.button("About"):
        st.text("Built by JG7 with lots of ❤️ in God's Own Country, Kerala")
        
    
    
    

  #  st.write(score)
 #   print(
#    "This image most likely belongs to {} with a {:.2f} percent confidence."
 #   .format(class_names[np.argmax(score)], 100 * np.max(score))
#)
    
    

