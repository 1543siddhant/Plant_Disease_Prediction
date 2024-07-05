import streamlit as st
import tensorflow as tf
import numpy as np

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"



#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

#Main Page
if(app_mode=="Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    # 🌿 Welcome to the Plant Disease Recognition System! 🔍

Identify plant diseases effortlessly with our advanced recognition system. Upload an image, and let our technology do the rest.

---

## 🚀 Get Started
1. **Upload Your Image**: Navigate to the **Disease Recognition** page.
2. **Instant Analysis**: Our system processes your image with state-of-the-art algorithms.
3. **Receive Results**: Get immediate feedback and recommendations.

---

## 🌟 Why Choose Our System?
- **High Accuracy**: Leveraging cutting-edge machine learning for precise disease identification.
- **User-Friendly Interface**: Designed for ease of use.
- **Rapid Results**: Quick analysis for timely decision-making.

---

## 📋 Steps to Use
1. **Visit the Disease Recognition Page**
2. **Upload an Image of Your Plant**
3. **Wait for the System to Analyze**
4. **View Results and Recommendations**

---

## 📖 About Us
Discover more about our mission, team, and the technology behind the system on the **About** page.

Join us in safeguarding crops and promoting healthier harvests with the Plant Disease Recognition System.
    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                # 🗂️ About the Dataset

Our dataset is generated through offline augmentation based on the original dataset, which is available on [GitHub](https://github.com). It comprises approximately 87,000 RGB images of both healthy and diseased crop leaves, categorized into 38 distinct classes.

The dataset is split into training and validation sets in an 80/20 ratio while maintaining the directory structure. Additionally, a separate directory with 33 test images has been created for prediction purposes.

---

## 📁 Dataset Structure

1. **Training Set**: 70,295 images
2. **Validation Set**: 17,572 images
3. **Test Set**: 33 images


                """)

#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))
