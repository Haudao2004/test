from keras.models import load_model  
from PIL import Image, ImageOps  
import numpy as np
import streamlit as st 
from dotenv import load_dotenv 
import os
import openai
import json
import openai

# Load environment variables


load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

def load_api_key(secrets_file="secrets.json"):
    with open(secrets_file) as f:
        secrets = json.load(f)
    return secrets["OPENAI_API_KEY"]

# Set secret API key
# Typically, we'd use an environment variable (e.g., echo "export OPENAI_API_KEY='yourkey'" >> ~/.zshrc)
# However, using "internalConsole" in launch.json requires setting it in the code for compatibility with Hebrew
api_key = load_api_key()
openai.api_key = api_key

def classify_waste(img):
    np.set_printoptions(suppress=True)

    # Load the Keras model
    model = load_model("keras_model.h5", compile=False)
    class_names = open("labels.txt", "r").readlines()

    # Prepare the image
    image = img.convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Predict the class of the image
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    return class_name, confidence_score

def generate_carbon_footprint_info(label):
    material = label.split(' ')[1]
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"Lượng phát thải cacbon hoặc lượng khí thải cacbon gần đúng được tạo ra từ {material}? Tôi chỉ cần một con số gần đúng để tạo ra nhận thức. Xây dựng trong 100 từ.\n",
        temperature=0.7,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response['choices'][0]['text']

st.set_page_config(layout='wide')
st.title("Ứng dụng bền vững phân loại chất thải")

input_img = st.file_uploader("Nhập hình ảnh của bạn", type=['jpg', 'png', 'jpeg'])

if input_img is not None:
    if st.button("Phân Loại"):
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.info("Tải lên ảnh của bạn")
            st.image(input_img, use_column_width=True)

        with col2:
            st.info("Kết quả của bạn")
            image_file = Image.open(input_img)
            label, confidence_score = classify_waste(image_file)
            col4, col5 = st.columns([1, 1])
            
            if label == "cardboard":
                st.success("Hình ảnh được phân loại là CARDBOARD.")                
                with col4:
                    st.image("sdg_goals/12.png", use_column_width=True)
                    st.image("sdg_goals/13.png", use_column_width=True)
                with col5:
                    st.image("sdg_goals/14.png", use_column_width=True)
                    st.image("sdg_goals/15.png", use_column_width=True) 
            elif label == "plastic":
                st.success("Hình ảnh được phân loại là PLASTIC.")
                with col4:
                    st.image("sdg_goals/6.jpg", use_column_width=True)
                    st.image("sdg_goals/12.png", use_column_width=True)
                with col5:
                    st.image("sdg_goals/14.png", use_column_width=True)
                    st.image("sdg_goals/15.png", use_column_width=True) 
            elif label == "glass":
                st.success("Hình ảnh được phân loại là GLASS.")
                with col4:
                    st.image("sdg_goals/12.png", use_column_width=True)
                with col5:
                    st.image("sdg_goals/14.png", use_column_width=True)
            elif label == "metal":
                st.success("Hình ảnh được phân loại là METAL.")
                with col4:
                    st.image("sdg_goals/3.png", use_column_width=True)
                    st.image("sdg_goals/6.jpg", use_column_width=True)
                with col5:
                    st.image("sdg_goals/12.png", use_column_width=True)
                    st.image("sdg_goals/14.png", use_column_width=True) 
            else:
                st.error("Hình ảnh không được phân loại vào bất kỳ lớp liên quan nào.")

        with col3:
            result = generate_carbon_footprint_info(label)
            st.success(result)
