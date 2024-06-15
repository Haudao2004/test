from keras.models import load_model  
from PIL import Image, ImageOps  
import numpy as np
import streamlit as st 
from dotenv import load_dotenv 
import os
import openai
import json
# from config import OPENAI_API_KEY
# import openai
# from flask import Flask, render_template

# -*- coding: utf-8 -*-

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')


# def load_api_key(secrets_file="secrets.json"):
#     with open(secrets_file) as f:
#         secrets = json.load(f)
#     return secrets["OPENAI_API_KEY"]

# # Set secret API key
# # Typically, we'd use an environment variable (e.g., echo "export OPENAI_API_KEY='yourkey'" >> ~/.zshrc)
# # However, using "internalConsole" in launch.json requires setting it in the code for compatibility with Hebrew
# api_key = load_api_key()
# openai.api_key = api_key


def classify_waste(img):
   
    np.set_printoptions(suppress=True)

    model = load_model("keras_model.h5", compile=False)
    # model_name = "gpt2"  
    # tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    # model = GPT2LMHeadModel.from_pretrained(model_name)
    class_names = open("labels.txt", "r").readlines()
  
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        
    image = img.convert("RGB")

    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
   
    image_array = np.asarray(image)

    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
   
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score
def generate_carbon_footprint_info(label):
    label = label.split(' ')[1]
    cache_file = "carbon_footprint_cache.json"
    
    # Kiểm tra xem kết quả đã được lưu trong cache chưa
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cache = json.load(f)
        
        if label in cache:
            return cache[label]
    
    # Nếu kết quả không tồn tại trong cache, gửi yêu cầu API mới
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-instruct",
        prompt="Lượng phát thải cacbon hoặc lượng khí thải cacbon gần đúng được tạo ra từ "+label+
        "? Tôi chỉ cần một con số gần đúng để tạo ra nhận thức. Xây dựng trong 100 từ.\n",
        temperature=0.7,
        max_tokens=600,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    
    result = response['choices'][0]['text']
    
    # Lưu kết quả vào cache
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cache = json.load(f)
    else:
        cache = {}
    
    cache[label] = result
    
    with open(cache_file, "w") as f:
        json.dump(cache, f)
    
    return result


# def generate_carbon_footprint_info(label):
#     label = label.split(' ')[1]
#     print(label)
#     response = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo-instruct",
#     prompt="Lượng phát thải cacbon hoặc lượng khí thải cacbon gần đúng được tạo ra từ "+label+
#     "? Tôi chỉ cần một con số gần đúng để tạo ra nhận thức. Xây dựng trong 100 từ.\n",
#     temperature=0.7,
#     max_tokens=600,
#     top_p=1,
#     frequency_penalty=0,
#     presence_penalty=0
#     )
#     return response['choices'][0]['text']


st.set_page_config(layout='wide')

st.title("Ứng dụng bền vững phân loại rác thải")

input_img = st.file_uploader("Nhập hình ảnh của bạn", type=['jpg', 'png', 'jpeg'])

if input_img is not None:
    if st.button("Phân Loại"):
        
        col1, col2, col3 = st.columns([1,1,1])

        with col1:
            st.info("Tải lên ảnh của bạn")
            st.image(input_img, use_column_width=True)

        with col2:
            st.info("Kết quả của bạn")
            image_file = Image.open(input_img)
            label, confidence_score = classify_waste(image_file)
            col4, col5 = st.columns([1,1])
            if label == "0 cardboard\n":
                st.success("Hình ảnh được phân loại là CARDBOARD.")                
                with col4:
                    st.image("sdg goals/12.png", use_column_width=True)
                    st.image("sdg goals/13.png", use_column_width=True)
                with col5:
                    st.image("sdg goals/14.png", use_column_width=True)
                    st.image("sdg goals/15.png", use_column_width=True) 
            elif label == "1 plastic\n":
                st.success("Hình ảnh được phân loại là PLASTIC.")
                with col4:
                    st.image("sdg goals/6.jpg", use_column_width=True)
                    st.image("sdg goals/12.png", use_column_width=True)
                with col5:
                    st.image("sdg goals/14.png", use_column_width=True)
                    st.image("sdg goals/15.png", use_column_width=True) 
            elif label == "2 glass\n":
                st.success("Hình ảnh được phân loại là GLASS.")
                with col4:
                    st.image("sdg goals/12.png", use_column_width=True)
                with col5:
                    st.image("sdg goals/14.png", use_column_width=True)
            elif label == "3 metal\n":
                st.success("Hình ảnh được phân loại là METAL.")
                with col4:
                    st.image("sdg goals/3.png", use_column_width=True)
                    st.image("sdg goals/6.jpg", use_column_width=True)
                with col5:
                    st.image("sdg goals/12.png", use_column_width=True)
                    st.image("sdg goals/14.png", use_column_width=True) 
            else:
                st.error("Hình ảnh không được phân loại vào bất kỳ lớp liên quan nào.")

        with col3:
            result = generate_carbon_footprint_info(label)
            st.success(result)

