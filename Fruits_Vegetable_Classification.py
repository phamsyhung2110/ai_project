import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import cv2
from keras.models import load_model
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from freshness import freshness_percentage_by_cv_image, price_by_freshness_percentage, freshness_label

model = load_model('FV.h5') #load model

labels = {0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot',
          7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger',
          14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce',
          19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple',
          26: 'pomegranate', 27: 'potato', 28: 'raddish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn',
          32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'}

fruits = ['Apple', 'Banana', 'Bello Pepper', 'Chilli Pepper', 'Grapes', 'Jalepeno', 'Kiwi', 'Lemon', 'Mango', 'Orange',
          'Paprika', 'Pear', 'Pineapple', 'Pomegranate', 'Watermelon']
vegetables = ['Beetroot', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Corn', 'Cucumber', 'Eggplant', 'Ginger',
              'Lettuce', 'Onion', 'Peas', 'Potato', 'Raddish', 'Soy Beans', 'Spinach', 'Sweetcorn', 'Sweetpotato',
              'Tomato', 'Turnip']


def get_info(prediction):
    try:
        url = 'https://www.google.com/search?&q=calories in ' + prediction
        req = requests.get(url).text
        scrap = BeautifulSoup(req, 'html.parser')
        calories = scrap.find("div", class_="BNeawe iBp4i AP7Wnd").text
        return calories
    except Exception as e:
        st.error("Can't able to fetch the Calories")
        print(e)


def processing_img(img_path):
    img = load_img(img_path, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, [0])
    answer = model.predict(img)

    plt.bar(labels.values(), answer[0])
    plt.xticks(rotation=90)
    plt.show()

    y_class = answer.argmax(axis=-1)
    print(y_class)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = labels[y]
    print(res)
    return res.capitalize()


def run():
    st.title("Fruits&Vegetable Identification")
    img_file = st.file_uploader("Choose an Image", type=["jpg", "png"])
    if img_file is not None:
        img = Image.open(img_file).resize((250, 250))
        st.image(img, use_column_width=False)
        save_image_path = './upload_images/' + img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())
        if st.button("Clear Image"):
            os.remove(save_image_path)
            st.success("Image cleared!")
        if st.button("Predict"):
        # if img_file is not None:
            result = processing_img(save_image_path)
            print(result)
            if result in vegetables:
                st.info('**Category : Vegetables**')
            else:
                st.info('**Category : Fruit**')
            st.success("**Predicted : " + result + '**')
            cal = get_info(result)
            
            if cal:
                st.warning('**' + cal + '(100 grams)**')

            # fruit_counts = defaultdict(int)
            # fruit_counts[result] += 1
            # for i in range(10):
            #     result = processing_img(save_image_path)
            #     fruit_counts[result] += 1

            # total_fruits = sum(fruit_counts.values())
            # st.write('**Percentage of fruits detected in the image:**')
            # for fruit, count in fruit_counts.items():
            #     percentage = count / total_fruits * 100
            #     st.write(f'- {fruit}: {percentage:.2f}%')

            freshness_percentage = freshness_percentage_by_cv_image(cv2.imread(save_image_path))
            freshness = freshness_label(freshness_percentage)
            print(freshness)
            if freshness_percentage > 90:
                st.warning('**Freshness percentage: ' + str(freshness) + '**')
                st.warning('**Freshness level: Very Fresh')
            elif freshness_percentage > 65:
                st.warning('**Freshness percentage: ' + str(freshness) + '**')
                st.warning('**Freshness level: Normal')
            elif freshness_percentage > 50:
                st.warning('**Freshness percentage: ' + str(freshness) + '**')
                st.warning('**Freshness level: Not fresh')
            elif freshness_percentage > 0:
                st.warning('**Freshness percentage: ' + str(freshness) + '**')
                st.warning('**Freshness level: Spoiled')
            else:
                st.warning('**Freshness percentage: ' + str(freshness) + '**')
                # st.warning('**Freshness level: Fresh')
            

        


run()
