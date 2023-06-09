import streamlit as st
from PIL import Image
import requests

#from bs4 import BeautifulSoup

labels = {0: 'watermelon',1: 'lettuce',2: 'cabbage',3: 'mango',4: 'paprika',5: 'pineapple',6:
 'chilli pepper',7: 'peas',8: 'spinach',9: 'ginger',10: 'cauliflower',11: 'apple',12: 'orange',13:
 'banana',14: 'corn',15: 'tomato',16: 'bell pepper',17: 'sweetpotato',18: 'sweetcorn',19:
 'cucumber',20: 'beetroot',21: 'soy beans',22: 'onion',23: 'pear',24: 'turnip',25: 'lemon',
 26: 'carrot',27: 'garlic',28: 'pomegranate',29: 'kiwi',30: 'grapes',31: 'eggplant',32: 'potato',33:
 'capsicum',34: 'jalepeno',35: 'raddish'
}
    

def run():
    st.title("Food Identifier")
    img_file = st.file_uploader("Choose an Image", type=["jpg", "png"])
    if img_file is not None:
        img = Image.open(img_file).resize((250, 250))
        st.image(img, use_column_width=False)
        save_image_path = './upload_images/' + img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())
          
        if img_file is not None:
            url = 'http://192.168.1.5:5000/predict'
            form_data = {'file': open(save_image_path, 'rb')}
            resp = requests.post(url, files=form_data)
            resp_dict = resp.json()
            result = resp_dict['prediction']
            print(result)
            st.success("**Predicted : " + result + '**')

run()
