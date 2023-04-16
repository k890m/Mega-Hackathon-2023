import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from flask import Flask, jsonify, request

model = load_model('FV.h5')

labels = {0: 'watermelon',1: 'lettuce',2: 'cabbage',3: 'mango',4: 'paprika',5: 'pineapple',6:
 'chilli pepper',7: 'peas',8: 'spinach',9: 'ginger',10: 'cauliflower',11: 'apple',12: 'orange',13:
 'banana',14: 'corn',15: 'tomato',16: 'bell pepper',17: 'sweetpotato',18: 'sweetcorn',19:
 'cucumber',20: 'beetroot',21: 'soy beans',22: 'onion',23: 'pear',24: 'turnip',25: 'lemon',
 26: 'carrot',27: 'garlic',28: 'pomegranate',29: 'kiwi',30: 'grapes',31: 'eggplant',32: 'potato',33:
 'capsicum',34: 'jalepeno',35: 'raddish'
}


def prepare_image(img_path):
    img = load_img(img_path, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, [0])
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)
    print(y_class)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = labels[y]
    print(res)
    return res.capitalize()


app = Flask(__name__)


#@app.route('/predict', methods=['POST'])
def infer_image():
    if 'file' not in request.files:
        return jsonify(error="Please try again. The Image doesn't exist")

    file = request.files.get('file')
    img_bytes = file.read()
    img_path = "./upload_images/test.jpg"
    with open(img_path, "wb") as img:
        img.write(img_bytes)
    result = prepare_image(img_path)
    return jsonify(prediction=result)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
