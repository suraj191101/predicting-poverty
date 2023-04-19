from flask import Flask, render_template, jsonify, request, Markup
import numpy as np
import os, re, glob, sys
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import time
import cv2
start = time.time()
app = Flask(__name__)
start = time.time()
model = load_model('Models/model.h5')
print(f'load model took {time.time()-start}')

def model_predict(img_path, model):
    start = time.time()
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))
    print(f'load_image took {time.time()-start}')
    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
    print(f'x took {time.time()-start}')
   

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
   # x = preprocess_input(x)

    preds = model.predict(x)
    #model.make_predict_function
    print(f'model predict took {time.time()-start}')
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="Aloe_Vera"
    elif preds==1:
        preds="Amla"
    elif preds==2:
        preds="Ashoka"
    elif preds==3:
        preds="Ashwagandha"
    elif preds==4:
        preds="Bael"
    elif preds == 5:
        preds = "Cinnamon"


    elif preds == 6:
        preds = "Henna"
    elif preds == 7:
        preds = "Lavender"
    elif preds == 8:
        preds = "Marigold"
    elif preds == 9:
        preds = "Neem"
    elif preds == 10:
        preds = "Peppermint"
    elif preds == 11:
        preds = "Tulsi"
    elif preds == 12:
        preds = "Turmeric"

        
    else:
        preds="Invaild"
        
    
    #time.sleep(1)
    end = time.time()
    print(f"Runtime model of the program is {end - start}")
    return preds


@app.route('/', methods=['GET'])
def home():
    start = time.time()
    time.sleep(1)
    end = time.time()
    print(f"Runtime home of the program is {end - start}")
    return render_template('index.html')




@app.route('/predict', methods=['GET', 'POST'])
def predict():
        if request.method == 'POST':
        # Get the file from post request
            f = request.files['file']

            # Save the file to ./uploads
            start = time.time()
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(
                basepath, 'uploads', secure_filename(f.filename))
            f.save(file_path)
            print(f'saving file took {time.time()-start}')

            # Make prediction
            preds = model_predict(file_path, model)

            
            result=preds
            #time.sleep(1)
            end = time.time()
            print(f"Runtime predict of the program is {end - start}")
            return render_template('display.html',result= result)
    


if __name__ == "__main__":
    #app.run(debug=True)
    model_predict('2.jpg',model)
    print(f'server started in {time.time()-start} sec')
    app.run(host="127.0.0.1", port=5000, threaded=True)
