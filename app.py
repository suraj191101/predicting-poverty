from flask import Flask, render_template, jsonify, request, Markup
from jinja2 import Environment, FileSystemLoader
import numpy as np
import os, re, glob, sys
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import time
import cv2

app = Flask(__name__)

model = load_model('Models/model.h5')
#model.make_prediction()
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
        preds="rich"
    elif preds==1:
        preds="middle."
    elif preds==2:
        preds="poor"

    else:
        preds="invalid"
        
    # elif preds==3:
    #     preds="Ashwagandha Ashwagandha contains chemicals that might help calm the brain, reduce swelling, lower blood pressure, and alter the immune system."
    # elif preds==4:
    #     preds="Bael bael : -used to cure constipation, diarrhea, diabetes -Controls cholesterol -Treatment for cholera -It is useful in curing scurvy."
    # elif preds == 5:
    #     preds = "Cinnamon  -  used to cure stomach aches, nausea, and diarrhea -Cure the Risk of Heart Disease."


    # elif preds == 6:
    #     preds = "Henna  -antiseptic for fungal or bacterial skin infections -Relief from headaches -Treats a variety of skin conditions -Promotes healthy hair"
    # elif preds == 7:
    #     preds = "Lavender LAVENDER-Aromatherapists use lavender in inhalation therapy to treat headaches, nervous disorders, and exhaustion. Herbalists treat skin ailments, such as fungal infections (like candidiasis), wounds, eczema, and acne, with lavender oil. It is also used in a healing bath for joint and muscle pain."
        
    # elif preds == 8:
    #     preds = "Marigold MARIGOLD-The main medicinal applications of marigold are skin conditions of all kinds, including contusions, bruises and varicose veins. Minor skin injuries and inflammation can also be successfully treated. Marigold ointment promotes wound healing for eczema and sunburns."
        
    # elif preds == 9:
    #     preds = "Neem Neem leaf is used for leprosy, eye disorders, bloody nose, intestinal worms, stomach upset, loss of appetite, skin ulcers, diseases of the heart and blood vessels (cardiovascular disease), fever, diabetes, gum disease (gingivitis), and liver problems. The leaf is also used for birth control and to cause abortions."
    # elif preds == 10:
    #     preds = "Peppermint Peppermint-Common cold, cough, inflammation of the mouth and throat, sinus infections and other respiratory infections "
    # elif preds == 11:
    #     preds = "Tulsi Tulsi-Cure fewer, skin problems like acne, blackheads, premature ageing, insect bites, heart disease, respiratory problems,kidney stones, sore throat, common cold,fewer,headaches,asthma"
        
    # elif preds == 12:
    #     preds = "Turmeric Turmeric-Colds,jaundice,intestinal worms"

        
    # else:
    #     preds="Invaild"
        
    
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
    app.run(host="127.0.0.1", port=5000, threaded=True)
