# Importing essential libraries and modules
from flask import Flask, render_template, request
from markupsafe import Markup
import numpy as np
import pandas as pd
import io
import os
import torch
from torchvision import transforms
from PIL import Image
import joblib
import requests

from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
from utils.model import ResNet9   # <-- MUST be your corrected model.py


# -----------------------------------------------------
# FLASK APP
# -----------------------------------------------------
app = Flask(__name__)


# -----------------------------------------------------
# LOAD DISEASE MODEL ONCE (correct architecture)
# -----------------------------------------------------
disease_classes = [
    'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
    'Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy','Grape___Black_rot',
    'Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy',
    'Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight',
    'Potato___Late_blight','Potato___healthy','Raspberry___healthy','Soybean___healthy',
    'Squash___Powdery_mildew','Strawberry___Leaf_scorch','Strawberry___healthy',
    'Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight',
    'Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

APP_ROOT = os.path.dirname(__file__)
MODEL_PATH = os.path.join(APP_ROOT, "models", "plant_disease_model.pth")

# Load disease model (architecture matches `utils/model.py`)
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
disease_model.eval()

# Training transform used in your notebook
disease_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()   # EXACT transform from training (resize for inference)
])


# -----------------------------------------------------
# LOAD CROP RECOMMENDATION MODEL
# -----------------------------------------------------
try:
    crop_recommendation_model = joblib.load(os.path.join(APP_ROOT, "models", "RandomForest.pkl"))
except Exception as e:
    print("Could not load crop recommendation model:", e)
    crop_recommendation_model = None


# -----------------------------------------------------
# WEATHER FETCH FUNCTION
# -----------------------------------------------------
def weather_fetch(city_name):
    try:
        api_key = "74d9ea3363ac4ac9fbe1578ae953035f"
        base_url = "http://api.openweathermap.org/data/2.5/weather?"
        complete_url = base_url + "appid=" + api_key + "&q=" + city_name

        response = requests.get(complete_url)
        x = response.json()

        if x.get("cod") != 200:
            print("City not found:", x.get("message"))
            return None

        temp = round(x["main"]["temp"] - 273.15, 2)
        humidity = x["main"]["humidity"]
        return temp, humidity

    except:
        return None


# -----------------------------------------------------
# HOME PAGE
# -----------------------------------------------------
@app.route('/')
def home():
    title = 'AgriSense AI - Home'
    return render_template('index.html', title=title)


# -----------------------------------------------------
# CROP RECOMMENDATION PAGE
# -----------------------------------------------------
@app.route('/crop-recommend')
def crop_recommend():
    title = 'AgriSense AI - Crop Recommendation'
    return render_template('crop.html', title=title)


@app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'AgriSense AI - Crop Recommendation'

    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])
    city = request.form.get("city")

    weather = weather_fetch(city)

    if weather is None:
        return render_template('try_again.html', title=title)

    temp, humidity = weather

    data = np.array([[N, P, K, temp, humidity, ph, rainfall]])

    if crop_recommendation_model is None:
        return render_template('try_again.html', title=title)

    prediction = crop_recommendation_model.predict(data)[0]

    return render_template('crop-result.html',
                           prediction=prediction,
                           title=title)


# -----------------------------------------------------
# FERTILIZER RECOMMENDATION PAGE
# -----------------------------------------------------
@app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'AgriSense AI - Fertilizer Suggestion'
    return render_template('fertilizer.html', title=title)


@app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'Harvestify - Fertilizer Suggestion'

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])

    try:
        fert_path = os.path.join(APP_ROOT, 'Data', 'fertilizer.csv')
        df = pd.read_csv(fert_path)
    except Exception as e:
        print('Could not load fertilizer data:', e)
        return render_template('try_again.html', title=title)

    try:
        nr = df[df['Crop'] == crop_name]['N'].iloc[0]
        pr = df[df['Crop'] == crop_name]['P'].iloc[0]
        kr = df[df['Crop'] == crop_name]['K'].iloc[0]
    except Exception as e:
        print('Fertilizer lookup error:', e)
        return render_template('try_again.html', title=title)

    diff = {
        abs(nr - N): "N",
        abs(pr - P): "P",
        abs(kr - K): "K"
    }

    max_key = diff[max(diff)]

    if max_key == "N":
        key = 'NHigh' if nr < N else "Nlow"
    elif max_key == "P":
        key = 'PHigh' if pr < P else "Plow"
    else:
        key = 'KHigh' if kr < K else "Klow"

    response = Markup(str(fertilizer_dic[key]))

    return render_template('fertilizer-result.html',
                           recommendation=response,
                           title=title)



# -----------------------------------------------------
# DISEASE PREDICTION PAGE
# -----------------------------------------------------
@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'AgriSense AI - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('disease.html', title=title)

        file = request.files['file']
        if file.filename == '':
            return render_template('disease.html', title=title)

        filepath = os.path.join(APP_ROOT, "static", "uploads", file.filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        file.save(filepath)

        try:
            image = Image.open(filepath).convert("RGB")

            # Inference transform (resize to training size then tensor)
            img_t = disease_transform(image)
            batch_t = torch.unsqueeze(img_t, 0)

            # Use the preloaded global model for inference
            with torch.no_grad():
                output = disease_model(batch_t)
                _, predicted = torch.max(output, 1)
                predicted_class = predicted.item()

            disease_name = disease_classes[predicted_class]
            info = disease_dic.get(disease_name, "No detailed information available.")

            return render_template(
                'disease-result.html',
                prediction=disease_name,
                info=info,
                image_path=filepath,
                title=title
            )

        except Exception as e:
            print("Prediction error:", e)
            return render_template('disease.html', title=title)

    return render_template('disease.html', title=title)


# -----------------------------------------------------
# ABOUT PAGE
# -----------------------------------------------------
@app.route('/about')
def about():
    title = 'AgriSense AI - About Project'
    return render_template('about.html', title=title)



# -----------------------------------------------------
# RUN APP
# -----------------------------------------------------
if __name__ == "__main__":
    app.run(debug=False)
