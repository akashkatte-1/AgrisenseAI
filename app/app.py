# Importing essential libraries and modules
from flask import Flask, render_template, request, redirect, url_for, flash, session
from markupsafe import Markup
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
import numpy as np
import pandas as pd
import os
import torch
from torchvision import transforms
from PIL import Image
import joblib
import requests

from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
from utils.model import ResNet9

# -----------------------------------------------------
# FLASK APP & SECRET KEY
# -----------------------------------------------------
app = Flask(__name__)
app.secret_key = 'agrisense-ai-secret-key-2025-rohit-rasale'  # Change this to a random secret key

# -----------------------------------------------------
# MONGODB CONNECTION
# -----------------------------------------------------
MONGO_URI = "mongodb+srv://rohit:rohit@cluster0.zeovrje.mongodb.net/?appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client['agrisense_db']

# Collections
users_collection = db['users']
crop_predictions_collection = db['crop_predictions']
fertilizer_predictions_collection = db['fertilizer_predictions']
disease_predictions_collection = db['disease_predictions']

# -----------------------------------------------------
# FLASK-LOGIN SETUP
# -----------------------------------------------------
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'

# User class for Flask-Login
class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data['_id'])
        self.name = user_data['name']
        self.email = user_data['email']
        self.is_admin = user_data.get('is_admin', False)

@login_manager.user_loader
def load_user(user_id):
    user_data = users_collection.find_one({'_id': ObjectId(user_id)})
    if user_data:
        return User(user_data)
    return None

# -----------------------------------------------------
# SEED ADMIN USER (Run once)
# -----------------------------------------------------
def seed_admin():
    admin_exists = users_collection.find_one({'email': 'admin@agrisense.com'})
    if not admin_exists:
        admin_data = {
            'name': 'Admin',
            'email': 'admin@agrisense.com',
            'password': generate_password_hash('admin123'),
            'is_admin': True,
            'created_at': datetime.utcnow()
        }
        users_collection.insert_one(admin_data)
        print("Admin user created successfully!")
        print("Email: admin@agrisense.com")
        print("Password: admin123")

# Seed admin on app start
seed_admin()

# -----------------------------------------------------
# LOAD DISEASE MODEL ONCE
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

disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
disease_model.eval()

disease_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
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
# AUTHENTICATION ROUTES
# -----------------------------------------------------
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Check if user already exists
        existing_user = users_collection.find_one({'email': email})
        if existing_user:
            flash('Email already registered. Please login.', 'danger')
            return redirect(url_for('signup'))
        
        # Create new user
        hashed_password = generate_password_hash(password)
        user_data = {
            'name': name,
            'email': email,
            'password': hashed_password,
            'is_admin': False,
            'created_at': datetime.utcnow()
        }
        users_collection.insert_one(user_data)
        
        flash('Account created successfully! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('signup.html', title='Sign Up - AgriSense AI')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user_data = users_collection.find_one({'email': email})
        
        if user_data and check_password_hash(user_data['password'], password):
            user = User(user_data)
            login_user(user)
            flash(f'Welcome back, {user.name}!', 'success')
            
            # Redirect to admin panel if admin
            if user.is_admin:
                return redirect(url_for('admin_dashboard'))
            
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password.', 'danger')
    
    return render_template('login.html', title='Login - AgriSense AI')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('home'))

# -----------------------------------------------------
# USER DASHBOARD
# -----------------------------------------------------
@app.route('/dashboard')
@login_required
def dashboard():
    user_id = current_user.id
    
    # Get user's prediction history
    crop_history = list(crop_predictions_collection.find({'user_id': user_id}).sort('timestamp', -1).limit(10))
    fertilizer_history = list(fertilizer_predictions_collection.find({'user_id': user_id}).sort('timestamp', -1).limit(10))
    disease_history = list(disease_predictions_collection.find({'user_id': user_id}).sort('timestamp', -1).limit(10))
    
    return render_template('dashboard.html', 
                          title='Dashboard - AgriSense AI',
                          crop_history=crop_history,
                          fertilizer_history=fertilizer_history,
                          disease_history=disease_history)

# -----------------------------------------------------
# ADMIN DASHBOARD
# -----------------------------------------------------
@app.route('/admin')
@login_required
def admin_dashboard():
    if not current_user.is_admin:
        flash('Access denied. Admin only.', 'danger')
        return redirect(url_for('home'))
    
    # Get all users
    all_users = list(users_collection.find({'is_admin': False}))
    
    # Get statistics
    total_users = len(all_users)
    total_crop_predictions = crop_predictions_collection.count_documents({})
    total_fertilizer_predictions = fertilizer_predictions_collection.count_documents({})
    total_disease_predictions = disease_predictions_collection.count_documents({})
    
    return render_template('admin_dashboard.html',
                          title='Admin Dashboard - AgriSense AI',
                          users=all_users,
                          total_users=total_users,
                          total_crop_predictions=total_crop_predictions,
                          total_fertilizer_predictions=total_fertilizer_predictions,
                          total_disease_predictions=total_disease_predictions)

@app.route('/admin/user/<user_id>')
@login_required
def admin_user_details(user_id):
    if not current_user.is_admin:
        flash('Access denied. Admin only.', 'danger')
        return redirect(url_for('home'))
    
    # Get user details
    user = users_collection.find_one({'_id': ObjectId(user_id)})
    if not user:
        flash('User not found.', 'danger')
        return redirect(url_for('admin_dashboard'))
    
    # Get user's prediction history
    crop_history = list(crop_predictions_collection.find({'user_id': user_id}).sort('timestamp', -1))
    fertilizer_history = list(fertilizer_predictions_collection.find({'user_id': user_id}).sort('timestamp', -1))
    disease_history = list(disease_predictions_collection.find({'user_id': user_id}).sort('timestamp', -1))
    
    return render_template('admin_user_details.html',
                          title=f"User Details - {user['name']}",
                          user=user,
                          crop_history=crop_history,
                          fertilizer_history=fertilizer_history,
                          disease_history=disease_history)

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
@login_required
def crop_recommend():
    title = 'AgriSense AI - Crop Recommendation'
    return render_template('crop.html', title=title)

@app.route('/crop-predict', methods=['POST'])
@login_required
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
    
    # Save prediction to MongoDB
    prediction_data = {
        'user_id': current_user.id,
        'user_name': current_user.name,
        'user_email': current_user.email,
        'nitrogen': N,
        'phosphorous': P,
        'potassium': K,
        'ph': ph,
        'rainfall': rainfall,
        'city': city,
        'temperature': temp,
        'humidity': humidity,
        'prediction': prediction,
        'timestamp': datetime.utcnow()
    }
    crop_predictions_collection.insert_one(prediction_data)

    return render_template('crop-result.html',
                           prediction=prediction,
                           title=title)

# -----------------------------------------------------
# FERTILIZER RECOMMENDATION PAGE
# -----------------------------------------------------
@app.route('/fertilizer')
@login_required
def fertilizer_recommendation():
    title = 'AgriSense AI - Fertilizer Suggestion'
    return render_template('fertilizer.html', title=title)

@app.route('/fertilizer-predict', methods=['POST'])
@login_required
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
    
    # Save prediction to MongoDB
    prediction_data = {
        'user_id': current_user.id,
        'user_name': current_user.name,
        'user_email': current_user.email,
        'crop_name': crop_name,
        'nitrogen': N,
        'phosphorous': P,
        'potassium': K,
        'recommendation': key,
        'timestamp': datetime.utcnow()
    }
    fertilizer_predictions_collection.insert_one(prediction_data)

    return render_template('fertilizer-result.html',
                           recommendation=response,
                           title=title)

# -----------------------------------------------------
# DISEASE PREDICTION PAGE
# -----------------------------------------------------
@app.route('/disease-predict', methods=['GET', 'POST'])
@login_required
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

            img_t = disease_transform(image)
            batch_t = torch.unsqueeze(img_t, 0)

            with torch.no_grad():
                output = disease_model(batch_t)
                _, predicted = torch.max(output, 1)
                predicted_class = predicted.item()

            disease_name = disease_classes[predicted_class]
            info = disease_dic.get(disease_name, "No detailed information available.")
            
            # Save prediction to MongoDB
            prediction_data = {
                'user_id': current_user.id,
                'user_name': current_user.name,
                'user_email': current_user.email,
                'image_filename': file.filename,
                'disease_name': disease_name,
                'timestamp': datetime.utcnow()
            }
            disease_predictions_collection.insert_one(prediction_data)

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
