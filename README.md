# AgriSense AI 🌿
#### A ML and DL based web application with user authentication system which recommends the best crop to grow, fertilizers to use and detects diseases from plant leaf images.

#### **NEW FEATURES:**
- ✅ User Authentication (Login/Signup with MongoDB)
- ✅ User Dashboard with Prediction History
- ✅ Admin Panel to manage users and monitor all activities
- ✅ All predictions stored in MongoDB database
- ✅ Secure password hashing with bcrypt



## DISCLAIMER ⚠️
This is a POC(Proof of concept) kind-of project. The data used here comes up with no guarantee from the creator. So, don't use it for making farming decisions. If you do so, the creator is not responsible for anything. However, this project presents the idea that how we can use ML/DL into precision farming if developed at large scale and with authentic and verified data.

## MOTIVATION 💪
- Farming is one of the major sectors that influences a country’s economic growth. 

- In country like India, majority of the population is dependent on agriculture for their livelihood. Many new technologies, such as Machine Learning and Deep Learning, are being implemented into agriculture so that it is easier for farmers to grow and maximize their yield. 

- In this project, I present a website in which the following applications are implemented; Crop recommendation, Fertilizer recommendation and Plant disease prediction, respectively. 

    - In the crop recommendation application, the user can provide the soil data from their side and the application will predict which crop should the user grow. 
    
    - For the fertilizer recommendation application, the user can input the soil data and the type of crop they are growing, and the application will predict what the soil lacks or has excess of and will recommend improvements. 
    
    - For the last application, that is the plant disease prediction application, the user can input an image of a diseased plant leaf, and the application will predict what disease it is and will also give a little background about the disease and suggestions to cure it.

## DATA SOURCE 📊
- [Crop recommendation dataset ](https://www.kaggle.com/atharvaingle/crop-recommendation-dataset) (custom built dataset)
- [Fertilizer suggestion dataset](https://github.com/Gladiator07/Harvestify/blob/master/Data-processed/fertilizer.csv) (custom built dataset)
- [Disease detection dataset](https://www.kaggle.com/vipoooool/new-plant-diseases-dataset)

## Notebooks 📓
##### I have also published the corresponding code on Kaggle Notebooks.
- [Crop Recommendation](https://www.kaggle.com/atharvaingle/what-crop-to-grow)
- [Disease Detection](https://www.kaggle.com/atharvaingle/plant-disease-classification-resnet-99-2)

# Built with 🛠️

### Backend
- **Flask** - Web framework
- **Flask-Login** - User session management
- **PyMongo** - MongoDB database driver
- **Bcrypt** - Password hashing
- **MongoDB Atlas** - Cloud database

### Machine Learning
- **PyTorch** - Deep learning framework for disease detection
- **scikit-learn** - Machine learning library for crop recommendation
- **NumPy & Pandas** - Data processing
- **Pillow** - Image processing

### Frontend
- **HTML/CSS/JavaScript**
- **Bootstrap 4** - Responsive design
- **Jinja2** - Template engine

<code><img height="30" src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/python/python.png"></code>
<code><img height="30" src="https://symbols.getvecta.com/stencil_80/56_flask.3a79b5a056.jpg"></code>
<code><img height="30" src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/mongodb/mongodb.png"></code>
<code><img height="30" src="https://raw.githubusercontent.com/pytorch/pytorch/39fa0b5d0a3b966a50dcd90b26e6c36942705d6d/docs/source/_static/img/pytorch-logo-dark.svg"></code>
<code><img height="30" src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/1280px-Scikit_learn_logo_small.svg.png"></code>
<code><img height="30" src="https://raw.githubusercontent.com/numpy/numpy/7e7f4adab814b223f7f917369a72757cd28b10cb/branding/icons/numpylogo.svg"></code>
<code><img height="30" src="https://raw.githubusercontent.com/pandas-dev/pandas/761bceb77d44aa63b71dda43ca46e8fd4b9d7422/web/pandas/static/img/pandas.svg"></code>
<code><img height="30" src="https://github.com/tomchen/stack-icons/raw/master/logos/bootstrap.svg"></code>
<code><img height="30" src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/html/html.png"></code>

## DEPLOYMENT 🚀

#### This website is deployed at [Heroku](https://www.heroku.com/)
#### Note: The website may take a minute to load sometimes, as the server may be in hibernate state

## How to use 💻
- Crop Recommendation system ==> enter the corresponding nutrient values of your soil, state and city. Note that, the N-P-K (Nitrogen-Phosphorous-Pottasium) values to be entered should be the ratio between them. Refer [this website](https://www.gardeningknowhow.com/garden-how-to/soil-fertilizers/fertilizer-numbers-npk.htm) for more information.
Note: When you enter the city name, make sure to enter mostly common city names. Remote cities/towns may not be available in the [Weather API](https://openweathermap.org/) from where humidity, temperature data is fetched.

- Fertilizer suggestion system ==> Enter the nutrient contents of your soil and the crop you want to grow. The algorithm will tell which nutrient the soil has excess of or lacks. Accordingly, it will give suggestions for buying fertilizers.

- Disease Detection System ==> Upload an image of leaf of your plant. The algorithm will tell the crop type and whether it is diseased or healthy. If it is diseased, it will tell you the cause of the disease and suggest you how to prevent/cure the disease accordingly.
Note that, for now it only supports following crops

<details>
  <summary>Supported crops
</summary>

- Apple
- Blueberry
- Cherry
- Corn
- Grape
- Pepper
- Orange
- Peach
- Potato
- Soybean
- Strawberry
- Tomato
- Squash
- Raspberry
</details>

## Authentication Features 🔐

### User Authentication
- **Signup:** Users can create an account with Name, Email, and Password
- **Login:** Secure login with email and password
- **Session Management:** Uses Flask-Login for secure session handling
- **Password Security:** All passwords are hashed using bcrypt

### User Dashboard
- View all your prediction history:
  - Crop Recommendations
  - Fertilizer Suggestions
  - Disease Detections
- Track your farming decisions over time
- Access personalized insights

### Admin Panel 👨‍💼
- **Admin Credentials:**
  - Email: `admin@agrisense.com`
  - Password: `admin123`
- **Admin Features:**
  - View all registered users
  - Monitor system statistics
  - View detailed user activity and prediction history
  - Track total predictions across all users

### MongoDB Integration 🍃
All user data and predictions are securely stored in MongoDB:
- **Users Collection:** Stores user authentication data
- **Crop Predictions Collection:** Stores all crop recommendation data
- **Fertilizer Predictions Collection:** Stores fertilizer suggestion data
- **Disease Predictions Collection:** Stores disease detection data

**Database Connection:** MongoDB Atlas Cloud Database
```
mongodb+srv://rohit:rohit@cluster0.zeovrje.mongodb.net/?appName=Cluster0
```

### Navigation System
- **Before Login:** Only Home and About pages are accessible
- **After Login:** Full access to all features (Crop Recommendation, Fertilizer Advice, Disease Detection, Dashboard)
- **Admin Access:** Special admin panel for system monitoring

## How to run locally 🛠️
- Before the following steps make sure you have [git](https://git-scm.com/download), [Python 3.6+](https://www.python.org/downloads/) installed on your system
- Clone this repository
- Navigate to the project directory
- Install required dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Run the application:
  ```bash
  cd app
  python app.py
  ```
  Or from the root directory:
  ```bash
  python run_server.py
  ```
- Open your web browser and go to `http://127.0.0.1:5000`
- **Admin Login:** Use email `admin@agrisense.com` and password `admin123` to access admin panel
- **Create User Account:** Click on "Sign Up" to create a new user account
## DEMO




## Usage ⚙️
You can use this project for further developing it and adding your work in it. If you use this project, kindly mention the original source of the project and mention the link of this repo in your report.

## Further Improvements 📈
This was my first big project so there are lot of things to improve upon

- CSS code is totally messed up :pensive: (some code in file and some inline)
- Frontend can be made more nicer (PS: I suck at frontend development) :cry:	
- More data can be collected manually via web scrapping to make the system more accurate :monocle_face:	
- Additional plant images can be collected to make the disease detection part more robust and generalized :face_with_head_bandage:
- Modularized code can be written instead of writing in Jupyter Notebooks (will follow this in upcoming projects)
