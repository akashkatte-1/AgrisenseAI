# 🌾 Train Crop Recommendation Model (compatible with scikit-learn 1.3.2)
# Author: Rohit Rasale

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1️⃣ Load dataset
data = pd.read_csv(r"C:\Users\Dell\Desktop\Harvestify\Data-raw\Crop_recommendation.csv")

# 2️⃣ Prepare data
X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = data['label']

# 3️⃣ Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4️⃣ Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5️⃣ Save model
joblib.dump(model, r"C:\Users\Dell\Desktop\Harvestify\models\crop_recommendation_model.pkl")

# 6️⃣ Show accuracy
accuracy = model.score(X_test, y_test)
print(f"✅ Model trained successfully! Accuracy: {accuracy:.2f}")
print('💾 Model saved to: C:\\Users\\Dell\\Desktop\\Harvestify\\models\\crop_recommendation_model.pkl')
