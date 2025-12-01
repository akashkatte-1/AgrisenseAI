"""Retrain crop recommendation RandomForest model and save it where the app expects it.

Run with: python retrain_crop_model.py
"""
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

ROOT = os.path.dirname(__file__)
# Prefer processed dataset if present
candidates = [
    os.path.join(ROOT, 'Data-processed', 'crop_recommendation.csv'),
    os.path.join(ROOT, 'Data-raw', 'Crop_recommendation.csv'),
    os.path.join(ROOT, 'Data-raw', 'crop_recommendation.csv')
]

data_path = None
for p in candidates:
    if os.path.exists(p):
        data_path = p
        break

if data_path is None:
    raise SystemExit('Could not find crop recommendation dataset. Checked: ' + ', '.join(candidates))

print('Using dataset:', data_path)

data = pd.read_csv(data_path)
# Inspect columns
print('Columns:', list(data.columns))

# Expected feature columns
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
missing = [c for c in features if c not in data.columns]
if missing:
    raise SystemExit(f'Missing expected feature columns in dataset: {missing}')

if 'label' in data.columns:
    target = 'label'
elif 'crop' in data.columns:
    target = 'crop'
elif 'labelled' in data.columns:
    target = 'labelled'
else:
    # try to guess last column
    target = data.columns[-1]
    print('Using last column as target:', target)

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

out_dir = os.path.join(ROOT, 'app', 'models')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'RandomForest.pkl')
joblib.dump(model, out_path)

acc = model.score(X_test, y_test)
print(f'Model trained and saved to {out_path} (test accuracy: {acc:.4f})')
