"""Smoke tests for POST endpoints: fertilizer and disease prediction.

Run with: python test_post_endpoints.py
"""
import os
from app.app import app

client = app.test_client()

# 1) Test fertilizer predict
fert_data = {
    'cropname': 'rice',
    'nitrogen': '50',
    'phosphorous': '30',
    'pottasium': '20'
}
resp = client.post('/fertilizer-predict', data=fert_data)
print('/fertilizer-predict ->', resp.status_code)
print(resp.get_data(as_text=True)[:1000])

# 2) Test disease predict - try to upload an existing sample image
# Search for an image in static/images or static/uploads
candidates = [
    os.path.join('app','static','images'),
    os.path.join('app','static','uploads'),
    os.path.join('app','static')
]
img_path = None
for d in candidates:
    if os.path.isdir(d):
        for fname in os.listdir(d):
            if fname.lower().endswith(('.png','.jpg','.jpeg')):
                img_path = os.path.join(d, fname)
                break
    if img_path:
        break

if img_path is None:
    print('No sample image found for disease test; skipping')
else:
    with open(img_path, 'rb') as f:
        data = {'file': (f, os.path.basename(img_path))}
        resp2 = client.post('/disease-predict', data=data, content_type='multipart/form-data')
        print('/disease-predict ->', resp2.status_code)
        print(resp2.get_data(as_text=True)[:1000])
