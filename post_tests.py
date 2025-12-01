"""Automated POST tests for fertilizer and disease endpoints using Flask test client.

Run with: python post_tests.py
"""
import io
import os
from app.app import app, APP_ROOT, MODEL_PATH

client = app.test_client()

# Test fertilizer POST
fert_data = {
    'cropname': 'rice',
    'nitrogen': '100',
    'phosphorous': '50',
    'pottasium': '50'
}
print('POST /fertilizer-predict')
resp = client.post('/fertilizer-predict', data=fert_data)
print('status:', resp.status_code)
print(resp.data.decode()[:1000])

# Prepare a tiny red square image for disease prediction
from PIL import Image
img = Image.new('RGB', (256, 256), color=(255,0,0))
img_bytes = io.BytesIO()
img.save(img_bytes, format='JPEG')
img_bytes.seek(0)

print('\nPOST /disease-predict (image upload)')
resp2 = client.post('/disease-predict', data={'file': (img_bytes, 'test.jpg')}, content_type='multipart/form-data')
print('status:', resp2.status_code)
print(resp2.data.decode()[:1000])
