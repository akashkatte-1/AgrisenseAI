"""Run disease model predictions on images in app/static/uploads and save results.

Outputs a CSV `disease_predictions.csv` with image, top1, prob, and top5 list.
"""
import os
import csv
from PIL import Image
import torch
import numpy as np

from app.app import disease_classes, disease_transform, disease_model, APP_ROOT

uploads_dir = os.path.join(APP_ROOT, 'static', 'uploads')
out_path = os.path.join(APP_ROOT, 'disease_predictions.csv')

images = [f for f in os.listdir(uploads_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
if not images:
    print('No images found in', uploads_dir)
else:
    print('Found images:', images)

results = []
for img_name in images:
    img_path = os.path.join(uploads_dir, img_name)
    try:
        img = Image.open(img_path).convert('RGB')
        inp = disease_transform(img).unsqueeze(0)
        with torch.no_grad():
            out = disease_model(inp)
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]
            top_idx = np.argsort(probs)[::-1][:5]
            top5 = [(disease_classes[i], float(probs[i])) for i in top_idx]
            top1 = top5[0]
            print(f"{img_name} -> {top1[0]} ({top1[1]:.4f})")
            results.append({'image': img_name, 'top1': top1[0], 'top1_prob': top1[1], 'top5': ';'.join([f"{c}:{p:.4f}" for c,p in top5])})
    except Exception as e:
        print('Failed for', img_name, e)

if results:
    with open(out_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['image','top1','top1_prob','top5'])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print('Wrote predictions to', out_path)
