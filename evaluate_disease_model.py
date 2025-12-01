"""Evaluate the disease classification model.

Usage examples:
  # Evaluate on an ImageFolder-style dataset (root/class_name/*.jpg)
  python evaluate_disease_model.py --data DATA_ROOT

  # Run single-image prediction with top-5 probabilities
  python evaluate_disease_model.py --image app/static/uploads/peruleaf.jpg

Outputs a classification report and confusion matrix when running on a labeled dataset.
"""
import os
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from app.utils.model import ResNet9

APP_ROOT = os.path.dirname(__file__)
MODEL_PATH = os.path.join(APP_ROOT, 'app', 'models', 'plant_disease_model.pth')

# Inference transform: match training transform (resize + to tensor)
INF_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


def load_model(device='cpu'):
    # number of classes is not known until dataset is loaded; we'll lazy-load when needed
    # but create a placeholder with 38 classes (matches training) and then load state_dict
    # If the saved state_dict needs a specific num_classes, ensure the classifier size matches.
    # We'll load the state_dict to a model with a matching output size.
    state = torch.load(MODEL_PATH, map_location='cpu')
    # state is a dict of parameters (state_dict) so we need to infer num_classes from classifier weight
    # find the Linear weight param and get its out_features
    lin_keys = [k for k in state.keys() if 'classifier' in k and 'weight' in k]
    if lin_keys:
        # key example: 'classifier.2.weight'
        weight = state[lin_keys[0]]
        out_features = weight.shape[0]
    else:
        out_features = 38

    model = ResNet9(3, out_features)
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate_dataset(data_root, batch_size=16, device='cpu'):
    dataset = ImageFolder(data_root, transform=INF_TRANSFORM)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = load_model(device=device)
    model.to(device)

    y_true = []
    y_pred = []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.tolist())
            y_pred.extend(preds.cpu().tolist())

    target_names = dataset.classes
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print('\nClassification report:')
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

    print('\nConfusion matrix:')
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    # Save a small CSV of mispredictions
    out_dir = os.path.join(APP_ROOT, 'evaluation_output')
    os.makedirs(out_dir, exist_ok=True)
    mis_path = os.path.join(out_dir, 'misclassified.txt')
    with open(mis_path, 'w', encoding='utf-8') as f:
        for idx, (true, pred) in enumerate(zip(y_true, y_pred)):
            if true != pred:
                img_path, _ = dataset.samples[idx]
                f.write(f"{img_path}\t{dataset.classes[true]}\t{dataset.classes[pred]}\n")
    print(f"Saved misclassified list to {mis_path}")


def predict_image(img_path, topk=5, device='cpu'):
    model = load_model(device=device)
    model.to(device)

    img = Image.open(img_path).convert('RGB')
    inp = INF_TRANSFORM(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(inp)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
        top_idx = np.argsort(probs)[::-1][:topk]

    # We don't know dataset classes here; print top indices and probs
    print(f'Top-{topk} predictions (index:prob):')
    for i in top_idx:
        print(f'{i}: {probs[i]:.4f}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', help='Path to ImageFolder-style dataset root (labeled).')
    p.add_argument('--image', help='Path to single image to predict (shows top-k indices).')
    p.add_argument('--topk', type=int, default=5)
    args = p.parse_args()

    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f'Model not found at {MODEL_PATH}.')

    if args.data:
        evaluate_dataset(args.data)
    elif args.image:
        predict_image(args.image, topk=args.topk)
    else:
        print('Specify --data DIR (for labeled eval) or --image FILE (single prediction).')
