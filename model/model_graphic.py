# Fonetic and semantic similarity model for trademark data
import subprocess
import jellyfish

# Artificial Intelligence liberaries
import torch
import clip
from PIL import Image
from sentence_transformers.util import cos_sim
from io import BytesIO

# Data processing libraries
import pickle
import numpy as np
import pandas as pd

# Os and Path libraries
import subprocess
from dotenv import load_dotenv
from pathlib import Path
import os
import requests


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def get_image_from_url(url):
    """Pobierz obraz z URL i przygotuj do modelu"""
    response = requests.get(url)
    
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
    else:
        print("Nie udało się pobrać obrazu")

    image = Image.open(BytesIO(response.content)).convert("RGB")
    return preprocess(image).unsqueeze(0).to(device)

def compare_images_clip_cos_sim(url1, url2):
    """Porównaj dwa obrazy CLIP-em i oblicz podobieństwo z cos_sim (z sentence_transformers)"""
    image1 = get_image_from_url(url1)
    image2 = get_image_from_url(url2)

    with torch.no_grad():
        embedding1 = model.encode_image(image1)
        embedding2 = model.encode_image(image2)

    similarity = cos_sim(embedding1, embedding2).item()
    return similarity



